from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.models.kimi_linear import KimiLinearForCausalLM


def _is_kimi_k3_vision_weight(name: str) -> bool:
    vision_prefixes = (
        "vision_tower.",
        "model.vision_tower.",
        "mm_projector.",
        "model.mm_projector.",
        "multi_modal_projector.",
        "model.multi_modal_projector.",
    )
    return name.startswith(vision_prefixes)


def _strip_kimi_k3_language_prefix(name: str) -> str:
    language_prefixes = (
        "model.language_model.",
        "language_model.",
    )
    for prefix in language_prefixes:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith(("layers.", "embed_tokens.", "norm.")):
        return "model." + name
    return name


class KimiK3ForConditionalGeneration(nn.Module):
    """Minimal native Kimi-K3 wrapper.

    The first NPU milestone is text-only: Kimi-K3's text_config is KimiLinear,
    and ModelSlim conversion targets the language weights. Vision loading can be
    added once the NPU text path is validated.
    """

    packed_modules_mapping = KimiLinearForCausalLM.packed_modules_mapping

    def __init__(
        self,
        config: KimiK3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        language_prefix = (
            "language_model" if prefix == "" else f"{prefix}.language_model"
        )
        self.language_model = KimiLinearForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=language_prefix,
        )

    @property
    def model(self):
        return self.language_model

    def __setattr__(self, name, value):
        if name == "model":
            return
        super().__setattr__(name, value)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if get_embedding:
            raise NotImplementedError("Kimi-K3 embedding mode is not implemented yet.")

        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[tuple]):
        def stream_language_weights():
            for args in weights:
                name, loaded_weight = args[:2]
                extra_args = args[2:]
                if _is_kimi_k3_vision_weight(name):
                    continue
                yield (
                    _strip_kimi_k3_language_prefix(name),
                    loaded_weight,
                    *extra_args,
                )

        self.language_model.load_weights(stream_language_weights())

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer

    @property
    def end_layer(self) -> int:
        return self.language_model.end_layer

    @property
    def routed_experts_weights_of_layer(self):
        routed_experts_weights = getattr(
            self.language_model, "_routed_experts_weights_of_layer", None
        )
        if routed_experts_weights is not None:
            return routed_experts_weights.value
        return getattr(self.language_model, "routed_experts_weights_of_layer", {})

    def post_load_weights(self):
        if hasattr(self.language_model, "post_load_weights"):
            self.language_model.post_load_weights()

    @property
    def stacked_params_mapping(self):
        return getattr(self.language_model, "stacked_params_mapping", [])

    @property
    def expert_params_mapping(self):
        return getattr(self.language_model, "expert_params_mapping", [])

    def mutate_weight_preload(self, name):
        return self.language_model.mutate_weight_preload(name)

    def custom_scale_remap(self, name):
        return self.language_model.custom_scale_remap(name)

    @classmethod
    def get_model_config_for_expert_location(cls, config: KimiK3Config):
        text_config = config.text_config
        num_experts = getattr(text_config, "n_routed_experts", None)
        if num_experts is None:
            num_experts = getattr(text_config, "num_experts", None)
        if num_experts is None:
            return None

        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=getattr(text_config, "n_group", None),
        )

    @property
    def lm_head(self):
        return self.language_model.lm_head

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def get_num_kv_cache_layers(self) -> int:
        return self.config.text_config.num_hidden_layers

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.language_model.model.embed_tokens.weight,
            self.language_model.lm_head.weight,
        )

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.language_model.model.embed_tokens.weight
        del self.language_model.lm_head.weight
        self.language_model.model.embed_tokens.weight = embed
        self.language_model.lm_head.weight = head


EntryClass = [KimiK3ForConditionalGeneration]
