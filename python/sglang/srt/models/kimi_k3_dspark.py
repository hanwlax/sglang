from collections.abc import Iterable

import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.dspark import DSparkDraftModel


class K3DSparkModel(DSparkDraftModel):
    """Kimi K3's MLA draft with its own checkpoint vocabulary modules."""

    uses_own_vocab_modules = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config, prefix)
        if config.target_hidden_size != config.hidden_size:
            raise ValueError(
                "Kimi K3 DSpark requires matching target and draft widths."
            )
        if self.num_context_features != config.num_target_layers:
            raise ValueError(
                "Kimi K3 DSpark target layer IDs must match num_target_layers."
            )
        if config.draft_vocab_size != config.vocab_size:
            raise ValueError(
                "Kimi K3 DSpark requires matching target and draft vocabularies."
            )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        aliases = {
            "context_proj.weight": "fc.weight",
            "context_norm.weight": "hidden_norm.weight",
            "final_norm.weight": "norm.weight",
        }
        vocab_params = {
            "embed_tokens.weight": self.embed_tokens.weight,
            "lm_head.weight": self.lm_head.weight,
        }
        loaded_vocab = set()

        def backbone_weights():
            for name, weight in weights:
                name = name.removeprefix("model.")
                param = vocab_params.get(name)
                if param is None:
                    yield aliases.get(name, name), weight
                    continue
                if name in loaded_vocab:
                    raise ValueError(f"Duplicate Kimi K3 DSpark weight {name}.")
                param.weight_loader(param, weight)
                loaded_vocab.add(name)

        super().load_weights(backbone_weights())
        missing = vocab_params.keys() - loaded_vocab
        if missing:
            raise ValueError(f"Kimi K3 DSpark checkpoint is missing {sorted(missing)}.")


EntryClass = K3DSparkModel
