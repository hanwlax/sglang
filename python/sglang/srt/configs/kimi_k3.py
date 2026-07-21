"""
Kimi K3 model configuration.
"""

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.kimi_linear import KimiLinearConfig


class KimiK3VisionConfig(PretrainedConfig):
    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple[int, int] = (2, 2),
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        text_hidden_size: int = 7168,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            vt_hidden_size if mm_hidden_size is None else mm_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.text_hidden_size = text_hidden_size


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"
    sub_configs = {
        "text_config": KimiLinearConfig,
        "vision_config": KimiK3VisionConfig,
    }

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        language_only: bool = True,
        encoder_only: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["KimiK3ForConditionalGeneration"])

        if text_config is None:
            text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)

        if vision_config is None:
            vision_config = KimiK3VisionConfig(
                text_hidden_size=getattr(text_config, "hidden_size", 7168)
            )
        elif isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.image_placeholder = image_placeholder
        self.media_placeholder_token_id = media_placeholder_token_id
        self.language_only = language_only
        self.encoder_only = encoder_only

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self._sync_text_token_ids()

        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

    def _sync_text_token_ids(self) -> None:
        default_token_ids = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        for attr, default_value in default_token_ids.items():
            parent_value = getattr(self, attr, None)
            text_value = getattr(self.text_config, attr, None)
            if parent_value is not None and (
                text_value is None or text_value == default_value
            ):
                setattr(self.text_config, attr, parent_value)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.text_config.linear_layer_ids

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return self.text_config.full_attention_layer_ids

    @property
    def mamba2_cache_params(self):
        return self.text_config.mamba2_cache_params

    @property
    def is_mla(self) -> bool:
        return self.text_config.is_mla

    @property
    def is_moe(self) -> bool:
        return self.text_config.is_moe

    @property
    def is_linear_attn(self) -> bool:
        return self.text_config.is_linear_attn

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self.text_config.is_kda_layer(layer_idx)

    def get_text_config(self, *args, **kwargs) -> KimiLinearConfig:
        return self.text_config
