"""Unit tests for ``sglang.srt.configs.kimi_k3.KimiK3Config``."""

import json
import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs import KimiK3Config, KimiK3VisionConfig
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiK3Config(CustomTestCase):
    def test_defaults_are_text_only_kimi_linear(self):
        cfg = KimiK3Config()

        self.assertEqual(cfg.model_type, "kimi_k3")
        self.assertTrue(cfg.language_only)
        self.assertEqual(cfg.architectures, ["KimiK3ForConditionalGeneration"])
        self.assertIsInstance(cfg.text_config, KimiLinearConfig)
        self.assertIsInstance(cfg.vision_config, KimiK3VisionConfig)
        self.assertIs(cfg.get_text_config(), cfg.text_config)
        self.assertIs(cfg.get_text_config(decoder=True), cfg.text_config)

    def test_subconfigs_from_dict(self):
        cfg = KimiK3Config(
            text_config={
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_hidden_layers": 4,
            },
            vision_config={"vt_hidden_size": 512, "mm_hidden_size": 768},
        )

        self.assertIsInstance(cfg.text_config, KimiLinearConfig)
        self.assertIsInstance(cfg.vision_config, KimiK3VisionConfig)
        self.assertEqual(cfg.text_config.hidden_size, 2048)
        self.assertEqual(cfg.vision_config.vt_hidden_size, 512)
        self.assertEqual(cfg.vision_config.mm_hidden_size, 768)

    def test_top_level_token_ids_are_synced_to_text_config(self):
        cfg = KimiK3Config(
            pad_token_id=163839,
            bos_token_id=163584,
            eos_token_id=163586,
        )

        self.assertEqual(cfg.pad_token_id, 163839)
        self.assertEqual(cfg.text_config.pad_token_id, 163839)
        self.assertEqual(cfg.bos_token_id, 163584)
        self.assertEqual(cfg.text_config.bos_token_id, 163584)
        self.assertEqual(cfg.eos_token_id, 163586)
        self.assertEqual(cfg.text_config.eos_token_id, 163586)

    def test_explicit_text_token_ids_are_preserved(self):
        cfg = KimiK3Config(
            pad_token_id=163839,
            text_config={"pad_token_id": 7, "bos_token_id": 8, "eos_token_id": 9},
        )

        self.assertEqual(cfg.pad_token_id, 163839)
        self.assertEqual(cfg.text_config.pad_token_id, 7)
        self.assertEqual(cfg.text_config.bos_token_id, 8)
        self.assertEqual(cfg.text_config.eos_token_id, 9)

    def test_text_quantization_config_is_promoted(self):
        quantization_config = {
            "quant_method": "modelslim",
            "quantization": {"quant_algo": "W8A8_DYNAMIC"},
        }
        cfg = KimiK3Config(
            text_config={"quantization_config": quantization_config}
        )

        self.assertEqual(cfg.quantization_config, quantization_config)

    def test_registered_in_config_registry(self):
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

        self.assertIs(_CONFIG_REGISTRY.get("kimi_k3"), KimiK3Config)

    def test_hf_parser_loads_kimi_k3_without_transformers_auto_config(self):
        from sglang.srt.utils.hf_transformers.config import HfModelConfigParser

        with TemporaryDirectory() as tmpdir:
            KimiK3Config(pad_token_id=163839).save_pretrained(tmpdir)
            override_file = "alt_config.json"
            with open(Path(tmpdir) / override_file, "w", encoding="utf-8") as f:
                json.dump(KimiK3Config(pad_token_id=42).to_dict(), f)

            with patch(
                "sglang.srt.utils.hf_transformers.config.AutoConfig.from_pretrained",
                side_effect=AssertionError("AutoConfig should not load kimi_k3"),
            ), patch.object(
                KimiK3Config,
                "from_pretrained",
                wraps=KimiK3Config.from_pretrained,
            ) as from_pretrained:
                cfg = HfModelConfigParser().parse(
                    tmpdir,
                    trust_remote_code=False,
                    _configuration_file=override_file,
                )

        self.assertIsInstance(cfg, KimiK3Config)
        self.assertEqual(cfg.pad_token_id, 42)
        self.assertEqual(cfg.text_config.pad_token_id, 42)
        self.assertEqual(from_pretrained.call_count, 1)
        self.assertEqual(
            from_pretrained.call_args.kwargs["_configuration_file"], override_file
        )

    def test_registered_in_model_registry(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
        from sglang.srt.models.registry import ModelRegistry

        model_cls, arch = ModelRegistry.resolve_model_cls(
            ["KimiK3ForConditionalGeneration"]
        )

        self.assertEqual(arch, "KimiK3ForConditionalGeneration")
        self.assertIs(model_cls, KimiK3ForConditionalGeneration)

    def test_model_wrapper_delegates_runtime_interfaces_to_language_model(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        fake_language_model = SimpleNamespace(
            start_layer=1,
            end_layer=4,
            _routed_experts_weights_of_layer=SimpleNamespace(value={2: "expert"}),
            stacked_params_mapping=[("gate_up_proj", "gate_proj")],
            expert_params_mapping=[("experts", "gate_proj")],
            post_load_weights=lambda: setattr(fake_language_model, "post_loaded", True),
            mutate_weight_preload=lambda name: f"mutated:{name}",
            custom_scale_remap=lambda name: f"remapped:{name}",
        )

        with patch(
            "sglang.srt.models.kimi_k3.KimiLinearForCausalLM",
            return_value=fake_language_model,
        ) as kimi_linear_cls:
            model = KimiK3ForConditionalGeneration(KimiK3Config())

        self.assertEqual(kimi_linear_cls.call_args.kwargs["prefix"], "language_model")
        self.assertEqual(model.start_layer, 1)
        self.assertEqual(model.end_layer, 4)
        self.assertEqual(model.routed_experts_weights_of_layer, {2: "expert"})
        self.assertEqual(
            model.stacked_params_mapping, [("gate_up_proj", "gate_proj")]
        )
        self.assertEqual(model.expert_params_mapping, [("experts", "gate_proj")])
        self.assertEqual(model.mutate_weight_preload("w"), "mutated:w")
        self.assertEqual(model.custom_scale_remap("scale"), "remapped:scale")

        model.post_load_weights()
        self.assertTrue(fake_language_model.post_loaded)

    def test_model_wrapper_reports_text_config_for_expert_location(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        cfg = KimiK3Config(
            text_config={
                "layers_block_type": ["mamba", "moe", "attention", "moe"],
                "n_routed_experts": 16,
                "n_group": 4,
            }
        )

        expert_config = KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
            cfg
        )

        self.assertEqual(expert_config.num_layers, 4)
        self.assertEqual(expert_config.num_logical_experts, 16)
        self.assertEqual(expert_config.num_groups, 4)

    def test_model_wrapper_exposes_kimi_linear_packed_modules_mapping(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
        from sglang.srt.models.kimi_linear import KimiLinearForCausalLM

        self.assertIs(
            KimiK3ForConditionalGeneration.packed_modules_mapping,
            KimiLinearForCausalLM.packed_modules_mapping,
        )
        model_mapping = KimiK3ForConditionalGeneration.packed_modules_mapping["model"]
        self.assertEqual(
            model_mapping["fused_qkvbfg_a_proj"],
            ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
        )
        self.assertEqual(model_mapping["fused_fg_b_proj"], ["f_b_proj", "g_b_proj"])

    def test_model_config_defaults_to_text_only(self):
        from sglang.srt.configs.model_config import MM_DISABLED_MODEL_ARCHS

        self.assertIn("KimiK3ForConditionalGeneration", MM_DISABLED_MODEL_ARCHS)

    def test_arg_overrides_treat_kimi_k3_as_mamba_cache_model(self):
        from sglang.srt.arg_groups.overrides import (
            _MAMBA_EXTRA_BUFFER_ARCHS,
            _MAMBA_RADIX_CACHE_ARCHS,
            supports_mamba_cache_extra_buffer,
        )

        self.assertIn("KimiK3ForConditionalGeneration", _MAMBA_RADIX_CACHE_ARCHS)
        self.assertIn("KimiK3ForConditionalGeneration", _MAMBA_EXTRA_BUFFER_ARCHS)
        self.assertTrue(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(linear_attn_backend="triton"),
                "KimiK3ForConditionalGeneration",
            )
        )
        self.assertFalse(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(linear_attn_backend="flashinfer"),
                "KimiK3ForConditionalGeneration",
            )
        )

    def test_model_config_uses_text_config_for_kimi_k3_attention_params(self):
        from sglang.srt.configs.model_config import AttentionArch, ModelConfig

        cfg = KimiK3Config(
            text_config={
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128,
                "v_head_dim": 96,
            }
        )
        model_config = object.__new__(ModelConfig)
        model_config.hf_text_config = cfg.text_config

        ModelConfig._apply_kimi_linear_attention_config(model_config)

        self.assertEqual(model_config.head_dim, 72)
        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.qk_nope_head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 96)
        self.assertAlmostEqual(model_config.scaling, 1 / (192**0.5))

    def test_hybrid_arch_uses_text_config_for_kimi_linear_cache(self):
        from sglang.srt.configs.hybrid_arch import kimi_linear_config

        cfg = KimiK3Config()
        self.assertIs(kimi_linear_config(SimpleNamespace(hf_config=cfg)), cfg.text_config)

    def test_text_config_preserves_kimi_linear_attention_layout(self):
        cfg = KimiK3Config(
            text_config={
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "kv_lora_rank": 128,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 96,
                "num_experts": 16,
                "linear_attn_config": {
                    "kda_layers": [1, 3],
                    "full_attn_layers": [2, 4],
                    "num_heads": 8,
                    "head_dim": 64,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                },
            }
        )

        self.assertTrue(cfg.text_config.is_linear_attn)
        self.assertTrue(cfg.is_linear_attn)
        self.assertTrue(cfg.is_mla)
        self.assertTrue(cfg.is_moe)
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(cfg.num_attention_heads, 8)
        self.assertEqual(cfg.num_key_value_heads, 8)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.text_config.linear_layer_ids, [0, 2])
        self.assertEqual(cfg.linear_layer_ids, [0, 2])
        self.assertEqual(cfg.text_config.full_attention_layer_ids, [1, 3])
        self.assertEqual(cfg.full_attention_layer_ids, [1, 3])
        self.assertTrue(cfg.is_kda_layer(0))
        self.assertFalse(cfg.is_kda_layer(1))
        self.assertEqual(cfg.text_config.linear_attn_config["gate_lower_bound"], -5.0)

    def test_representative_local_config_json_fields_are_preserved(self):
        quantization_config = {
            "config_groups": {},
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
        }
        cfg = KimiK3Config(
            architectures=["KimiK3ForConditionalGeneration"],
            pad_token_id=163839,
            media_placeholder_token_id=163605,
            text_config={
                "model_type": "kimi_linear",
                "hidden_size": 7168,
                "vocab_size": 163840,
                "num_hidden_layers": 93,
                "pad_token_id": 0,
                "quantization_config": quantization_config,
                "linear_attn_config": {
                    "kda_layers": [1],
                    "full_attn_layers": [2],
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                },
            },
            vision_config={
                "vt_hidden_size": 1024,
                "mm_hidden_size": 1024,
                "merge_kernel_size": [2, 2],
            },
        )

        self.assertEqual(cfg.architectures, ["KimiK3ForConditionalGeneration"])
        self.assertEqual(cfg.pad_token_id, 163839)
        self.assertEqual(cfg.text_config.pad_token_id, 163839)
        self.assertEqual(cfg.media_placeholder_token_id, 163605)
        self.assertEqual(cfg.quantization_config, quantization_config)
        self.assertEqual(cfg.text_config.linear_attn_config["gate_lower_bound"], -5.0)
        self.assertEqual(cfg.vision_config.mm_hidden_size, 1024)

    def test_weight_name_helpers_route_language_weights(self):
        from sglang.srt.models.kimi_k3 import (
            _is_kimi_k3_vision_weight,
            _strip_kimi_k3_language_prefix,
        )

        self.assertEqual(
            _strip_kimi_k3_language_prefix("language_model.model.layers.0.weight"),
            "model.layers.0.weight",
        )
        self.assertEqual(
            _strip_kimi_k3_language_prefix(
                "model.language_model.model.layers.0.weight"
            ),
            "model.layers.0.weight",
        )
        self.assertEqual(
            _strip_kimi_k3_language_prefix("language_model.layers.0.weight"),
            "model.layers.0.weight",
        )
        self.assertEqual(
            _strip_kimi_k3_language_prefix(
                "model.language_model.layers.0.weight"
            ),
            "model.layers.0.weight",
        )
        self.assertEqual(
            _strip_kimi_k3_language_prefix("language_model.lm_head.weight"),
            "lm_head.weight",
        )
        self.assertEqual(
            _strip_kimi_k3_language_prefix("model.lm_head.weight"),
            "lm_head.weight",
        )
        self.assertTrue(_is_kimi_k3_vision_weight("vision_tower.patch_embed.weight"))
        self.assertTrue(_is_kimi_k3_vision_weight("model.mm_projector.0.weight"))
        self.assertFalse(
            _is_kimi_k3_vision_weight("language_model.model.embed_tokens.weight")
        )

    def test_model_wrapper_load_weights_streams_language_weights_with_extra_args(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        class FakeLanguageModel:
            def load_weights(self, weights):
                self.loaded = list(weights)

        fake_language_model = FakeLanguageModel()
        with patch(
            "sglang.srt.models.kimi_k3.KimiLinearForCausalLM",
            return_value=fake_language_model,
        ):
            model = KimiK3ForConditionalGeneration(KimiK3Config())

        model.load_weights(
            [
                ("vision_tower.patch_embed.weight", "vision"),
                ("language_model.model.layers.0.mlp.gate_proj.weight", "w0", "meta"),
                ("model.language_model.layers.1.mlp.up_proj.weight", "w1"),
                ("model.lm_head.weight", "head", "head_meta"),
            ]
        )

        self.assertEqual(
            fake_language_model.loaded,
            [
                ("model.layers.0.mlp.gate_proj.weight", "w0", "meta"),
                ("model.layers.1.mlp.up_proj.weight", "w1"),
                ("lm_head.weight", "head", "head_meta"),
            ],
        )

    def test_model_wrapper_custom_prefix_only_affects_construction_prefix(self):
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        class FakeLanguageModel:
            def load_weights(self, weights):
                self.loaded = list(weights)

        fake_language_model = FakeLanguageModel()
        with patch(
            "sglang.srt.models.kimi_k3.KimiLinearForCausalLM",
            return_value=fake_language_model,
        ) as kimi_linear_cls:
            model = KimiK3ForConditionalGeneration(KimiK3Config(), prefix="root")

        self.assertEqual(
            kimi_linear_cls.call_args.kwargs["prefix"], "root.language_model"
        )

        model.load_weights(
            [
                ("model.language_model.model.layers.0.self_attn.o_proj.weight", "w"),
                ("model.language_model.lm_head.weight", "head"),
            ]
        )

        self.assertEqual(
            fake_language_model.loaded,
            [
                ("model.layers.0.self_attn.o_proj.weight", "w"),
                ("lm_head.weight", "head"),
            ],
        )

    def test_modelslim_prefix_candidates_cover_text_only_export(self):
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        candidates = ModelSlimConfig._prefix_candidates(
            "language_model.model.layers.0.mlp.gate_proj"
        )

        self.assertIn("language_model.model.layers.0.mlp.gate_proj", candidates)
        self.assertIn("language_model.layers.0.mlp.gate_proj", candidates)
        self.assertIn("model.layers.0.mlp.gate_proj", candidates)
        self.assertIn("model.language_model.layers.0.mlp.gate_proj", candidates)
        self.assertIn("model.language_model.model.layers.0.mlp.gate_proj", candidates)

        short_language_candidates = ModelSlimConfig._prefix_candidates(
            "language_model.layers.0.mlp.gate_proj"
        )

        self.assertIn(
            "language_model.model.layers.0.mlp.gate_proj",
            short_language_candidates,
        )
        self.assertIn("model.layers.0.mlp.gate_proj", short_language_candidates)

        wrapped_candidates = ModelSlimConfig._prefix_candidates(
            "model.language_model.model.layers.0.mlp.gate_proj"
        )

        self.assertIn(
            "language_model.model.layers.0.mlp.gate_proj", wrapped_candidates
        )
        self.assertIn("language_model.layers.0.mlp.gate_proj", wrapped_candidates)
        self.assertIn("model.layers.0.mlp.gate_proj", wrapped_candidates)

        moe_candidates = ModelSlimConfig._prefix_candidates(
            "language_model.model.layers.0.mlp.experts"
        )

        self.assertIn(
            "language_model.model.layers.0.block_sparse_moe.experts",
            moe_candidates,
        )
        self.assertIn("model.layers.0.block_sparse_moe.experts", moe_candidates)

    def test_modelslim_linear_scheme_uses_short_language_prefix_candidate(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        class DummyLinearScheme:
            def __init__(self, quant_config, prefix):
                self.quant_config = quant_config
                self.prefix = prefix

        cfg = ModelSlimConfig(
            {"language_model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC"}
        )

        with patch.object(modelslim, "ModelSlimW8A8Int8", DummyLinearScheme):
            scheme = cfg.get_linear_scheme(
                SimpleNamespace(), "language_model.model.layers.0.mlp.gate_proj"
            )

        self.assertEqual(scheme.prefix, "language_model.layers.0.mlp.gate_proj")

    def test_modelslim_config_normalizes_weight_packed_description_keys(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        class DummyLinearScheme:
            def __init__(self, quant_config, prefix):
                self.quant_config = quant_config
                self.prefix = prefix

        cfg = ModelSlimConfig(
            {
                "language_model.model.layers.0.self_attn.o_proj.weight_packed": "W8A8_DYNAMIC",
                "language_model.model.layers.0.mlp.experts.0.w1.weight_packed": "W8A8_DYNAMIC",
                "language_model.model.layers.0.mlp.experts.0.w3.weight_packed": "W8A8_DYNAMIC",
                "language_model.model.layers.0.mlp.experts.0.w2.weight_packed": "W8A8_DYNAMIC",
            }
        )

        self.assertEqual(
            cfg.quant_description[
                "language_model.model.layers.0.self_attn.o_proj.weight"
            ],
            "W8A8_DYNAMIC",
        )

        with patch.object(modelslim, "ModelSlimW8A8Int8", DummyLinearScheme):
            scheme = cfg.get_linear_scheme(
                SimpleNamespace(), "language_model.model.layers.0.self_attn.o_proj"
            )

        self.assertEqual(
            scheme.prefix, "language_model.model.layers.0.self_attn.o_proj"
        )

        class DummyMoEScheme:
            def __init__(self, quant_config, weight_group):
                self.quant_config = quant_config
                self.weight_group = weight_group

        with patch.object(modelslim, "ModelSlimW8A8Int8MoE", DummyMoEScheme):
            w13_scheme, w2_scheme = cfg.get_moe_scheme(
                SimpleNamespace(), "language_model.model.layers.0.mlp.experts"
            )

        self.assertEqual(w13_scheme.weight_group, "w13")
        self.assertEqual(w2_scheme.weight_group, "w2")
        self.assertIs(w13_scheme.quant_config, cfg.quant_description)
        self.assertIs(w2_scheme.quant_config, cfg.quant_description)

    def test_modelslim_fused_kda_linear_scheme_uses_packed_shard_prefix(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        class DummyLinearScheme:
            def __init__(self, quant_config, prefix):
                self.quant_config = quant_config
                self.prefix = prefix

        cfg = ModelSlimConfig(
            {
                "packed_modules_mapping": KimiK3ForConditionalGeneration.packed_modules_mapping,
                "language_model.model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            }
        )
        prefix = "language_model.model.layers.0.self_attn.fused_qkvbfg_a_proj"
        mapping = cfg.packed_modules_mapping["model"]
        proj_name = prefix.split(".")[-1]
        prefix_in_quant_config = prefix.replace(proj_name, mapping[proj_name][0])

        with patch.object(modelslim, "ModelSlimW8A8Int8", DummyLinearScheme):
            scheme = cfg.get_linear_scheme(SimpleNamespace(), prefix_in_quant_config)

        self.assertEqual(
            scheme.prefix, "language_model.model.layers.0.self_attn.q_proj"
        )

    def test_modelslim_moe_scheme_uses_text_only_prefix_candidate(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        class DummyMoEScheme:
            def __init__(self, quant_config, weight_group):
                self.quant_config = quant_config
                self.weight_group = weight_group

        quant_config = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
        }
        cfg = ModelSlimConfig(quant_config)

        with patch.object(modelslim, "ModelSlimW8A8Int8MoE", DummyMoEScheme):
            w13_scheme, w2_scheme = cfg.get_moe_scheme(
                SimpleNamespace(), "language_model.model.layers.0.mlp.experts"
            )

        self.assertEqual(w13_scheme.weight_group, "w13")
        self.assertEqual(w2_scheme.weight_group, "w2")

    def test_modelslim_moe_scheme_uses_hf_expert_names_candidate(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        class DummyMoEScheme:
            def __init__(self, quant_config, weight_group):
                self.quant_config = quant_config
                self.weight_group = weight_group

        quant_config = {
            "language_model.model.layers.19.block_sparse_moe.experts.0.w1.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.19.block_sparse_moe.experts.0.w3.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.19.block_sparse_moe.experts.0.w2.weight": "W8A8_DYNAMIC",
        }
        cfg = ModelSlimConfig(quant_config)

        with patch.object(modelslim, "ModelSlimW8A8Int8MoE", DummyMoEScheme):
            w13_scheme, w2_scheme = cfg.get_moe_scheme(
                SimpleNamespace(), "language_model.model.layers.19.mlp.experts"
            )

        self.assertEqual(w13_scheme.weight_group, "w13")
        self.assertEqual(w2_scheme.weight_group, "w2")

    def test_modelslim_moe_scheme_uses_block_sparse_sglang_names_candidate(self):
        from sglang.srt.layers.quantization.modelslim import modelslim
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        class DummyMoEScheme:
            def __init__(self, quant_config, weight_group):
                self.quant_config = quant_config
                self.weight_group = weight_group

        quant_config = {
            "language_model.model.layers.19.block_sparse_moe.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.19.block_sparse_moe.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.19.block_sparse_moe.experts.0.down_proj.weight": "W8A8_DYNAMIC",
        }
        cfg = ModelSlimConfig(quant_config)

        with patch.object(modelslim, "ModelSlimW8A8Int8MoE", DummyMoEScheme):
            w13_scheme, w2_scheme = cfg.get_moe_scheme(
                SimpleNamespace(), "language_model.model.layers.19.mlp.experts"
            )

        self.assertEqual(w13_scheme.weight_group, "w13")
        self.assertEqual(w2_scheme.weight_group, "w2")

    def test_modelslim_moe_scheme_requires_complete_w13_group(self):
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        quant_config = {
            "language_model.model.layers.19.block_sparse_moe.experts.0.w1.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.19.block_sparse_moe.experts.0.w2.weight": "W8A8_DYNAMIC",
        }
        cfg = ModelSlimConfig(quant_config)

        with self.assertRaisesRegex(ValueError, "Missing ModelSlim MoE"):
            cfg.get_moe_scheme(
                SimpleNamespace(), "language_model.model.layers.19.mlp.experts"
            )

    def test_kimi_linear_npu_expert_loader_maps_weight_packed(self):
        from sglang.srt.models import kimi_linear
        from sglang.srt.models.kimi_linear import KimiLinearForCausalLM

        loaded = []
        expected_weight_name = (
            "model.layers.19.block_sparse_moe.experts.w13_weight"
        )
        expected_scale_name = (
            "model.layers.19.block_sparse_moe.experts.w13_weight_scale"
        )

        class FakeParam:
            def weight_loader(
                self, param, loaded_weight, name, expert_id=None, shard_id=None
            ):
                loaded.append((param, loaded_weight, name, expert_id, shard_id))

        fake_param = FakeParam()
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                is_moe=True,
                num_experts=1,
                full_attention_layer_ids=[],
            ),
            named_parameters=lambda: [
                (expected_weight_name, fake_param),
                (expected_scale_name, fake_param),
            ],
        )

        with patch.object(kimi_linear, "_is_npu", True):
            KimiLinearForCausalLM.load_weights(
                fake_model,
                [
                    (
                        "model.layers.19.block_sparse_moe.experts.0.w1.weight_packed",
                        "packed-weight",
                    ),
                    (
                        "model.layers.19.block_sparse_moe.experts.0.w1.weight_scale",
                        "weight-scale",
                    ),
                ],
            )

        self.assertEqual(
            loaded,
            [
                (fake_param, "packed-weight", expected_weight_name, 0, "w1"),
                (fake_param, "weight-scale", expected_scale_name, 0, "w1"),
            ],
        )

    def test_kimi_linear_npu_fallback_loader_maps_weight_packed(self):
        from sglang.srt.models import kimi_linear
        from sglang.srt.models.kimi_linear import KimiLinearForCausalLM

        loaded = []
        expected_name = "model.layers.0.self_attn.o_proj.weight"

        class FakeParam:
            def weight_loader(self, param, loaded_weight, **kwargs):
                loaded.append((param, loaded_weight, kwargs))

        fake_param = FakeParam()
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                is_moe=False,
                is_linear_attn=True,
                full_attention_layer_ids=[],
            ),
            named_parameters=lambda: [(expected_name, fake_param)],
        )

        with patch.object(kimi_linear, "_is_npu", True):
            KimiLinearForCausalLM.load_weights(
                fake_model,
                [
                    (
                        "model.layers.0.self_attn.o_proj.weight_packed",
                        "packed-weight",
                        {"meta": "value"},
                    )
                ],
            )

        self.assertEqual(loaded, [(fake_param, "packed-weight", {"meta": "value"})])

    def test_modelslim_quant_config_prefers_file_over_hf_config(self):
        from sglang.srt.model_loader import weight_utils

        class DummyModelSlimConfig:
            @classmethod
            def get_config_filenames(cls):
                return ["quant_model_description.json"]

            @classmethod
            def from_config(cls, config):
                return config

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "quant_model_description.json").write_text(
                '{"model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC"}',
                encoding="utf-8",
            )
            model_config = SimpleNamespace(
                quantization="modelslim",
                hf_config=SimpleNamespace(
                    quantization_config={"quant_method": "compressed-tensors"},
                ),
                model_path=str(model_dir),
                revision=None,
            )
            load_config = SimpleNamespace(download_dir=None)

            with patch.object(
                weight_utils,
                "get_quantization_config",
                return_value=DummyModelSlimConfig,
            ):
                quant_config = weight_utils.get_quant_config(
                    model_config=model_config,
                    load_config=load_config,
                    packed_modules_mapping={"model": {}},
                )

        self.assertEqual(
            quant_config["model.layers.0.mlp.gate_proj.weight"], "W8A8_DYNAMIC"
        )
        self.assertEqual(quant_config["packed_modules_mapping"], {"model": {}})

    def test_npu_quant_config_merges_kimi_k3_packed_modules_mapping(self):
        from sglang.srt.model_loader import loader
        from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

        class DummyQuantConfig:
            def __init__(self, packed_modules_mapping):
                self.packed_modules_mapping = packed_modules_mapping

            def get_name(self):
                return "modelslim"

            def get_supported_act_dtypes(self):
                return ["float16"]

        captured = {}

        def fake_get_quant_config(
            model_config, load_config, packed_modules_mapping, remap_prefix
        ):
            captured["mapping"] = packed_modules_mapping
            return DummyQuantConfig(packed_modules_mapping)

        model_config = SimpleNamespace(
            quantization="modelslim",
            dtype="float16",
            is_fp4_experts=False,
            hf_config=SimpleNamespace(),
        )

        with (
            patch.object(loader, "_is_npu", True),
            patch.object(
                loader,
                "get_model_architecture",
                return_value=(KimiK3ForConditionalGeneration, "kimi_k3"),
            ),
            patch.object(loader, "get_quant_config", side_effect=fake_get_quant_config),
        ):
            quant_config = loader._get_quantization_config(
                model_config, SimpleNamespace()
            )

        self.assertIs(quant_config.packed_modules_mapping, captured["mapping"])
        model_mapping = captured["mapping"]["model"]
        self.assertEqual(
            model_mapping["fused_qkvbfg_a_proj"],
            ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
        )
        self.assertEqual(
            model_mapping["fused_qkv_a_proj_with_mqa"],
            ["q_a_proj", "kv_a_proj_with_mqa"],
        )
        self.assertNotIn(
            "fused_qkv_a_proj_with_mqa",
            KimiK3ForConditionalGeneration.packed_modules_mapping["model"],
        )

    def test_model_config_detects_modelslim_description_file(self):
        from sglang.srt.configs.model_config import ModelConfig

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "quant_model_description.json").write_text(
                '{"model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC"}',
                encoding="utf-8",
            )
            model_config = object.__new__(ModelConfig)
            model_config.model_path = str(model_dir)
            model_config.is_draft_model = False

            quant_config = ModelConfig._find_quant_modelslim_config(model_config)

        self.assertEqual(quant_config["quant_method"], "modelslim")
        self.assertEqual(
            quant_config["model.layers.0.mlp.gate_proj.weight"], "W8A8_DYNAMIC"
        )

    def test_model_config_verify_quantization_prefers_modelslim_file(self):
        from sglang.srt.configs.model_config import ModelConfig

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "quant_model_description.json").write_text(
                '{"model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC"}',
                encoding="utf-8",
            )
            model_config = object.__new__(ModelConfig)
            model_config.model_path = str(model_dir)
            model_config.is_draft_model = False
            model_config.quantization = None
            model_config.hf_config = SimpleNamespace(
                quantization_config={"quant_method": "compressed-tensors"}
            )

            ModelConfig._verify_quantization(model_config)

        self.assertEqual(model_config.quantization, "modelslim")
        self.assertFalse(getattr(model_config, "use_scale_ue8m0", True))

    def test_model_config_ignores_modelslim_file_for_draft_model(self):
        from sglang.srt.configs.model_config import ModelConfig

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "quant_model_description.json").write_text(
                '{"model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC"}',
                encoding="utf-8",
            )
            model_config = object.__new__(ModelConfig)
            model_config.model_path = str(model_dir)
            model_config.is_draft_model = True

            quant_config = ModelConfig._find_quant_modelslim_config(model_config)

        self.assertIsNone(quant_config)

    def test_npu_kda_rejects_target_verify_kwargs(self):
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            _check_npu_kda_target_verify_kwargs,
        )

        decode_kwargs = {"disable_state_update": False}
        _check_npu_kda_target_verify_kwargs(decode_kwargs)
        self.assertEqual(decode_kwargs, {})

        with self.assertRaisesRegex(
            NotImplementedError, "write intermediate states"
        ):
            _check_npu_kda_target_verify_kwargs({"disable_state_update": True})

        with self.assertRaisesRegex(NotImplementedError, "target_verify on NPU"):
            _check_npu_kda_target_verify_kwargs(
                {"intermediate_states_buffer": object()}
            )

    def test_kimi_k3_modelslim_artifact_checker_flags_missing_quant(self):
        script_path = (
            Path(__file__).parents[4]
            / "scripts"
            / "check_kimi_k3_modelslim_artifacts.py"
        )
        spec = importlib.util.spec_from_file_location("kimi_k3_artifact_check", script_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    KimiK3Config(
                        text_config={
                            "quantization_config": {"quant_method": "compressed-tensors"}
                        }
                    ).to_dict(),
                    f,
                )
            with open(
                model_dir / "model.safetensors.index.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "weight_map": {
                            "language_model.model.layers.0.self_attn.o_proj.weight_packed": "a.safetensors",
                            "language_model.model.layers.0.self_attn.o_proj.weight_scale": "a.safetensors",
                            "vision_tower.patch_embed.proj.weight": "b.safetensors",
                        }
                    },
                    f,
                )

            errors = []
            warnings = []
            checker.check_config(model_dir, errors, warnings)
            language_names, vision_names, suffixes = checker.collect_weight_index(
                model_dir, errors
            )
            checker.check_weight_index(
                language_names, vision_names, suffixes, errors, warnings
            )
            checker.check_quant_description(
                model_dir,
                language_names,
                allow_missing_quant=True,
                errors=errors,
                warnings=warnings,
            )

            self.assertFalse(errors)
            self.assertIn("missing quant_model_description.json", warnings)

            checker.check_quant_description(
                model_dir,
                language_names,
                allow_missing_quant=False,
                errors=errors,
                warnings=warnings,
            )

            self.assertIn("missing quant_model_description.json", errors)

    def test_kimi_k3_modelslim_artifact_checker_normalizes_packed_keys(self):
        script_path = (
            Path(__file__).parents[4]
            / "scripts"
            / "check_kimi_k3_modelslim_artifacts.py"
        )
        spec = importlib.util.spec_from_file_location("kimi_k3_artifact_check", script_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            with open(model_dir / "quant_model_description.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "language_model.model.layers.0.self_attn.o_proj.weight_packed": "W8A8_DYNAMIC"
                    },
                    f,
                )

            errors = []
            warnings = []
            checker.check_quant_description(
                model_dir,
                [
                    "language_model.model.layers.0.self_attn.o_proj.weight_packed",
                ],
                allow_missing_quant=False,
                errors=errors,
                warnings=warnings,
            )

            self.assertFalse(errors)
            self.assertNotIn(
                "no checkpoint *.weight_packed keys are directly covered by "
                "quant_model_description.json after _packed normalization",
                warnings,
            )

    def test_kimi_k3_modelslim_artifact_checker_validates_tokenizer_files(self):
        script_path = (
            Path(__file__).parents[4]
            / "scripts"
            / "check_kimi_k3_modelslim_artifacts.py"
        )
        spec = importlib.util.spec_from_file_location("kimi_k3_artifact_check", script_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            with open(model_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tokenizer_class": "TikTokenTokenizer",
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]
                        },
                        "bos_token": "[BOS]",
                        "eos_token": "[EOS]",
                        "pad_token": "[PAD]",
                        "unk_token": "[UNK]",
                    },
                    f,
                )
            (model_dir / "tokenization_kimi.py").write_text(
                "# tokenizer shim\n", encoding="utf-8"
            )
            (model_dir / "tiktoken.model").write_bytes(b"fake")

            errors = []
            warnings = []
            checker.check_tokenizer_artifacts(model_dir, errors, warnings)

            self.assertFalse(errors)
            self.assertIn(
                "tokenizer uses local auto_map code; launch SGLang with "
                "--trust-remote-code unless tokenizer initialization is skipped",
                warnings,
            )

            (model_dir / "tokenization_kimi.py").unlink()
            errors = []
            warnings = []
            checker.check_tokenizer_artifacts(model_dir, errors, warnings)

            self.assertIn(
                "tokenizer auto_map points to missing local module: tokenization_kimi.py",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
