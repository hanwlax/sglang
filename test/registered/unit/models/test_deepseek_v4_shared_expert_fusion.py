import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import MoeA2ABackend
from sglang.srt.layers.moe.utils import (
    is_shared_experts_fusion_disabled,
    resolve_shared_experts_config,
)
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    remap_dspark_weight_name_npu,
)
from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_model,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(CustomTestCase):
    """V4 shared-expert fusion requires explicit startup intent."""

    def setUp(self):
        super().setUp()
        cm = get_parallel().override(moe_ep_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _publish(self, enforce):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce
        )
        override.install()
        self.addCleanup(override.restore)

    def _install(self, model_class=DeepseekV4ForCausalLM, n_shared_experts=1):
        config = resolve_shared_experts_config(
            model_class,
            SimpleNamespace(n_shared_experts=n_shared_experts),
            None,
        )

        scope = get_model().override(shared_experts=config)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def _make_dspark_config(self):
        return DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLMDSpark"],
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            engram_num_embeddings=[],
            n_shared_experts=1,
            dspark_markov_rank=1,
            num_nextn_predict_layers=1,
        )

    def test_disables_shared_fusion_without_enforce(self):
        self._publish(enforce=False)
        self.assertEqual(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), None
            ),
            "Config does not support fused shared expert(s).",
        )
        self._install()
        # The resolved model decision leaves the startup intent untouched.
        self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    def test_v41_vision_admits_cuda_megamoe_with_dp_attention(self):
        """A vision-capable checkpoint must initialize for text or image serving
        with DP attention and CUDA MegaMoE. CP, PP and other A2A paths still lack
        vision routing support and must fail before constructing the tower.
        """
        self._publish(enforce=False)
        config = DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLM"],
            model_type="deepseek_v41",
            vision_n_layers=1,
            hidden_size=4,
            tie_word_embeddings=True,
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            engram_num_embeddings=[],
        )
        backbone = nn.Module()
        backbone.embed_tokens = nn.Identity()
        backbone.start_layer = 0
        backbone.end_layer = 0
        cases = (
            (MoeA2ABackend.MEGAMOE, True, 1, 1, True),
            (MoeA2ABackend.NONE, True, 1, 1, True),
            (MoeA2ABackend.MEGAMOE, False, 1, 1, False),
            (MoeA2ABackend.DEEPEP, True, 1, 1, False),
            (MoeA2ABackend.MEGAMOE, True, 2, 1, False),
            (MoeA2ABackend.MEGAMOE, True, 1, 2, False),
        )
        for backend, cuda, cp, pp, admitted in cases:
            with self.subTest(backend=backend, cuda=cuda, cp=cp, pp=pp):
                with (
                    get_parallel().override(
                        tp_size=4,
                        attn_dp_size=2,
                        attn_tp_size=2 // cp,
                        attn_cp_size=cp,
                        moe_ep_size=4,
                        pp_size=pp,
                        pp_group=SimpleNamespace(world_size=pp, is_last_rank=True),
                    ),
                    get_flags().moe.override(a2a_backend=backend),
                    get_model().override(
                        shared_experts=SharedExpertsConfig(fusion_disabled=True)
                    ),
                    patch.object(deepseek_v4, "_is_cuda", cuda),
                    patch.object(deepseek_v4, "ViT", return_value=nn.Identity()),
                    patch.object(deepseek_v4, "Aligner", return_value=nn.Identity()),
                    patch.object(deepseek_v4, "DeepseekV4Model", return_value=backbone),
                    patch.object(
                        deepseek_v4, "LogitsProcessor", return_value=nn.Identity()
                    ),
                    patch.object(deepseek_v4, "get_attn_tp_context"),
                ):
                    if admitted:
                        model = DeepseekV4ForCausalLM(config)
                        self.assertIsInstance(model.vision, nn.Module)
                        self.assertEqual(model.image_start.shape, (4,))
                    else:
                        with self.assertRaisesRegex(ValueError, "V4.1 vision"):
                            DeepseekV4ForCausalLM(config)

    def test_enables_shared_fusion_when_enforced(self):
        self._publish(enforce=True)
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), None
            )
        )
        self._install()
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_enforcing_with_more_than_one_shared_expert_is_rejected(self):
        self._publish(enforce=True)
        with self.assertRaisesRegex(ValueError, "exactly one shared"):
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=2), None
            )

    def test_mixed_precision_quant_vetoes_even_when_enforced(self):
        """A precision mismatch causes crash when shared expert fusion is enabled,
        so --enforce-shared-experts-fusion must not override it. Guards the gap
        where the enforce early-return skipped the quant check entirely."""
        self._publish(enforce=True)
        mixed = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: False
        )
        self.assertIn(
            "higher precision",
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), mixed
            ),
        )
        matched = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: True
        )
        self.assertIsNone(
            DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                SimpleNamespace(n_shared_experts=1), matched
            )
        )

    def test_dspark_entry_class_uses_the_v4_gate(self):
        """A DSV4 DSpark draft must inherit the target's default fusion policy."""
        self._publish(enforce=False)

        self._install(DeepseekV4ForCausalLMDSpark)

        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_dspark_records_explicitly_forced_fusion(self):
        """A forced DSpark build must retain its fused shared-expert count."""
        self._publish(enforce=True)
        self._install(DeepseekV4ForCausalLMDSpark)

        class Stage(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

        class MarkovHead(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

        with (
            patch(
                "sglang.srt.models.deepseek_v4_dspark.DSparkV4Stage",
                Stage,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.DSparkV4MarkovHead",
                MarkovHead,
            ),
            patch(
                "sglang.srt.models.deepseek_v4_dspark.build_dspark_v4_confidence_head",
                return_value=None,
            ),
        ):
            model = DeepseekV4ForCausalLMDSpark(self._make_dspark_config())

        self.assertEqual(model.num_fused_shared_experts, 1)

    def test_dspark_loads_forced_shared_expert_into_fused_slot(self):
        """Forced DSpark shared tensors must load instead of being skipped."""
        config = self._make_dspark_config()
        loaded = []

        class Param:
            def weight_loader(
                self,
                _param,
                loaded_weight,
                candidate,
                *,
                shard_id,
                expert_id,
            ):
                loaded.append((loaded_weight, candidate, shard_id, expert_id))

        class DraftModel:
            num_fused_shared_experts = 1
            confidence_head = None

            def __init__(self):
                self.config = config

            def named_parameters(self):
                return [("stages.0.mlp.experts.w13_weight", Param())]

            def _remap_dspark_weight_name(self, name):
                return DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(self, name)

            def _assert_confidence_head_loaded(self, **_kwargs):
                return None

            def post_load_weights(self):
                return None

        weight = torch.ones(1)
        DeepseekV4ForCausalLMDSpark.load_weights(
            DraftModel(),
            [("mtp.0.ffn.shared_experts.w1.weight", weight)],
        )

        self.assertEqual(len(loaded), 1)
        self.assertIs(loaded[0][0], weight)
        self.assertEqual(loaded[0][1], "stages.0.mlp.experts.w13_weight")
        self.assertEqual(loaded[0][2:], ("w1", config.n_routed_experts))


class TestDSparkNpuWeightMapping(CustomTestCase):
    def test_modelslim_uses_loader_names_and_preserves_config_precedence(self):
        # Checkpoint names must resolve the same runtime modules in quantization
        # and loading, including already normalized paths and stage-less heads.
        cases = {
            "mtp.0.attn.wq_a.weight": "stages.0.self_attn.wq_a.weight",
            "mtp.0.self_attn.wkv.weight": "stages.0.self_attn.wkv.weight",
            "mtp.1.ffn.experts.0.w1.weight": "stages.1.mlp.experts.0.gate_proj.weight",
            "mtp.1.ffn.experts.0.w2.weight": "stages.1.mlp.experts.0.down_proj.weight",
            "mtp.1.ffn.experts.0.w3.weight": "stages.1.mlp.experts.0.up_proj.weight",
            "mtp.0.attn_norm.weight": "stages.0.input_layernorm.weight",
            "mtp.1.ffn_norm.weight": "stages.1.post_attention_layernorm.weight",
            "mtp.1.ffn.gate.bias": "stages.1.mlp.gate.e_score_correction_bias",
            "mtp.1.ffn.gate.tid2eid": "stages.1.mlp.topk.tid2eid",
            "mtp.0.markov_head.embed.weight": "markov_head.embed.weight",
            "mtp.0.confidence_head.proj.weight": "confidence_head.proj.weight",
            "mtp.0.embed.weight": "embed_tokens.weight",
            "mtp.1.head.weight": "lm_head.weight",
        }
        raw = {name: "W8A8_DYNAMIC" for name in cases}
        canonical = "stages.0.self_attn.wq_a.weight"
        raw[canonical] = "FLOAT"
        raw["ignore"] = ["untouched"]
        raw["packed_modules_mapping"] = {}
        config = ModelSlimConfig(raw.copy())

        for name, mapped in cases.items():
            with self.subTest(name=name):
                self.assertEqual(remap_dspark_weight_name_npu(name), mapped)
                self.assertEqual(config.quant_description[name], raw[name])
                if mapped in ("embed_tokens.weight", "lm_head.weight"):
                    self.assertNotIn(mapped, config.quant_description)
                else:
                    self.assertEqual(
                        config.quant_description[mapped],
                        "FLOAT" if mapped == canonical else "W8A8_DYNAMIC",
                    )
        self.assertTrue(config.is_layer_skipped("stages.0.self_attn.wq_a"))
        self.assertFalse(config.is_layer_skipped("stages.0.self_attn.wkv"))
        self.assertEqual(config.ignore, ["untouched"])
        self.assertEqual(config.packed_modules_mapping, {})

        # Explicit global vocabulary settings must survive the alias filter.
        for name in ("embed_tokens.weight", "lm_head.weight"):
            config = ModelSlimConfig({**raw, name: "FLOAT"})
            self.assertEqual(config.quant_description[name], "FLOAT")

        # The full V4 config may normalize mtp paths before adding DSpark aliases.
        config = ModelSlimConfig(
            {"hc_head_fn": "FLOAT", "mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC"}
        )
        self.assertFalse(config.is_layer_skipped("stages.0.self_attn.wq_a"))

    def test_config_aliases_preserve_scale_names_rotary_entries_and_collisions(self):
        # Loader normalization must not merge distinct config entries or remove
        # the old scale/rotary aliases, even when their schemes differ.
        raw = {
            "mtp.0.attn.wo_a.scale": "FLOAT",
            "mtp.0.attn.wo_a.weight_scale_inv": "W8A8_DYNAMIC",
            "mtp.0.ffn.experts.0.w1.scale": "FLOAT",
            "mtp.0.attn.rotary_emb.inv_freq": "FLOAT",
            "mtp.0.markov_head.rotary_emb.inv_freq": "FLOAT",
            "mtp.0.confidence_head.scale": "FLOAT",
            "stages.0.self_attn.wo_a.scale": "W4A8_MXFP",
        }
        config = ModelSlimConfig(raw.copy())
        self.assertEqual(
            config.quant_description,
            {
                **raw,
                "stages.0.self_attn.wo_a.weight_scale_inv": "W8A8_DYNAMIC",
                "stages.0.mlp.experts.0.gate_proj.scale": "FLOAT",
                "stages.0.self_attn.rotary_emb.inv_freq": "FLOAT",
                "markov_head.rotary_emb.inv_freq": "FLOAT",
                "confidence_head.scale": "FLOAT",
            },
        )
        self.assertEqual(
            remap_dspark_weight_name_npu("mtp.0.attn.wo_a.scale"),
            "stages.0.self_attn.wo_a.weight_scale_inv",
        )
        self.assertIsNone(
            remap_dspark_weight_name_npu("mtp.0.attn.rotary_emb.inv_freq")
        )

        # Full V4 config preprocessing already converts attention scale names;
        # the DSpark alias pass must preserve that earlier normalization too.
        config = ModelSlimConfig({**raw, "hc_head_fn": "FLOAT"})
        self.assertNotIn("stages.0.self_attn.wo_a.scale", config.quant_description)
        self.assertEqual(
            config.quant_description["stages.0.self_attn.wo_a.weight_scale_inv"],
            "W4A8_MXFP",
        )

    def test_rejects_non_draft_names_buffers_and_malformed_stages(self):
        for name in (
            "embed.weight",
            "lm_head.weight",
            "layers.0.attn.wq_a.weight",
            "mtp.0.attn.rotary_emb.inv_freq",
            "mtp",
            "mtp.0",
            "mtp.x.attn.wq_a.weight",
            "mtp..attn.wq_a.weight",
            "mtp.0.embed.weight_scale",
        ):
            with self.subTest(name=name):
                self.assertIsNone(remap_dspark_weight_name_npu(name))
        with self.assertRaises(ValueError):
            remap_dspark_weight_name_npu("mtp.².attn.wq_a.weight")
        self.assertEqual(
            remap_dspark_weight_name_npu(
                "mtp.².attn.wq_a.weight", for_quant_config=True
            ),
            "stages.².self_attn.wq_a.weight",
        )

    @staticmethod
    def _make_loader_model(own_vocab, with_confidence, num_stages):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(n_routed_experts=0)
        model.num_fused_shared_experts = 0
        model.uses_own_vocab_modules = own_vocab
        model.num_stages = num_stages
        model.stages = nn.ModuleList()
        model.embed_tokens = nn.Embedding(4, 2)
        model.lm_head = nn.Linear(2, 4, bias=False)
        model.confidence_head = (
            nn.ModuleDict({"proj": nn.Linear(2, 1, bias=False)})
            if with_confidence
            else None
        )
        for parameter in model.parameters():
            parameter.data.zero_()
        return model

    def test_loader_selects_vocab_stages_and_filters_disabled_confidence(self):
        # A pure remap must not make checkpoint-local vocab overwrite shared
        # target modules, select the wrong stage, or warn about a disabled head.
        for own_vocab, with_confidence, num_stages in itertools.product(
            (False, True), (False, True), (1, 2, 3)
        ):
            with self.subTest(
                own_vocab=own_vocab,
                with_confidence=with_confidence,
                num_stages=num_stages,
            ):
                model = self._make_loader_model(own_vocab, with_confidence, num_stages)
                weights = []
                for stage in range(num_stages + 1):
                    weights.extend(
                        (
                            (
                                f"mtp.{stage}.embed.weight",
                                torch.full((4, 2), stage + 1.0),
                            ),
                            (
                                f"mtp.{stage}.head.weight",
                                torch.full((4, 2), stage + 11.0),
                            ),
                        )
                    )
                weights.append(
                    ("mtp.0.confidence_head.proj.weight", torch.full((1, 2), 99.0))
                )
                with (
                    patch.object(dspark, "_is_npu", True),
                    self.assertNoLogs(dspark.logger, level="WARNING"),
                ):
                    model.load_weights(weights)

                torch.testing.assert_close(
                    model.embed_tokens.weight,
                    torch.full((4, 2), 1.0 if own_vocab else 0.0),
                )
                torch.testing.assert_close(
                    model.lm_head.weight,
                    torch.full((4, 2), num_stages + 10.0 if own_vocab else 0.0),
                )
                if with_confidence:
                    torch.testing.assert_close(
                        model.confidence_head["proj"].weight, torch.full((1, 2), 99.0)
                    )

    def test_cuda_loader_skips_disabled_confidence_without_warning(self):
        model = self._make_loader_model(False, False, 1)
        name = "mtp.0.confidence_head.proj.weight"
        with (
            patch.object(dspark, "_is_npu", False),
            self.assertNoLogs(dspark.logger, level="WARNING"),
        ):
            model.load_weights([(name, torch.full((1, 2), 99.0))])

    def test_enabled_confidence_head_still_requires_checkpoint_weights(self):
        model = self._make_loader_model(True, True, 2)
        with (
            patch.object(dspark, "_is_npu", True),
            self.assertRaisesRegex(ValueError, "confidence_head.proj.weight"),
        ):
            model.load_weights([])


if __name__ == "__main__":
    unittest.main()
