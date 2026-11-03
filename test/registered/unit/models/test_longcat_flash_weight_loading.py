"""LongCat MLA derived weights for NPU block-FP8 checkpoints."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import longcat_flash  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLongcatFlashNpuWeightLoading(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = self.enterContext(ExitStack())
        # Exercise NPU post-load logic with CPU tensors; no accelerator kernels.
        for name, value in (
            ("_is_npu", True),
            ("_is_cuda", False),
            ("_is_hip", False),
            ("_is_fp8_fnuz", False),
        ):
            stack.enter_context(patch.object(longcat_flash, name, value))
        stack.enter_context(
            patch.object(
                longcat_flash,
                "should_deepgemm_weight_requant_ue8m0",
                return_value=False,
            )
        )

    def _make_model(self, *, num_heads=2, num_layers=1, fp8=True):
        generator = torch.Generator().manual_seed(17)
        layers = []
        for _ in range(num_layers):
            attentions = []
            for _ in range(2):
                weight = torch.randn(num_heads * 256, 512, generator=generator)
                proj = SimpleNamespace(
                    weight=torch.nn.Parameter(
                        weight.to(torch.float8_e4m3fn if fp8 else torch.bfloat16),
                        requires_grad=False,
                    )
                )
                if fp8:
                    # Real FP32 dequantization scales, deliberately not UE8M0.
                    proj.weight_scale_inv = torch.nn.Parameter(
                        torch.rand(num_heads * 2, 4, generator=generator) * 0.003
                        + 0.000201634,
                        requires_grad=False,
                    )
                attentions.append(
                    SimpleNamespace(
                        kv_b_proj=proj,
                        qk_nope_head_dim=128,
                        v_head_dim=128,
                        w_kc=None,
                        w_vc=None,
                        w_scale=None,
                    )
                )
            layers.append(SimpleNamespace(self_attn=attentions))
        return SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=num_layers,
                mla_scale_q_lora=False,
                mla_scale_kv_lora=False,
            ),
            quant_config=SimpleNamespace(weight_block_size=[128, 128]) if fp8 else None,
            model=SimpleNamespace(layers=layers),
        )

    def _assert_derived_weights(self, attention):
        proj = attention.kv_b_proj
        reference = proj.weight.float()
        if hasattr(proj, "weight_scale_inv"):
            # Index the scale for each element independently of block expansion.
            rows = torch.arange(reference.shape[0]) // 128
            cols = torch.arange(reference.shape[1]) // 128
            reference = reference * proj.weight_scale_inv[rows[:, None], cols[None, :]]
        reference = reference.to(torch.bfloat16).reshape(-1, 256, 512)
        torch.testing.assert_close(attention.w_kc, reference[:, :128], rtol=0, atol=0)
        torch.testing.assert_close(
            attention.w_vc, reference[:, 128:].transpose(1, 2), rtol=0, atol=0
        )
        self.assertEqual(attention.w_kc.dtype, torch.bfloat16)
        self.assertEqual(attention.w_vc.dtype, torch.bfloat16)
        self.assertTrue(attention.w_kc.transpose(1, 2).is_contiguous())
        self.assertTrue(attention.w_vc.is_contiguous())
        self.assertIsNone(attention.w_scale)

    def test_block_fp8_dequantizes_derived_weights_without_changing_checkpoint(self):
        for tp_size in (1, 8, 32):
            with self.subTest(tp_size=tp_size):
                model = self._make_model(num_heads=64 // tp_size)
                originals = [
                    (
                        a.kv_b_proj.weight,
                        a.kv_b_proj.weight.view(torch.uint8).clone(),
                        a.kv_b_proj.weight_scale_inv,
                        a.kv_b_proj.weight_scale_inv.clone(),
                    )
                    for a in model.model.layers[0].self_attn
                ]
                longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
                for attention, (weight, payload, scale, scales) in zip(
                    model.model.layers[0].self_attn, originals
                ):
                    self._assert_derived_weights(attention)
                    self.assertIs(attention.kv_b_proj.weight, weight)
                    self.assertIs(attention.kv_b_proj.weight_scale_inv, scale)
                    torch.testing.assert_close(weight.view(torch.uint8), payload)
                    torch.testing.assert_close(scale, scales, rtol=0, atol=0)

    def test_partial_reload_refreshes_bf16_weights_in_place(self):
        model = self._make_model(num_layers=2)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
        original_buffers = [
            (a.w_kc, a.w_vc, a.w_kc.clone(), a.w_vc.clone())
            for layer in model.model.layers
            for a in layer.self_attn
        ]
        with torch.no_grad():
            for attention in model.model.layers[1].self_attn:
                attention.kv_b_proj.weight_scale_inv.mul_(1.3)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(
            model,
            weight_names=["model.layers.1.self_attn.0.kv_b_proj.weight_scale_inv"],
        )
        for layer_id, layer in enumerate(model.model.layers):
            for index, attention in enumerate(layer.self_attn):
                w_kc, w_vc, old_kc, old_vc = original_buffers[layer_id * 2 + index]
                self.assertIs(attention.w_kc, w_kc)
                self.assertIs(attention.w_vc, w_vc)
                self._assert_derived_weights(attention)
                if layer_id == 0:
                    torch.testing.assert_close(attention.w_kc, old_kc, rtol=0, atol=0)
                    torch.testing.assert_close(attention.w_vc, old_vc, rtol=0, atol=0)
                else:
                    self.assertFalse(torch.equal(attention.w_kc, old_kc))
                    self.assertFalse(torch.equal(attention.w_vc, old_vc))

    def test_unquantized_bf16_still_uses_original_weights(self):
        model = self._make_model(fp8=False)
        longcat_flash.LongcatFlashForCausalLM.post_load_weights(model)
        for attention in model.model.layers[0].self_attn:
            self._assert_derived_weights(attention)


if __name__ == "__main__":
    unittest.main()
