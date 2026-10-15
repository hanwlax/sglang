"""Exercise Qwen2.5-VL's NPU prefill projection and its fallback contracts."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.hardware_backend.npu.modules.linear import prefill_gate_up
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")


class TestNPUQwenVLPrefill(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy", device="npu"), role="test")
        self.addCleanup(reset_context)
        torch.manual_seed(42)

    def make_layer(self, k=64, n=128, dtype=torch.bfloat16, **kwargs):
        layer = MergedColumnParallelLinear(
            k,
            [n // 2, n // 2],
            params_dtype=dtype,
            tp_rank=0,
            tp_size=1,
            bias=kwargs.pop("bias", False),
            **kwargs,
        ).to("npu")
        with torch.no_grad():
            layer.weight.normal_(std=0.02)
            if layer.bias is not None:
                layer.bias.fill_(0.25)
        return layer

    def assert_fallback(self, layer, x, mode):
        expected, expected_bias = layer(x)
        with patch.object(
            torch_npu,
            "npu_grouped_matmul",
            side_effect=AssertionError("unexpected GMM"),
        ):
            actual, actual_bias = prefill_gate_up(layer, x, mode)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(actual_bias, expected_bias)

    def test_prefill_shapes_and_parameter_identity(self):
        # Include the full failing CI shape; a last-row GEMM misses its K tiling.
        for m, k, n in [(1, 64, 128), (17, 64, 128), (2526, 2048, 11008)]:
            for mode in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL):
                with self.subTest(shape=(m, k, n), mode=mode):
                    layer = self.make_layer(k, n)
                    x = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
                    weight = layer.weight
                    saved = weight.detach().clone()
                    actual, bias = prefill_gate_up(layer, x, mode)
                    expected = F.linear(x.float(), weight.float())
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertEqual(actual.shape, (m, n))
                    self.assertIsNone(bias)
                    self.assertIs(layer.weight, weight)
                    torch.testing.assert_close(weight, saved, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual.float(), expected, rtol=0.005, atol=0.001
                    )

    def test_batched_decode_and_other_modes_preserve_linear(self):
        layer = self.make_layer()
        for mode in ForwardMode:
            if mode in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL):
                continue
            with self.subTest(mode=mode):
                self.assert_fallback(
                    layer, torch.randn(8, 64, device="npu", dtype=torch.bfloat16), mode
                )
        self.assert_fallback(
            layer,
            torch.empty(0, 64, device="npu", dtype=torch.bfloat16),
            ForwardMode.EXTEND,
        )
        self.assert_fallback(
            layer, torch.ones(8, 64, device="npu", dtype=torch.bfloat16), None
        )

    def test_dtype_bias_and_layout_fallbacks(self):
        cases = [
            (
                self.make_layer(dtype=torch.float16),
                torch.ones(8, 64, device="npu", dtype=torch.float16),
            ),
            (
                self.make_layer(bias=True),
                torch.ones(8, 64, device="npu", dtype=torch.bfloat16),
            ),
            (
                self.make_layer(bias=True, skip_bias_add=True),
                torch.ones(8, 64, device="npu", dtype=torch.bfloat16),
            ),
            (
                self.make_layer(),
                torch.ones(64, 8, device="npu", dtype=torch.bfloat16).t(),
            ),
            (self.make_layer().cpu(), torch.ones(8, 64, dtype=torch.bfloat16)),
        ]
        for i, (layer, x) in enumerate(cases):
            with self.subTest(case=i):
                self.assert_fallback(layer, x, ForwardMode.EXTEND)

    def test_custom_methods_and_adapter_wrappers_are_preserved(self):
        class CustomMethod(UnquantizedLinearMethod):
            def apply(self, layer, x, bias=None):
                return F.linear(x, layer.weight, bias) + 1

        class Adapter(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                output, bias = self.base(x)
                return output + 2, bias

        x = torch.ones(8, 64, device="npu", dtype=torch.bfloat16)
        layer = self.make_layer()
        layer.quant_method = CustomMethod()
        self.assert_fallback(layer, x, ForwardMode.EXTEND)
        self.assert_fallback(Adapter(self.make_layer()), x, ForwardMode.EXTEND)


if __name__ == "__main__":
    unittest.main()
