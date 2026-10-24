import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Initialize the quantization registry as in model-config loading.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUDualLevelMXFP4LinearMethod,
    NPUMXFP4W4A8LinearMethod,
    NPUMXFP8LinearMethod,
    NPUSingleLevelMXFP4LinearMethod,
    NPUSingleLevelMXFP4OfflineLinearMethod,
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8LinearMethod,
    npu_w8a8_mxfp8_linear,
)
from sglang.srt.hardware_backend.npu.quantization.linear_utils import (
    cache_linear_bias_fp32,
    process_npu_arch35_mxfp8_linear_weights,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.test.test_utils import CustomTestCase


class TestNPULinearUtils(CustomTestCase):
    def test_online_placeholders_preserve_loader_and_partition_metadata(self):
        # Include inherited creators: a refactor must retain the TP loader ABI.
        for method_cls in (
            NPUMXFP8LinearMethod,
            NPUMXFP4W4A8LinearMethod,
            NPUSingleLevelMXFP4LinearMethod,
            NPUSingleLevelMXFP4OfflineLinearMethod,
            NPUDualLevelMXFP4LinearMethod,
        ):
            for dtype in (torch.float16, torch.bfloat16):
                for loader in (None, lambda *args: None):
                    with self.subTest(
                        method=method_cls.__name__, dtype=dtype, loader=loader
                    ):
                        layer = torch.nn.Module()
                        widths = [3, 5]
                        attrs = {"unused_attr": object()}
                        if loader is not None:
                            attrs["weight_loader"] = loader
                        method_cls().create_weights(
                            layer, 16, widths, 32, 64, dtype, **attrs
                        )
                        self.assertIs(
                            dict(layer.named_parameters())["weight"], layer.weight
                        )
                        self.assertEqual(layer.weight.shape, (8, 16))
                        self.assertEqual(layer.weight.dtype, dtype)
                        self.assertFalse(layer.weight.requires_grad)
                        self.assertEqual(
                            (layer.weight.input_dim, layer.weight.output_dim), (1, 0)
                        )
                        self.assertIs(layer.weight.weight_loader, loader)
                        self.assertIs(layer.logical_widths, widths)
                        self.assertEqual(layer.input_size_per_partition, 16)
                        self.assertEqual(layer.output_size_per_partition, 8)
                        self.assertEqual(layer.orig_dtype, dtype)
                        self.assertFalse(hasattr(layer.weight, "unused_attr"))

    def test_bias_cache_is_reset_and_only_created_for_non_fp32_bias(self):
        # A stale cache must not survive removal or replacement with FP32 bias.
        for dtype in (None, torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = torch.nn.Module()
                layer.bias_fp32 = torch.nn.Parameter(torch.full((4,), -100.0))
                if dtype is not None:
                    layer.bias = torch.nn.Parameter(torch.arange(4, dtype=dtype))
                cache_linear_bias_fp32(layer)
                if dtype in (None, torch.float32):
                    self.assertIsNone(layer.bias_fp32)
                else:
                    self.assertIs(
                        dict(layer.named_parameters())["bias_fp32"], layer.bias_fp32
                    )
                    self.assertFalse(layer.bias_fp32.requires_grad)
                    torch.testing.assert_close(layer.bias_fp32, layer.bias.float())
                    self.assertIsNot(layer.bias_fp32, layer.bias)
                layer.bias = None
                cache_linear_bias_fp32(layer)
                self.assertIsNone(layer.bias_fp32)

    def test_mxfp_apply_uses_cache_only_for_the_original_bias(self):
        for method_cls in (NPUMXFP8LinearMethod, NPUMXFP4W4A8LinearMethod):
            for use_cache in (False, True):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(torch.zeros(32, 4))
                layer.weight_scale = torch.nn.Parameter(torch.ones(1, 4, 2))
                layer.weight_scale_inv = layer.weight_scale
                layer.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.bfloat16))
                if use_cache:
                    cache_linear_bias_fp32(layer)
                # Equal values in another tensor must not select the cached object.
                for bias in (
                    None,
                    layer.bias,
                    layer.bias.detach().clone(),
                    layer.bias.float(),
                ):
                    with self.subTest(
                        method=method_cls.__name__, cached=use_cache, bias=bias
                    ):
                        npu_ops = MagicMock()
                        npu_ops.npu_dynamic_mx_quant.return_value = (
                            torch.zeros(6, 64, dtype=torch.float8_e4m3fn),
                            torch.zeros(6, 1, 2, dtype=torch.uint8),
                        )
                        npu_ops.npu_quant_matmul.return_value = torch.ones(
                            6, 4, dtype=torch.bfloat16
                        )
                        with patch.object(torch.ops, "npu", npu_ops, create=True):
                            output = method_cls().apply(
                                layer, torch.ones(2, 3, 64, dtype=torch.bfloat16), bias
                            )
                        quant_bias = npu_ops.npu_quant_matmul.call_args.kwargs["bias"]
                        self.assertEqual(output.shape, (2, 3, 4))
                        if bias is None:
                            self.assertIsNone(quant_bias)
                        else:
                            torch.testing.assert_close(quant_bias, bias.float())
                            if use_cache and bias is layer.bias:
                                self.assertIs(quant_bias, layer.bias_fp32)
                            elif use_cache:
                                self.assertIsNot(quant_bias, layer.bias_fp32)

    def test_int8_preparation_preserves_parameters_and_static_input_scales(self):
        original_to = torch.Tensor.to

        def cpu_to(tensor, *args, **kwargs):
            # Only stand in for the static method's mandatory NPU transfer.
            if kwargs.get("device") == "npu":
                kwargs = {**kwargs, "device": "cpu"}
            return original_to(tensor, *args, **kwargs)

        for method_cls in (NPUW8A8Int8LinearMethod, NPUW8A8Int8DynamicLinearMethod):
            for has_offset in (False, True):
                with self.subTest(method=method_cls.__name__, has_offset=has_offset):
                    layer = torch.nn.Module()
                    original_weight = torch.arange(24, dtype=torch.int8).reshape(4, 6)[
                        :, ::2
                    ]
                    layer.weight = torch.nn.Parameter(
                        original_weight, requires_grad=False
                    )
                    layer.weight.loader_marker = object()
                    layer.weight_scale = torch.nn.Parameter(
                        torch.arange(4.0).reshape(4, 1)
                    )
                    if has_offset:
                        layer.weight_offset = torch.nn.Parameter(
                            torch.arange(4.0).reshape(4, 1)
                        )
                    layer.input_scale = torch.nn.Parameter(torch.tensor([2.0]))
                    layer.input_offset = torch.nn.Parameter(torch.tensor([3.0]))
                    parameters = dict(layer.named_parameters())
                    marker = layer.weight.loader_marker

                    def format_cast(weight):
                        torch.testing.assert_close(weight, original_weight.T)
                        self.assertTrue(weight.is_contiguous())
                        return weight

                    with (
                        patch(
                            "sglang.srt.hardware_backend.npu.quantization.linear_utils.npu_format_cast",
                            side_effect=format_cast,
                        ) as cast,
                        patch.object(torch.Tensor, "to", cpu_to),
                    ):
                        method_cls().process_weights_after_loading(layer)
                    cast.assert_called_once()
                    for name, parameter in parameters.items():
                        self.assertIs(getattr(layer, name), parameter)
                    self.assertIs(layer.weight.loader_marker, marker)
                    torch.testing.assert_close(layer.weight, original_weight.T)
                    torch.testing.assert_close(layer.weight_scale, torch.arange(4.0))
                    if has_offset:
                        torch.testing.assert_close(
                            layer.weight_offset, torch.arange(4.0)
                        )
                    else:
                        self.assertFalse(hasattr(layer, "weight_offset"))
                    if method_cls is NPUW8A8Int8LinearMethod:
                        torch.testing.assert_close(
                            layer.aclnn_input_scale, torch.full((3,), 2.0)
                        )
                        torch.testing.assert_close(
                            layer.aclnn_input_scale_reciprocal, torch.full((3,), 0.5)
                        )
                        torch.testing.assert_close(
                            layer.aclnn_input_offset, torch.full((3,), 3.0)
                        )
                    else:
                        self.assertFalse(hasattr(layer, "aclnn_input_scale"))


class TestNPUW8A8BlockFP8Linear(unittest.TestCase):
    def test_fp8_config_preserves_ue8m0_scale_format(self):
        quant_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
            }
        )

        self.assertEqual(quant_config.scale_fmt, "ue8m0")

    def test_layout_only_ue8m0_conversion_preserves_fp8_weight(self):
        original_weight = torch.randint(1, 255, (128, 64), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(original_weight.clone(), requires_grad=False),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[2**-12]], dtype=torch.float32), requires_grad=False
            ),
        )

        process_npu_arch35_mxfp8_linear_weights(layer, [128, 128], scale_fmt="ue8m0")

        self.assertEqual(layer.weight.shape, (64, 128))
        torch.testing.assert_close(
            layer.weight.data.T.contiguous().view(torch.uint8),
            original_weight.view(torch.uint8),
        )
        self.assertEqual(layer.weight_scale_inv.shape, (1, 128, 2))
        self.assertTrue(
            torch.equal(
                layer.weight_scale_inv.data,
                torch.full((1, 128, 2), 0x73, dtype=torch.uint8),
            )
        )
        self.assertTrue(layer.weight_scale_inv.format_ue8m0)

    def test_rejects_unaligned_non_ue8m0_weight(self):
        layer = SimpleNamespace(weight=torch.empty(128, 65, dtype=torch.float8_e4m3fn))
        with self.assertRaisesRegex(ValueError, "requires K to be divisible by 64"):
            process_npu_arch35_mxfp8_linear_weights(
                layer, [128, 128], scale_fmt="float32"
            )

    def test_rejects_non_fp8_weight(self):
        with self.assertRaisesRegex(ValueError, "expects float8_e4m3fn weights"):
            npu_w8a8_mxfp8_linear(
                torch.empty(1, 128, dtype=torch.bfloat16),
                torch.empty(128, 64, dtype=torch.bfloat16),
                [128, 128],
                torch.empty(1),
            )

    def test_quantizes_flattened_input_and_restores_batch_shape(self):
        input_tensor = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.float32)
        quantized = torch.empty(6, 128, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(6, 2, 2, dtype=torch.uint8)
        matmul_output = torch.randn(6, 64, dtype=torch.bfloat16)

        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (quantized, input_scale)
        npu_ops.npu_quant_matmul.return_value = matmul_output
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            output = npu_w8a8_mxfp8_linear(
                input_tensor,
                weight,
                [64, 128],
                weight_scale,
                bias=bias,
            )

        self.assertEqual(output.shape, (2, 3, 64))
        quant_call = npu_ops.npu_dynamic_mx_quant.call_args
        self.assertEqual(quant_call.args[0].shape, (6, 128))
        self.assertEqual(quant_call.kwargs["dst_type"], torch.float8_e4m3fn)

        matmul_call = npu_ops.npu_quant_matmul.call_args
        self.assertIs(matmul_call.args[0], quantized)
        self.assertIs(matmul_call.args[1], weight)
        self.assertIs(matmul_call.kwargs["scale"], weight_scale)
        self.assertIs(matmul_call.kwargs["pertoken_scale"], input_scale)
        self.assertIs(matmul_call.kwargs["bias"], bias)
        self.assertEqual(matmul_call.kwargs["group_sizes"], (1, 1, 32))

    def test_rejects_noncontiguous_input(self):
        input_tensor = torch.randn(2, 3, 128, dtype=torch.bfloat16).transpose(0, 1)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)

        with self.assertRaisesRegex(RuntimeError, "view size is not compatible"):
            npu_w8a8_mxfp8_linear(input_tensor, weight, [64, 128], weight_scale)

    def test_preserves_supported_input_dtype(self):
        input_tensor = torch.randn(2, 128, dtype=torch.float16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            npu_w8a8_mxfp8_linear(input_tensor, weight, [128, 128], weight_scale)

        self.assertEqual(
            npu_ops.npu_quant_matmul.call_args.kwargs["output_dtype"],
            torch.float16,
        )

    def test_converts_bias_to_float32(self):
        input_tensor = torch.randn(2, 128, dtype=torch.bfloat16)
        weight = torch.empty(128, 64, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(2, 64, 2, dtype=torch.uint8)
        bias = torch.randn(64, dtype=torch.bfloat16)
        npu_ops = MagicMock()
        npu_ops.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 128, dtype=torch.float8_e4m3fn),
            torch.empty(2, 2, 2, dtype=torch.uint8),
        )
        npu_ops.npu_quant_matmul.return_value = torch.empty(2, 64)

        with patch.object(torch.ops, "npu", npu_ops, create=True):
            npu_w8a8_mxfp8_linear(
                input_tensor,
                weight,
                [128, 128],
                weight_scale,
                bias=bias,
            )

        quant_bias = npu_ops.npu_quant_matmul.call_args.kwargs["bias"]
        self.assertEqual(quant_bias.dtype, torch.float32)
        torch.testing.assert_close(quant_bias, bias.float())


if __name__ == "__main__":
    unittest.main()
