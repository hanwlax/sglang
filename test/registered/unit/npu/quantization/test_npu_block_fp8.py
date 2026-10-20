"""Native block FP8 ABI and runtime integration, with CPU vendor-op references.

The external CANN boundary is replaced by dequantize-and-matmul references.
The runtime, parameter loading, registry dispatch and Ascend runner stay real.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm import (
    npu_w8a8_block_fp8_grouped_matmul,
    npu_w8a8_block_fp8_linear,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _quantize(x, **kwargs):
    """Independent 1x128 FP8 activation quantization reference."""
    m, k = x.shape
    blocks = F.pad(x.float(), (0, (-k) % 128)).reshape(m, -1, 128)
    scale = blocks.abs().amax(-1).clamp_min(1e-12) / 448
    q = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.reshape(m, -1)[:, :k].contiguous(), scale


def _expand_weight(q, scale):
    expanded = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return q.float() * expanded[..., : q.shape[-2], : q.shape[-1]]


def _expand_input(q, scale):
    return q.float() * scale.repeat_interleave(128, -1)[:, : q.shape[-1]]


def _matmul(x, weight, *, scale, pertoken_scale, output_dtype, group_sizes):
    if group_sizes != [1, 128, 128]:
        raise ValueError("The reference accepts only CANN G-B mode")
    w = _expand_weight(weight.t(), scale.t())
    return (_expand_input(x, pertoken_scale) @ w.t()).to(output_dtype)


def _grouped_matmul(
    *,
    x,
    weight,
    scale,
    per_token_scale,
    group_list,
    group_list_type,
    output_dtype,
    **kwargs,
):
    counts = (
        group_list
        if group_list_type == 1
        else torch.diff(group_list, prepend=group_list.new_zeros(1))
    )
    w = _expand_weight(weight[0].transpose(-1, -2), scale[0].transpose(-1, -2))
    act = _expand_input(x[0], per_token_scale[0])
    output = []
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        output.append(act[offset : offset + count] @ w[expert].t())
        offset += count
    return [torch.cat(output).to(output_dtype)]


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


class TestNpuBlockFp8(CustomTestCase):
    def setUp(self):
        torch.manual_seed(13)
        # No MX op is supplied: any accidental MX conversion fails immediately.
        npu_ops = SimpleNamespace(
            npu_dynamic_block_quant=_quantize,
            npu_quant_matmul=_matmul,
            npu_grouped_matmul=_grouped_matmul,
            npu_swiglu=_swiglu,
        )
        p = patch.object(torch.ops, "npu", npu_ops, create=True)
        p.start()
        self.addCleanup(p.stop)

    @staticmethod
    def _weight(shape):
        q = (torch.randint(-8, 9, shape).float() / 4).to(torch.float8_e4m3fn)
        # Non-power-of-two, unequal scales expose scale casting and axis errors.
        scale_shape = (*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128)
        scale = torch.arange(1, 1 + torch.tensor(scale_shape).prod().item()).float()
        return q, (scale.reshape(scale_shape) * 0.0037)

    def test_linear_preserves_checkpoint_and_block_scaled_result(self):
        weight, scale = self._weight((256, 384))
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 3, 384, dtype=dtype).transpose(0, 1)
                bias = torch.randn(256, dtype=dtype)
                qx, sx = _quantize(x.reshape(-1, 384))
                expected = (
                    _expand_input(qx, sx) @ _expand_weight(weight, scale).t()
                ).to(dtype) + bias
                actual = npu_w8a8_block_fp8_linear(x, weight, scale, bias=bias)
                torch.testing.assert_close(actual, expected.reshape(3, 2, 256))

    def test_grouped_counts_and_cumsum_with_empty_expert(self):
        weight, scale = self._weight((3, 256, 384))
        x = torch.randn(5, 384, dtype=torch.bfloat16)
        counts = torch.tensor([2, 0, 3], dtype=torch.int64)
        qx, sx = _quantize(x)
        act = _expand_input(qx, sx)
        w = _expand_weight(weight, scale)
        expected = torch.cat((act[:2] @ w[0].t(), act[2:] @ w[2].t())).bfloat16()
        for kind, groups in ((1, counts), (0, counts.cumsum(0))):
            with self.subTest(group_list_type=kind):
                actual = npu_w8a8_block_fp8_grouped_matmul(
                    x, weight, scale, groups, kind, torch.bfloat16
                )
                torch.testing.assert_close(actual, expected)

    def test_linear_post_load_and_reload_preserve_parameters(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        with (
            patch("sglang.srt.layers.quantization.fp8._is_npu", True),
            patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35", return_value=True
            ),
        ):
            method = Fp8LinearMethod(config)
        weight, scale = self._weight((256, 256))
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        original = (layer.weight, layer.weight_scale_inv)
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        for iteration in range(2):
            method.process_weights_after_loading(layer)
            self.assertIs(layer.weight, original[0])
            self.assertIs(layer.weight_scale_inv, original[1])
            torch.testing.assert_close(
                method.apply(layer, x),
                npu_w8a8_block_fp8_linear(x, weight, scale),
            )
            if iteration == 0:
                # A reload must still target the original block-scale layout.
                scale.copy_(scale + 0.001)

    def test_moe_dispatch_loading_and_runner_preserve_block_scales(self):
        from sglang.srt.hardware_backend.npu.quantization.block_fp8_moe import (
            NPUBlockFp8MoEMethod,
        )
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
            AscendTPDispatchOutput,
        )
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.runtime_context import get_flags

        moe = get_flags().moe
        with patch.multiple(
            moe,
            runner_backend=MoeRunnerBackend.AUTO,
            a2a_backend=MoeA2ABackend.ASCEND_TP,
        ):
            config = Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
            )
            with patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35", return_value=True
            ):
                method = config.get_quant_method(FusedMoE.__new__(FusedMoE), "experts")
            self.assertIsInstance(method, NPUBlockFp8MoEMethod)
            layer = FusedMoE.__new__(FusedMoE)
            torch.nn.Module.__init__(layer)
            layer.moe_tp_size = 1
            layer.moe_tp_rank = 0
            layer._expert_storage_rank = 0
            layer._num_local_routed = 3
            layer._num_global_routed = 3
            layer._has_fused_shared = False
            layer.quant_method = method
            layer.quant_config = config
            layer.scheme = None
            layer.use_flashinfer_trtllm_moe = False
            layer.use_triton_kernels = False
            layer.use_padded_loading = False
            layer.use_presharded_weights = False
            layer.moe_runner_config = MoeRunnerConfig(activation="silu", is_gated=True)
            method.create_weights(
                layer,
                num_experts=3,
                hidden_size=256,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
                weight_loader=layer.weight_loader,
            )
            layer.dispatcher = SimpleNamespace(set_quant_config=lambda cfg: None)
            names = (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale_inv",
                "w2_weight_scale_inv",
            )
            for prefix in ("w13", "w2"):
                q, s = self._weight(tuple(getattr(layer, f"{prefix}_weight").shape))
                for expert in range(3):
                    for suffix, checkpoint in (("weight", q), ("weight_scale_inv", s)):
                        param = getattr(layer, f"{prefix}_{suffix}")
                        if prefix == "w13":
                            gate, up = checkpoint[expert].chunk(2, dim=0)
                            param.weight_loader(
                                param,
                                gate,
                                f"experts.{expert}.gate_proj.{suffix}",
                                "w1",
                                expert,
                            )
                            param.weight_loader(
                                param,
                                up,
                                f"experts.{expert}.up_proj.{suffix}",
                                "w3",
                                expert,
                            )
                        else:
                            param.weight_loader(
                                param,
                                checkpoint[expert],
                                f"experts.{expert}.down_proj.{suffix}",
                                "w2",
                                expert,
                            )
                torch.testing.assert_close(
                    getattr(layer, f"{prefix}_weight").view(torch.uint8),
                    q.view(torch.uint8),
                )
                torch.testing.assert_close(
                    getattr(layer, f"{prefix}_weight_scale_inv"), s, rtol=0, atol=0
                )
            originals = [
                (getattr(layer, name), getattr(layer, name).detach().clone())
                for name in names
            ]
            method.create_moe_runner(layer, layer.moe_runner_config)
            method.process_weights_after_loading(layer)
            for name, (parameter, data) in zip(names, originals):
                self.assertIs(getattr(layer, name), parameter)
                # CPU torch does not implement equal for float8; compare bytes.
                torch.testing.assert_close(
                    parameter.view(torch.uint8), data.view(torch.uint8)
                )

            x = torch.randn(5, 256, dtype=torch.bfloat16)
            counts = torch.tensor([2, 0, 3], dtype=torch.int64)
            qx, sx = _quantize(x)
            act = _expand_input(qx, sx)
            w13 = _expand_weight(layer.w13_weight, layer.w13_weight_scale_inv)
            w2 = _expand_weight(layer.w2_weight, layer.w2_weight_scale_inv)
            expected = []
            offset = 0
            for expert, count in enumerate(counts.tolist()):
                if count:
                    middle = _swiglu(
                        (act[offset : offset + count] @ w13[expert].t()).bfloat16()
                    )
                    qm, sm = _quantize(middle)
                    expected.append((_expand_input(qm, sm) @ w2[expert].t()).bfloat16())
                offset += count
            dispatch_output = AscendTPDispatchOutput(
                hidden_states=x,
                hidden_states_scale=None,
                topk_weights=torch.ones(5, 1),
                topk_ids=torch.tensor([[0], [0], [2], [2], [2]]),
                expanded_row_idx=torch.arange(5),
                expert_tokens=counts,
                group_list_type=1,
            )
            actual = method.apply(layer, dispatch_output).hidden_states
            torch.testing.assert_close(actual, torch.cat(expected))

    def test_unsupported_backend_fails_before_gate_up_shard_swap(self):
        from sglang.srt.hardware_backend.npu.quantization.block_fp8_moe import (
            NPUBlockFp8MoEMethod,
        )
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.block_fp8_moe.get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with self.assertRaisesRegex(ValueError, "requires --moe-runner-backend"):
                NPUBlockFp8MoEMethod(Fp8Config())

    def test_moe_dispatch_does_not_claim_other_checkpoint_formats(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        cases = (
            (False, None),
            (True, "ue8m0"),
        )
        for arch35, scale_fmt in cases:
            with self.subTest(arch35=arch35, scale_fmt=scale_fmt):
                config = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[128, 128],
                    scale_fmt=scale_fmt,
                )
                with patch(
                    "sglang.srt.layers.quantization.fp8.is_npu_arch35",
                    return_value=arch35,
                ):
                    method = config.get_quant_method(
                        FusedMoE.__new__(FusedMoE), "experts"
                    )
                self.assertIs(type(method), Fp8MoEMethod)


if __name__ == "__main__":
    unittest.main()
