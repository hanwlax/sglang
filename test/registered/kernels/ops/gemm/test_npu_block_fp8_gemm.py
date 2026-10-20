"""Real Ascend 950 G-B GEMMs, including Qwen3.6-35B expert shapes."""

import unittest

import torch

from sglang.kernels.ops.gemm import (
    npu_w8a8_block_fp8_grouped_matmul,
    npu_w8a8_block_fp8_linear,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="nightly-acc-4-npu-a5", nightly=True)


@unittest.skipUnless(is_npu_arch35(), "Requires Ascend 950 G-B FP8 kernels")
class TestNpuBlockFp8Kernels(CustomTestCase):
    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def _weight(shape):
        weight = (torch.randint(-8, 9, shape).float() / 4).to(torch.float8_e4m3fn)
        scales = (*shape[:-2], (shape[-2] + 127) // 128, (shape[-1] + 127) // 128)
        scale = torch.rand(scales) * 0.037 + 0.0013
        return weight.npu(), scale.npu()

    @staticmethod
    def _reference(x, w, scale):
        qx, sx = torch.ops.npu.npu_dynamic_block_quant(
            x, dst_type=torch.float8_e4m3fn, row_block_size=1, col_block_size=128
        )
        # Compute independently on CPU FP32 after applying both block scales.
        a = qx.cpu().float() * sx.cpu().repeat_interleave(128, -1)[:, : x.shape[-1]]
        expanded = scale.cpu().repeat_interleave(128, -2).repeat_interleave(128, -1)
        b = w.cpu().float() * expanded[..., : w.shape[-2], : w.shape[-1]]
        return a, b

    def test_linear_prefill_decode_and_bias(self):
        w, s = self._weight((512, 2048))
        original_w, original_s = w.cpu(), s.cpu()
        for m in (1, 33):
            with self.subTest(m=m):
                x = torch.randn(m, 2048, dtype=torch.bfloat16).npu()
                bias = torch.randn(512, dtype=torch.bfloat16).npu()
                a, b = self._reference(x, w, s)
                expected = (a @ b.t()).bfloat16() + bias.cpu()
                actual = npu_w8a8_block_fp8_linear(x, w, s, bias=bias)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(
            w.cpu().view(torch.uint8), original_w.view(torch.uint8)
        )
        torch.testing.assert_close(s.cpu(), original_s, rtol=0, atol=0)

    def test_qwen_expert_projections_with_empty_expert(self):
        # Fused gate/up: [2*512, 2048]; down: [2048, 512].
        counts = torch.tensor([2, 0, 3], dtype=torch.int64)
        for n, k in ((1024, 2048), (2048, 512)):
            with self.subTest(n=n, k=k):
                w, s = self._weight((3, n, k))
                x = torch.randn(5, k, dtype=torch.bfloat16).npu()
                a, b = self._reference(x, w, s)
                expected = torch.cat((a[:2] @ b[0].t(), a[2:] @ b[2].t())).bfloat16()
                for group_type in (0, 1):
                    groups = counts.cumsum(0) if group_type == 0 else counts
                    actual = npu_w8a8_block_fp8_grouped_matmul(
                        x, w, s, groups.npu(), group_type, torch.bfloat16
                    )
                    torch.testing.assert_close(
                        actual.cpu(), expected, rtol=0.02, atol=0.02
                    )

    def test_decode_graph_replay(self):
        w, s = self._weight((512, 2048))
        x = torch.randn(1, 2048, dtype=torch.bfloat16).npu()
        for _ in range(3):
            npu_w8a8_block_fp8_linear(x, w, s)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = npu_w8a8_block_fp8_linear(x, w, s)
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            expected = npu_w8a8_block_fp8_linear(x, w, s)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
