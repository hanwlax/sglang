"""Real A5 V1/V2 prefill parity with 12 heads; no model weights required.

PYTHONPATH=python python test/manual/npu/test_k3_mla_prefill_v2.py
Includes the reported Q=6528, KV=133760, Q/K D=192, V D=128 case.
This is a correctness test, not a performance benchmark or K3 model eval.
"""

import unittest

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.mla_prefill import fias_mha_prefill_v2
from sglang.test.test_utils import CustomTestCase


class TestK3MLAPrefillV2NPU(CustomTestCase):
    def check_pair(self, q, k, v, **kwargs):
        v1 = torch.ops.npu.npu_fused_infer_attention_score(q, k, v, **kwargs)[0]
        v2 = fias_mha_prefill_v2(q, k, v, **kwargs)[0]
        torch.npu.synchronize()
        self.assertEqual(v2.shape, v1.shape)
        self.assertTrue(torch.isfinite(v2).all().item())
        torch.testing.assert_close(v2, v1, atol=2e-3, rtol=2e-2)
        return v2.cpu()

    def params(self, layout):
        return {
            "num_heads": 12,
            "num_key_value_heads": 12,
            "input_layout": layout,
            "atten_mask": torch.ones(2048, 2048, dtype=torch.bool).triu(1).npu(),
            "sparse_mode": 3,
            "scale": 0.125,
            "next_tokens": 0,
        }

    def test_small_cached_against_fp32(self):
        torch.manual_seed(42)
        for dtype in (torch.bfloat16, torch.float16):
            q = torch.randn(1, 17, 12, 192, dtype=dtype) * 0.1
            k = torch.randn(1, 273, 12, 192, dtype=dtype) * 0.1
            v = torch.randn(1, 273, 12, 128, dtype=dtype) * 0.1
            scores = (
                q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)
            ) * 0.125
            allowed = torch.arange(273)[None, :] <= (256 + torch.arange(17))[:, None]
            expected = (
                (
                    scores.masked_fill(~allowed, float("-inf")).softmax(-1)
                    @ v.float().transpose(1, 2)
                )
                .transpose(1, 2)
                .to(dtype)
            )
            actual = self.check_pair(q.npu(), k.npu(), v.npu(), **self.params("BSND"))
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)

    def test_cold_tnd_ragged_and_suffix(self):
        torch.manual_seed(43)
        q = (torch.randn(25, 12, 192, dtype=torch.bfloat16) * 0.1).npu()
        k = (torch.randn(25, 12, 192, dtype=torch.bfloat16) * 0.1).npu()
        v = (torch.randn(25, 12, 128, dtype=torch.bfloat16) * 0.1).npu()
        self.check_pair(
            q[..., :128],
            k[..., :128].contiguous(),
            v,
            query_rope=q[..., 128:],
            key_rope=k[..., 128:].contiguous(),
            actual_seq_lengths=[8, 25],
            actual_seq_lengths_kv=[8, 25],
            **self.params("TND"),
        )

    def test_reported_long_prefill_shape(self):
        # About 1.1GB of inputs plus operator workspaces; no padded heads.
        torch.manual_seed(44)
        q = torch.randn(1, 6528, 12, 192, dtype=torch.bfloat16, device="npu") * 0.1
        k = torch.randn(1, 133760, 12, 192, dtype=torch.bfloat16, device="npu") * 0.1
        v = torch.randn(1, 133760, 12, 128, dtype=torch.bfloat16, device="npu") * 0.1
        result = self.check_pair(q, k, v, **self.params("BSND"))
        self.assertEqual(result.shape, (1, 6528, 12, 128))


if __name__ == "__main__":
    unittest.main()
