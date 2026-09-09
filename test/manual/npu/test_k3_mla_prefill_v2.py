"""Real-device ND/NZ paged MLA prefill smoke test; no model weights needed.

Run in the deployment's A5/CANN/torch_npu environment:
    PYTHONPATH=python python test/manual/npu/test_k3_mla_prefill_v2.py
This tests the attention call, not end-to-end K3 accuracy or distributed hangs.
"""

import unittest

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.mla_prefill import fias_v2_mla_prefill
from sglang.test.test_utils import CustomTestCase


def pack_nz(cache):
    blocks, page, _, dim = cache.shape
    return (
        cache.reshape(blocks, page, dim // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape_as(cache)
    )


class TestK3MLAPrefillV2NPU(CustomTestCase):
    def test_cold_and_cached_variable_length_prefill(self):
        torch.manual_seed(42)
        page = 128
        for dtype in (torch.bfloat16, torch.float16):
            # Small cold prefill, then the 128K-prefix shape including a
            # zero-length row and one-token tail. Each case has DP padding.
            for q_lens, kv_lens in (
                ([256, 17, 1], [256, 17, 1]),
                ([17, 0, 1], [128017, 128000, 128001]),
            ):
                blocks = (max(kv_lens) + page - 1) // page
                cache = torch.randn(blocks, page, 1, 512, dtype=dtype) * 0.1
                rope = torch.randn(blocks, page, 1, 64, dtype=dtype) * 0.1
                table = torch.stack([torch.randperm(blocks) for _ in q_lens]).int()
                q = torch.randn(sum(q_lens) + 2, 4, 512, dtype=dtype) * 0.1
                qr = torch.randn(sum(q_lens) + 2, 4, 64, dtype=dtype) * 0.1
                expected = torch.zeros_like(q)
                offset = 0
                for row, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
                    if ql == 0:
                        continue
                    k = cache[table[row].long()].reshape(-1, 512)[:kl].float()
                    kr = rope[table[row].long()].reshape(-1, 64)[:kl].float()
                    scores = (
                        q[offset : offset + ql].float().transpose(0, 1) @ k.T
                        + qr[offset : offset + ql].float().transpose(0, 1) @ kr.T
                    ) * 0.1
                    allowed = (
                        torch.arange(kl)[None, :]
                        <= (kl - ql + torch.arange(ql))[:, None]
                    )
                    expected[offset : offset + ql] = (
                        (scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ k)
                        .transpose(0, 1)
                        .to(dtype)
                    )
                    offset += ql
                q_npu, qr_npu = q.npu(), qr.npu()
                table_npu = table.npu()
                mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).npu()
                outputs = []
                for nz in (False, True):
                    with self.subTest(dtype=dtype, q_lens=q_lens, nz=nz):
                        k = (pack_nz(cache) if nz else cache).npu()
                        kr = (pack_nz(rope) if nz else rope).npu()
                        out = fias_v2_mla_prefill(
                            q_npu,
                            qr_npu,
                            k,
                            kr,
                            query_lens=q_lens,
                            kv_lens=kv_lens,
                            block_table=table_npu,
                            page_size=page,
                            scale=0.1,
                            mask=mask,
                            is_nz=nz,
                        ).cpu()
                        self.assertTrue(torch.isfinite(out).all().item())
                        torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-2)
                        self.assertEqual(out[-2:].count_nonzero().item(), 0)
                        outputs.append(out)
                torch.testing.assert_close(outputs[0], outputs[1], atol=2e-3, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
