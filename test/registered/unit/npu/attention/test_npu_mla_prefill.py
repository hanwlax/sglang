"""CPU regression tests for ragged FIAS V2 prefill and explicit NZ storage."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.hardware_backend.npu.attention.mla_prefill import fias_v2_mla_prefill
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def pack_nz(logical):
    pages, page_size, _, dim = logical.shape
    return (
        logical.reshape(pages, page_size, dim // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape_as(logical)
    )


class TestMLAPrefillV2(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.page_size = 128
        self.cache = torch.randn(5, 128, 1, 512, dtype=torch.bfloat16) * 0.1
        self.rope_cache = torch.randn(5, 128, 1, 64, dtype=torch.bfloat16) * 0.1
        self.table = torch.tensor([[3, 0, 4], [1, 4, 0], [2, 1, 3]], dtype=torch.int32)
        self.mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1)
        self.scale = 0.1

    @staticmethod
    def _unpack(cache):
        if cache.ndim == 5:
            b, n, tiles, p, d0 = cache.shape
            return cache.permute(0, 3, 1, 2, 4).reshape(b, p, n, tiles * d0)
        return cache.transpose(1, 2)

    def _kernel(self, q, key, value, **kw):
        # Interpret the actual operator arguments, not the caller's lengths
        # or tensors. In particular, never re-use the logical cache fixture.
        self.assertEqual(kw["input_layout"], "BNSD")
        self.assertEqual(kw["sparse_mode"], 3)
        self.assertEqual(kw["next_tokens"], 0)
        self.assertEqual(kw["num_key_value_heads"], 1)
        self.assertIs(kw["atten_mask"], self.mask)
        self.assertEqual(key.data_ptr(), value.data_ptr())
        self.assertTrue(q.is_contiguous())
        self.assertTrue(kw["query_rope"].is_contiguous())
        self.assertTrue(kw["block_table"].is_contiguous())
        n, s = q.shape[1:3]
        self.assertEqual(kw["actual_seq_qlen"], [s])
        self.assertEqual(kw["num_query_heads"], n)
        kv_len = kw["actual_seq_kvlen"][0]
        ids = kw["block_table"][0].long()
        k = self._unpack(key)[ids].reshape(-1, 512)[:kv_len].float()
        kr = self._unpack(kw["key_rope"])[ids].reshape(-1, 64)[:kv_len].float()
        logits = (q[0].float() @ k.T + kw["query_rope"][0].float() @ kr.T) * kw[
            "softmax_scale"
        ]
        causal = (
            torch.arange(kv_len)[None, :] <= (kv_len - s + torch.arange(s))[:, None]
        )
        probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
        return (probs @ k).unsqueeze(0).to(q.dtype), None

    def _expected(self, q, qr, q_lens, kv_lens):
        out = torch.zeros_like(q)
        offset = 0
        for row, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
            if not ql:
                continue
            ids = self.table[row].long()
            k = self.cache[ids].reshape(-1, 512)[:kl].float()
            kr = self.rope_cache[ids].reshape(-1, 64)[:kl].float()
            # Explicit token loop independently defines right-aligned causality.
            for i in range(ql):
                end = kl - ql + i + 1
                logits = (
                    q[offset + i].float() @ k[:end].T
                    + qr[offset + i].float() @ kr[:end].T
                ) * self.scale
                out[offset + i] = (logits.softmax(-1) @ k[:end]).to(q.dtype)
            offset += ql
        return out

    def test_cold_cached_ragged_and_padding(self):
        for q_lens, kv_lens in (
            ([129, 1, 2], [129, 1, 2]),
            ([3, 0, 5], [131, 0, 259]),
            ([1, 8, 17], [128, 256, 260]),
        ):
            for heads in (3, 4):
                q = torch.randn(sum(q_lens) + 2, heads, 512, dtype=torch.bfloat16) * 0.1
                # Non-contiguous query_rope, like the slice from q_b_proj.
                qr = (torch.randn(q.shape[0], heads, 576, dtype=q.dtype) * 0.1)[
                    ..., -64:
                ]
                self.assertFalse(qr.is_contiguous())
                expected = self._expected(q, qr, q_lens, kv_lens)
                for nz in (False, True):
                    with self.subTest(q_lens=q_lens, heads=heads, nz=nz):
                        k = pack_nz(self.cache) if nz else self.cache
                        kr = pack_nz(self.rope_cache) if nz else self.rope_cache
                        before = k.clone()
                        kernel = Mock(side_effect=self._kernel)
                        with patch.dict(
                            sys.modules,
                            {
                                "torch_npu": SimpleNamespace(
                                    npu_fused_infer_attention_score_v2=kernel
                                )
                            },
                        ):
                            actual = fias_v2_mla_prefill(
                                q,
                                qr,
                                k,
                                kr,
                                query_lens=q_lens,
                                kv_lens=kv_lens,
                                block_table=self.table,
                                page_size=128,
                                scale=self.scale,
                                mask=self.mask,
                                is_nz=nz,
                            )
                        torch.testing.assert_close(
                            actual, expected, atol=2e-4, rtol=2e-2
                        )
                        self.assertTrue(torch.equal(k, before))
                        self.assertEqual(kernel.call_count, sum(v > 0 for v in q_lens))
                        self.assertEqual(
                            [
                                c.kwargs["actual_seq_kvlen"]
                                for c in kernel.call_args_list
                            ],
                            [[kl] for ql, kl in zip(q_lens, kv_lens) if ql],
                        )
                        for call in kernel.call_args_list:
                            self.assertEqual(call.args[1].data_ptr(), k.data_ptr())
                            self.assertEqual(call.args[1].ndim, 5 if nz else 4)
                        self.assertEqual(torch.count_nonzero(actual[-2:]).item(), 0)

    def test_zero_query_rows_and_empty_dp(self):
        for lengths in ([], [0, 0, 0]):
            kernel = Mock()
            with patch.dict(
                sys.modules,
                {
                    "torch_npu": SimpleNamespace(
                        npu_fused_infer_attention_score_v2=kernel
                    )
                },
            ):
                output = fias_v2_mla_prefill(
                    torch.zeros(2, 4, 512, dtype=torch.bfloat16),
                    torch.zeros(2, 4, 64, dtype=torch.bfloat16),
                    pack_nz(self.cache),
                    pack_nz(self.rope_cache),
                    query_lens=lengths,
                    kv_lens=lengths,
                    block_table=self.table[: len(lengths)],
                    page_size=128,
                    scale=self.scale,
                    mask=self.mask,
                    is_nz=True,
                )
            kernel.assert_not_called()
            self.assertEqual(output.count_nonzero().item(), 0)

    def test_rejects_invalid_lengths_before_launch(self):
        for ql, kl in (([2], []), ([2], [1]), ([2], [385]), ([4], [4]), ([-1], [0])):
            kernel = Mock()
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch_npu": SimpleNamespace(
                            npu_fused_infer_attention_score_v2=kernel
                        )
                    },
                ),
                self.assertRaises(ValueError),
            ):
                fias_v2_mla_prefill(
                    torch.zeros(3, 4, 512, dtype=torch.bfloat16),
                    torch.zeros(3, 4, 64, dtype=torch.bfloat16),
                    self.cache,
                    self.rope_cache,
                    query_lens=ql,
                    kv_lens=kl,
                    block_table=self.table[:1],
                    page_size=128,
                    scale=self.scale,
                    mask=self.mask,
                    is_nz=False,
                )
            kernel.assert_not_called()

    def test_mla_absorption_preserves_skip_rope_attention(self):
        # Mathematical reference in float64: K3's 64-d suffix is still part
        # of QK even though no rotary embedding is applied to it.
        q = torch.randn(5, 4, 128, dtype=torch.float64)
        qr = torch.randn(5, 4, 64, dtype=torch.float64)
        c = torch.randn(13, 512, dtype=torch.float64)
        kr = torch.randn(13, 64, dtype=torch.float64)
        wk = torch.randn(4, 128, 512, dtype=torch.float64) * 0.01
        wv = torch.randn(4, 512, 128, dtype=torch.float64) * 0.01
        expanded_k = torch.einsum("sl,hdl->shd", c, wk)
        expanded_v = torch.einsum("sl,hlv->shv", c, wv)
        mask = torch.arange(13)[None, :] <= (8 + torch.arange(5))[:, None]
        logits = (
            torch.einsum("thd,shd->hts", q, expanded_k)
            + torch.einsum("thr,sr->hts", qr, kr)
        ) * self.scale
        expected = torch.einsum(
            "hts,shv->thv",
            logits.masked_fill(~mask, float("-inf")).softmax(-1),
            expanded_v,
        )
        absorbed_q = torch.bmm(q.transpose(0, 1), wk).transpose(0, 1)
        logits = (
            torch.einsum("thl,sl->hts", absorbed_q, c)
            + torch.einsum("thr,sr->hts", qr, kr)
        ) * self.scale
        latent_out = torch.einsum(
            "hts,sl->thl", logits.masked_fill(~mask, float("-inf")).softmax(-1), c
        )
        actual = torch.einsum("thl,hlv->thv", latent_out, wv)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
