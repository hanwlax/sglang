"""P0 regressions: V2 prefill must preserve expanded tensors and true heads."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sglang.srt.hardware_backend.npu.attention.mla_prefill import fias_mha_prefill_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMLAPrefillV2(CustomTestCase):
    def call(self, q, k, v, **kwargs):
        expected = (torch.empty_like(v), None)
        kernel = Mock(return_value=expected)
        with patch.dict(
            sys.modules,
            {"torch_npu": SimpleNamespace(npu_fused_infer_attention_score_v2=kernel)},
        ):
            result = fias_mha_prefill_v2(
                q,
                k,
                v,
                num_heads=12,
                atten_mask=self.mask,
                sparse_mode=3,
                scale=0.125,
                next_tokens=0,
                **kwargs,
            )
        self.assertIs(result, expected)
        for actual, original in zip(kernel.call_args.args, (q, k, v)):
            self.assertIs(actual, original)
        args = kernel.call_args.kwargs
        self.assertEqual(args["num_query_heads"], 12)
        self.assertEqual(args["softmax_scale"], 0.125)
        self.assertIs(args["atten_mask"], self.mask)
        self.assertEqual(args["sparse_mode"], 3)
        self.assertEqual(args["next_tokens"], 0)
        self.assertNotIn("block_table", args)
        self.assertNotIn("num_heads", args)
        self.assertNotIn("scale", args)
        return args

    def setUp(self):
        super().setUp()
        self.mask = torch.ones(2048, 2048, dtype=torch.bool, device="meta")

    def test_reported_long_prefill_shapes_no_padding_or_latent_projection(self):
        # Meta tensors cover the exact 18ms baseline shape without allocating GBs.
        q = torch.empty(1, 6528, 12, 192, dtype=torch.bfloat16, device="meta")
        k = torch.empty(1, 133760, 12, 192, dtype=q.dtype, device="meta")
        v = torch.empty(1, 133760, 12, 128, dtype=q.dtype, device="meta")
        args = self.call(q, k, v, input_layout="BSND", num_key_value_heads=12)
        self.assertEqual(args["num_key_value_heads"], 12)
        self.assertIsNone(args["actual_seq_qlen"])
        self.assertIsNone(args["actual_seq_kvlen"])
        self.assertIsNone(args["query_rope"])
        self.assertIsNone(args["key_rope"])

    def test_cold_tnd_preserves_suffix_and_cumulative_lengths(self):
        q = torch.randn(8, 12, 192)
        k = torch.randn(8, 12, 192)
        v = torch.randn(8, 12, 128)
        qr, kr = q[..., 128:], k[..., 128:]
        args = self.call(
            q[..., :128],
            k[..., :128],
            v,
            input_layout="TND",
            query_rope=qr,
            key_rope=kr,
            actual_seq_lengths=np.array([3, 8], dtype=np.int64),
            actual_seq_lengths_kv=np.array([3, 8], dtype=np.int64),
        )
        self.assertEqual(args["actual_seq_qlen"], [3, 8])
        self.assertEqual(args["actual_seq_kvlen"], [3, 8])
        self.assertTrue(all(type(n) is int for n in args["actual_seq_qlen"]))
        self.assertIs(args["query_rope"], qr)
        self.assertIs(args["key_rope"], kr)
        self.assertEqual(args["input_layout"], "TND")

    def test_single_token_and_equal_dim_shapes_are_not_rewritten(self):
        for tokens in (1, 17):
            q = torch.randn(1, tokens, 12, 128)
            args = self.call(q, q, q, input_layout="BSND", num_key_value_heads=12)
            self.assertEqual(args["input_layout"], "BSND")


if __name__ == "__main__":
    unittest.main()
