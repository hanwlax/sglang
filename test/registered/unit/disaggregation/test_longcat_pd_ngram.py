"""Regression tests for LongCat OE history at PD decode admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner_components import ngram_embedding_manager
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _copy_token_table_on_cpu(
    *, tokens, ne_token_table, row_indices, column_starts, req_lens, ignore_tokens
):
    offset = 0
    for row, start, length in zip(row_indices, column_starts, req_lens, strict=True):
        ne_token_table[row, start : start + length] = tokens[offset : offset + length]
        offset += length


class TestLongCatPDNgram(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.table = torch.full((3, 12), -1, dtype=torch.int32)
        self.manager = NgramEmbeddingManager(enabled=True, table=self.table, n=5)
        self.enterContext(
            patch.object(
                ngram_embedding_manager, "update_token_table", _copy_token_table_on_cpu
            )
        )

    def _admit(self, token_sequences, rows):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT,
            reqs=[
                SimpleNamespace(full_untruncated_fill_ids=ids)
                for ids in token_sequences
            ],
            req_pool_indices=torch.tensor(rows, dtype=torch.int64),
            # PREBUILT counts committed KV and excludes the pending token.
            seq_lens_cpu=torch.tensor([len(ids) - 1 for ids in token_sequences]),
            ne_skip_token_table_update=None,
        )
        self.manager.prepare_for_forward(batch, chunked_req=None)
        return batch

    def test_first_decode_seeds_pending_token_and_local_history(self):
        """D skips EXTEND and must overwrite stale history in the reused row."""
        self.table[2].fill_(99)
        batch = self._admit([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23]], [2, 0])
        forward = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens_cpu + 1,
        )
        ForwardBatch._init_ngram_embedding_info(forward, batch, self.table.device)

        self.assertEqual(self.table[2, :8].tolist(), [99, 99, 13, 14, 15, 16, 17, 99])
        self.assertEqual(self.table[0, :4].tolist(), [21, 22, 23, -1])
        self.assertEqual(self.table[1].tolist(), [-1] * 12)

        self.manager.update_after_decode(torch.tensor([18, 24]), forward)
        self.assertEqual(self.table[2, 3:8].tolist(), [14, 15, 16, 17, 18])
        self.assertEqual(self.table[0, :4].tolist(), [21, 22, 23, 24])

    def test_linear_verify_reads_handoff_history_and_overwrites_rejected_tail(self):
        """NEXTN topk=1 verifies from the committed prefix after every accept."""
        batch = self._admit([[11, 12, 13, 14, 15, 16, 17]], [1])

        def stage_verify(committed_len, chain):
            forward = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=1,
                input_ids=torch.tensor(chain),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=torch.tensor([committed_len]),
                spec_info=SimpleNamespace(draft_token_num=len(chain)),
            )
            with patch.object(forward_batch_info, "_is_npu", True):
                ForwardBatch._init_ngram_embedding_info(
                    forward, batch, self.table.device
                )
            self.manager.prepare_for_verify(forward)

        stage_verify(6, [17, 18, 19, 20])
        self.assertEqual(self.table[1, 2:7].tolist(), [13, 14, 15, 16, 17])
        # The first verify consumed 17 and 18 and rejected the remaining tail.
        stage_verify(8, [24, 25, 26, 27])
        self.assertEqual(self.table[1, 4:9].tolist(), [15, 16, 17, 18, 24])
        self.assertEqual(self.table[1, 8:12].tolist(), [24, 25, 26, 27])


if __name__ == "__main__":
    unittest.main()
