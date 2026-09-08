"""Fused NZ indices with native scatter writes and dynamic graph replay."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sglang.kernels.ops.kvcache.triton_mla_nz_indices import build_mla_nz_indices
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMLATokenToKVPool,
    _mla_fia_nz_scatter_indices,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=80, suite="stage-a-unit-test-npu")


class TestMLANZIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("Requires Ascend NPU")
        cls.device = "npu:0"

    def _pool(self):
        pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
        pool.dtype = pool.store_dtype = torch.bfloat16
        pool.start_layer, pool.page_size = 13, 128
        pool.kv_lora_rank, pool.qk_rope_head_dim = 512, 64
        pool.use_fia_nz = True
        pool.k_buffer = torch.zeros(
            1, 4, 128, 1, 512, device=self.device, dtype=pool.dtype
        )
        pool.v_buffer = torch.zeros(
            1, 4, 128, 1, 64, device=self.device, dtype=pool.dtype
        )
        return pool

    def _assert_cache(self, pool, refs):
        ids = torch.tensor([3, 1, 3, 2, 0], device=self.device, dtype=torch.int32)
        for cache, ref in zip((pool.k_buffer[0], pool.v_buffer[0]), refs):
            packed = (
                ref.reshape(4, 128, -1, 16)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape_as(ref)
            )
            self.assertTrue(torch.equal(cache.cpu(), packed))
            self.assertTrue(
                torch.equal(
                    gather_mla_cache_pages(cache, ids, is_nz=True).cpu(),
                    ref[[3, 1, 3, 2, 0]],
                )
            )

    def test_indices_boundaries_strides_and_empty(self):
        for page_size in (16, 96, 128):
            for head_dim in (64, 512):
                for dtype in (torch.int32, torch.int64):
                    for count in (0, 1, 7, 65):
                        with self.subTest(
                            P=page_size, D=head_dim, dtype=dtype, N=count
                        ):
                            slots = torch.arange(count * 2, dtype=dtype) * 127 % 512
                            loc = slots.to(self.device)[::2]
                            expected = _mla_fia_nz_scatter_indices(
                                slots[::2].long(), head_dim, page_size
                            ).flatten()
                            actual = build_mla_nz_indices(
                                loc,
                                page_size,
                                head_dim,
                                (512 + page_size - 1) // page_size,
                            ).cpu()
                            self.assertEqual(actual.dtype, torch.int64)
                            self.assertTrue(torch.equal(actual, expected))

    def test_large_address_indices(self):
        for blocks in (2**31 // 4096 - 1, 2**31 // 4096, 2**31 // 4096 + 1):
            slots = torch.tensor([0, 127, 128, blocks * 128 - 1], dtype=torch.int64)
            actual = build_mla_nz_indices(slots.to(self.device), 128, 512, blocks)
            expected = _mla_fia_nz_scatter_indices(slots, 512, 128).flatten()
            self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_pool_partial_writes_and_prefix_reads(self):
        pool = self._pool()
        refs = [
            torch.zeros_like(buf[0], device="cpu")
            for buf in (pool.k_buffer, pool.v_buffer)
        ]
        for slots in ([127, 128, 255, 256, 383, 511], [128, 129, 384], []):
            loc = torch.tensor(slots, dtype=torch.int32, device=self.device)
            # Splitting the combined latent/RoPE source produces strided values.
            values = torch.randn(len(slots), 1, 576, dtype=pool.dtype)
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=13), loc, values.to(self.device), None
            )
            for ref, value in zip(refs, values.split([512, 64], dim=-1)):
                ref.view(-1, 1, ref.shape[-1])[slots] = value
            self._assert_cache(pool, refs)

    def test_graph_replay_updates_locations_and_values(self):
        pool = self._pool()
        loc = torch.tensor([127, 128, 255, 256], device=self.device, dtype=torch.int32)
        backing = torch.randn(4, 1, 576, device=self.device, dtype=pool.dtype)

        def write():
            pool.set_kv_buffer(SimpleNamespace(layer_id=13), loc, backing, None)

        write()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write()
        for slots in ([129, 257, 383, 511], [0, 126, 254, 510]):
            pool.k_buffer.zero_()
            pool.v_buffer.zero_()
            values = torch.randn(4, 1, 576, dtype=pool.dtype)
            backing.copy_(values)
            loc.copy_(torch.tensor(slots, dtype=loc.dtype, device=self.device))
            graph.replay()
            torch.npu.synchronize()
            refs = [
                torch.zeros_like(buf[0], device="cpu")
                for buf in (pool.k_buffer, pool.v_buffer)
            ]
            for ref, value in zip(refs, values.split([512, 64], dim=-1)):
                ref.view(-1, 1, ref.shape[-1])[slots] = value
            self._assert_cache(pool, refs)


if __name__ == "__main__":
    unittest.main()
