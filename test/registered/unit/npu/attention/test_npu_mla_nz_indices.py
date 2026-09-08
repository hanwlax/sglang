"""Fused NZ indices with native scatter writes and dynamic graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from sglang.kernels.ops.kvcache.triton_mla_nz_indices import build_mla_nz_indices
from sglang.srt.hardware_backend.npu.attention.mla_cache import (
    gather_mla_cache_pages,
    get_mla_nz_indices,
    with_mla_nz_index_cache,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMLATokenToKVPool,
    _mla_fia_nz_scatter_indices,
)
from sglang.srt.runtime_context import get_forward
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

    def test_token_counts_share_compiled_kernel(self):
        from sglang.kernels.ops.kvcache.triton_mla_nz_indices import (
            _mla_nz_indices_kernel,
        )

        loc = torch.arange(256, device=self.device, dtype=torch.int32)
        out = torch.empty(256 * 32, device=self.device, dtype=torch.int64)
        hashes = set()
        for count in (1, 7, 48, 65, 88, 128, 255):
            compiled = _mla_nz_indices_kernel[((count * 32 + 255) // 256,)](
                loc, out, count, 128, 7, 32, 1, True, 256
            )
            hashes.add(compiled.hash)
            expected = _mla_fia_nz_scatter_indices(loc[:count].cpu().long(), 512, 128)
            self.assertTrue(torch.equal(out[: count * 32].cpu(), expected.flatten()))
        self.assertEqual(len(hashes), 1)

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

        @with_mla_nz_index_cache
        def write():
            pool.set_kv_buffer(SimpleNamespace(layer_id=13), loc, backing, None)
            # A second layer must read the same captured index producers.
            pool.set_kv_buffer(SimpleNamespace(layer_id=14), loc, backing, None)

        pool.k_buffer = pool.k_buffer.repeat(2, 1, 1, 1, 1)
        pool.v_buffer = pool.v_buffer.repeat(2, 1, 1, 1, 1)
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
            self.assertTrue(torch.equal(pool.k_buffer[0], pool.k_buffer[1]))
            self.assertTrue(torch.equal(pool.v_buffer[0], pool.v_buffer[1]))

    def test_shared_indices_scopes_and_mutations(self):
        loc = torch.tensor([127, 128, 255], device=self.device, dtype=torch.int32)

        def get():
            return get_mla_nz_indices(loc, 128, 512, 4)

        with patch(
            "sglang.kernels.ops.kvcache.triton_mla_nz_indices.build_mla_nz_indices",
            wraps=build_mla_nz_indices,
        ) as build:
            with get_forward().scoped(npu_mla_nz_indices={}):
                first = get()
                for _ in range(23):
                    self.assertIs(get(), first)
                self.assertEqual(build.call_count, 1)
                with (
                    self.assertRaisesRegex(RuntimeError, "nested"),
                    get_forward().scoped(npu_mla_nz_indices={}),
                ):
                    self.assertIsNot(get(), first)
                    raise RuntimeError("nested")
                self.assertIs(get(), first)
                loc.add_(1)
                self.assertIsNot(get(), first)
            self.assertIsNone(get_forward().npu_mla_nz_indices)
            self.assertIsNot(get(), first)
            self.assertEqual(build.call_count, 4)

    @torch.inference_mode()
    def test_inference_locations_refresh_each_forward(self):
        loc = torch.tensor([127, 128, 255], device=self.device, dtype=torch.int32)

        @with_mla_nz_index_cache
        def run():
            return get_mla_nz_indices(loc, 128, 512, 4)

        first = run()
        loc.add_(1)
        second = run()
        self.assertIsNot(first, second)
        expected = _mla_fia_nz_scatter_indices(loc.cpu().long(), 512, 128).flatten()
        self.assertTrue(torch.equal(second.cpu(), expected))

    def test_streams_do_not_share_producers(self):
        loc = torch.tensor([127, 128, 255], device=self.device, dtype=torch.int32)
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with get_forward().scoped(npu_mla_nz_indices={}):
            first = get_mla_nz_indices(loc, 128, 512, 4)
            with torch.npu.stream(stream):
                second = get_mla_nz_indices(loc, 128, 512, 4)
        torch.npu.synchronize()
        self.assertIsNot(first, second)
        self.assertTrue(torch.equal(first.cpu(), second.cpu()))

    def test_compile_bypasses_contextvars(self):
        @with_mla_nz_index_cache
        def run(x):
            return x + 1

        compiled = torch.compile(run, backend="eager", fullgraph=True)
        self.assertTrue(torch.equal(compiled(torch.zeros(2)), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
