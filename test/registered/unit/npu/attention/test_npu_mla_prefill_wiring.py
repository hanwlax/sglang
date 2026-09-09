"""Run the real routing/backend bodies with CPU dependencies, without a model."""

import ast
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SRT = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def definition(file, name, scope=None):
    body = ast.parse((SRT / file).read_text()).body
    if scope:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == scope
        ).body
    return next(
        n
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )


def load(nodes, ns):
    exec(  # noqa: S102 -- selected production bodies from this checkout
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "<production-prefill-bodies>",
            "exec",
        ),
        ns,
    )
    return ns


class TestMLAPrefillWiring(CustomTestCase):
    def setUp(self):
        super().setUp()
        ns = load(
            [
                definition("model_executor/forward_batch_info.py", "ForwardMode"),
                definition(
                    "models/deepseek_common/attention_forward_methods/forward_methods.py",
                    "AttnForwardMethod",
                ),
            ],
            {"IntEnum": IntEnum, "auto": auto, "Enum": IntEnum},
        )
        self.mode, self.method = ns["ForwardMode"], ns["AttnForwardMethod"]

    def test_k3_route_not_changed_by_prefill_flag(self):
        method = self.method

        class Base:
            def dispatch_attn_forward_method(self, batch):
                return (
                    method.MHA_NPU
                    if batch.forward_mode.is_extend_without_speculative()
                    else method.MLA_NPU
                )

        cls = ast.ClassDef(
            name="K3",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            decorator_list=[],
            body=[
                definition(
                    "models/kimi_k3.py",
                    "dispatch_attn_forward_method",
                    "KimiK3MLAAttention",
                )
            ],
        )
        obj = load([cls], {"Base": Base, "envs": envs, "AttnForwardMethod": method})[
            "K3"
        ]()
        for enabled in (False, True):
            with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(enabled):
                for mode in self.mode:
                    batch = SimpleNamespace(forward_mode=mode)
                    self.assertEqual(
                        obj.dispatch_attn_forward_method(batch),
                        Base.dispatch_attn_forward_method(obj, batch),
                    )
                obj._kimi_split_gguf_kv_b = True
                self.assertEqual(obj.dispatch_attn_forward_method(batch), method.MLA)
                obj._kimi_split_gguf_kv_b = False

    def backend(self, nz=False):
        def output(q, k, v, **kw):
            return (q.new_zeros(*q.shape[:-1], v.shape[-1]), None)

        v1, v2 = Mock(side_effect=output), Mock(side_effect=output)
        # Preserve actual tensor operations but inject only the native API dependency.
        torch_proxy = SimpleNamespace(
            **{
                n: getattr(torch, n)
                for n in ("empty", "empty_like", "cat", "contiguous_format")
            }
        )
        torch_proxy.Tensor = torch.Tensor
        torch_proxy.ops = SimpleNamespace(
            npu=SimpleNamespace(npu_fused_infer_attention_score=v1)
        )
        ns = load(
            [
                definition(
                    "hardware_backend/npu/attention/ascend_backend.py",
                    "forward_extend",
                    "AscendAttnBackend",
                )
            ],
            {
                "torch": torch_proxy,
                "envs": envs,
                "Optional": __import__("typing").Optional,
                "RadixAttention": object,
                "ForwardBatch": object,
                "is_mla_preprocess_enabled": lambda: False,
                "is_fia_nz": lambda: nz,
                "gather_mla_cache_pages": gather_mla_cache_pages,
                "fias_mha_prefill_v2": v2,
            },
        )
        backend = SimpleNamespace(
            use_mla=True,
            is_dllm_model=False,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            fia_mask=torch.empty(2048, 2048, dtype=torch.bool, device="meta"),
            forward_mtp=Mock(return_value="verify"),
        )
        layer = SimpleNamespace(
            layer_id=3,
            tp_q_head_num=12,
            tp_k_head_num=12,
            qk_head_dim=192,
            v_head_dim=128,
            scaling=0.125,
        )
        return ns["forward_extend"], backend, layer, v1, v2

    def test_cached_nd_nz_have_identical_expanded_inputs(self):
        torch.manual_seed(8)
        logical = torch.randn(3, 16, 1, 512)
        rope = torch.randn(3, 16, 1, 64)
        q, k, v = (
            torch.randn(5, 12, 192),
            torch.randn(5, 12, 192),
            torch.randn(5, 12, 128),
        )
        weight = torch.randn(512, 12 * 256)
        calls = []
        for nz in (False, True):

            def pack(x, nz=nz):
                return (
                    x.reshape(3, 16, -1, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape_as(x)
                    if nz
                    else x
                )

            for enabled in (False, True):
                fn, backend, layer, v1, v2 = self.backend(nz)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_key_buffer=lambda _: pack(logical),
                    get_value_buffer=lambda _: pack(rope),
                )
                layer.kv_b_proj = Mock(side_effect=lambda x: (x @ weight,))
                backend.forward_metadata = SimpleNamespace(
                    flatten_prefix_block_tables=torch.tensor([2, 1, 0]),
                    extend_seq_lens_cpu_int=torch.tensor([3, 2]),
                    prefix_lens=torch.tensor([16, 32]),
                )
                batch = SimpleNamespace(
                    forward_mode=self.mode.EXTEND, extend_prefix_lens_cpu=[16, 32]
                )
                with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(enabled):
                    out = fn(backend, q, k, v, layer, batch, save_kv_cache=False)
                self.assertEqual(out.shape, (5, 12 * 128))
                selected, unused = (v2, v1) if enabled else (v1, v2)
                unused.assert_not_called()
                self.assertEqual(selected.call_count, 2)
                layer.kv_b_proj.assert_called_once()
                for call, ql, kl in zip(selected.call_args_list, [3, 2], [19, 34]):
                    self.assertEqual(call.args[0].shape, (1, ql, 12, 192))
                    self.assertEqual(call.args[1].shape, (1, kl, 12, 192))
                    self.assertEqual(call.args[2].shape, (1, kl, 12, 128))
                    self.assertEqual(call.kwargs["num_heads"], 12)
                    self.assertEqual(call.kwargs["input_layout"], "BSND")
                calls.append(selected.call_args_list)
        for run in calls[1:]:
            for ref, cur in zip(calls[0], run):
                for a, b in zip(ref.args, cur.args):
                    torch.testing.assert_close(a, b)

    def test_cold_ragged_tnd_padding_and_verify_are_preserved(self):
        for enabled in (False, True):
            fn, backend, layer, v1, v2 = self.backend()
            backend.forward_metadata = SimpleNamespace(
                seq_lens_list_cumsum=np.array([3, 8])
            )
            batch = SimpleNamespace(
                forward_mode=self.mode.EXTEND,
                extend_prefix_lens_cpu=[0, 0],
                num_token_non_padded_cpu=8,
            )
            q, k, v = (
                torch.randn(10, 12, 192),
                torch.randn(10, 12, 192),
                torch.randn(10, 12, 128),
            )
            with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(enabled):
                out = fn(backend, q, k, v, layer, batch, save_kv_cache=False)
                selected, unused = (v2, v1) if enabled else (v1, v2)
                unused.assert_not_called()
                selected.assert_called_once()
                self.assertEqual(selected.call_args.args[0].shape, (8, 12, 128))
                self.assertEqual(
                    selected.call_args.kwargs["query_rope"].shape, (8, 12, 64)
                )
                self.assertEqual(selected.call_args.kwargs["input_layout"], "TND")
                self.assertEqual(
                    list(selected.call_args.kwargs["actual_seq_lengths"]), [3, 8]
                )
                self.assertEqual(out.shape, (10, 12, 128))
                self.assertEqual(out[-2:].count_nonzero().item(), 0)
                for mode in (self.mode.TARGET_VERIFY, self.mode.DRAFT_EXTEND_V2):
                    batch.forward_mode = mode
                    self.assertEqual(fn(backend, q, k, v, layer, batch), "verify")
                selected.assert_called_once()

    def test_cold_equal_dim_call_uses_real_heads(self):
        for enabled in (False, True):
            fn, backend, layer, v1, v2 = self.backend()
            layer.qk_head_dim = 128
            batch = SimpleNamespace(
                forward_mode=self.mode.EXTEND,
                extend_prefix_lens_cpu=[0, 0],
                extend_seq_lens_cpu=[1, 3],
            )
            q = torch.randn(4, 12, 128)
            with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(enabled):
                out = fn(backend, q, q, q, layer, batch, save_kv_cache=False)
            selected, unused = (v2, v1) if enabled else (v1, v2)
            unused.assert_not_called()
            self.assertEqual(out.shape, (4, 12 * 128))
            self.assertEqual(selected.call_count, 2)
            self.assertEqual(selected.call_args_list[0].kwargs["sparse_mode"], 0)
            self.assertEqual(selected.call_args_list[1].kwargs["sparse_mode"], 3)


if __name__ == "__main__":
    unittest.main()
