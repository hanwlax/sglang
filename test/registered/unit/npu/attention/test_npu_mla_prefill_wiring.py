"""Exercise real prefill routing/preparation bodies without importing a model.

AST loading avoids importing K3's MoE/vision/accelerator dependencies on CPU;
the selected production functions are compiled unmodified from the checkout.
"""

import ast
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention import mla_prefill
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def definition(file, name, *, scope=None):
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


def load(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(  # noqa: S102 -- compile only selected functions from this checkout
        compile(
            ast.fix_missing_locations(module), "<production-prefill-bodies>", "exec"
        ),
        namespace,
    )
    return namespace


class TestMLAPrefillWiring(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.ns = load(
            [
                definition("model_executor/forward_batch_info.py", "ForwardMode"),
                definition(
                    "models/deepseek_common/attention_forward_methods/forward_methods.py",
                    "AttnForwardMethod",
                ),
            ],
            {"IntEnum": IntEnum, "auto": auto, "Enum": IntEnum},
        )
        self.mode = self.ns["ForwardMode"]
        self.method = self.ns["AttnForwardMethod"]

    def test_k3_dispatch_opt_in_and_unchanged_modes(self):
        method = self.method

        class Base:
            def dispatch_attn_forward_method(self, batch):
                if batch.forward_mode.is_extend_without_speculative():
                    return method.MHA_NPU
                return method.MLA_NPU

        cls = ast.ClassDef(
            name="K3",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[
                definition(
                    "models/kimi_k3.py",
                    "dispatch_attn_forward_method",
                    scope="KimiK3MLAAttention",
                )
            ],
            decorator_list=[],
        )
        ns = load([cls], {"Base": Base, "envs": envs, "AttnForwardMethod": method})
        obj = ns["K3"]()
        for enabled in (False, True):
            with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(enabled):
                for mode in self.mode:
                    batch = SimpleNamespace(forward_mode=mode, attn_cp_metadata=None)
                    expected = Base.dispatch_attn_forward_method(obj, batch)
                    if enabled and expected == method.MHA_NPU:
                        expected = method.MLA_NPU
                    self.assertEqual(obj.dispatch_attn_forward_method(batch), expected)
                if enabled:
                    with self.assertRaises(NotImplementedError):
                        obj.dispatch_attn_forward_method(
                            SimpleNamespace(
                                forward_mode=self.mode.EXTEND, attn_cp_metadata=object()
                            )
                        )
                obj._kimi_split_gguf_kv_b = True
                self.assertEqual(obj.dispatch_attn_forward_method(batch), method.MLA)
                obj._kimi_split_gguf_kv_b = False

    def test_prepare_reuses_cache_write_but_skips_kv_expansion(self):
        torch.manual_seed(7)
        q = torch.randn(5, 4, 192)
        latent = torch.randn(5, 576)
        pool = SimpleNamespace(set_kv_buffer=Mock())
        m = SimpleNamespace(
            q_lora_rank=None,
            use_dsa=False,
            num_local_heads=4,
            qk_head_dim=192,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            use_deepseek_yarn_rope=False,
            rotary_emb=None,
            q_proj=lambda x: (q.clone(),),
            kv_a_proj_with_mqa=lambda x: (latent.clone(),),
            kv_a_layernorm=lambda x: x * 2,
            w_kc=torch.randn(4, 128, 512),
            kv_b_proj=Mock(return_value=(torch.randn(5, 4 * 256),)),
            _concat_and_cast_mha_k=lambda k, r, b: torch.cat(
                (k, r.expand(-1, 4, -1)), dim=-1
            ),
        )
        ns = load(
            [
                definition(
                    "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
                    "forward_mha_prepare_npu",
                )
            ],
            {
                "torch": torch,
                "get_token_to_kv_pool": lambda: pool,
                "get_attn_tp_context": lambda: SimpleNamespace(
                    fetch_qkv_latent=lambda: torch.cat(
                        (torch.ones(5, 16), latent), dim=-1
                    )
                ),
                "_use_ag_after_qlora": False,
            },
        )
        prepare = ns["forward_mha_prepare_npu"]
        batch = SimpleNamespace(out_cache_loc=torch.arange(5))
        args = (m, torch.arange(5), torch.empty(5, 1), batch, None, None)
        for q_lora_rank in (None, 16):
            m.q_lora_rank = q_lora_rank
            m.q_a_layernorm = lambda x: x * 2
            m.q_b_proj = Mock(return_value=(q.clone(),))
            original = prepare(*args)
            original_write = pool.set_kv_buffer.call_args
            m.kv_b_proj.reset_mock()
            pool.set_kv_buffer.reset_mock()
            result = prepare(*args, absorb_kv=True)
            m.kv_b_proj.assert_not_called()
            pool.set_kv_buffer.assert_called_once()
            if q_lora_rank is not None:
                self.assertEqual(m.q_b_proj.call_count, 2)
                torch.testing.assert_close(
                    m.q_b_proj.call_args.args[0], torch.full((5, 16), 2.0)
                )
            for i in (1, 2, 3):
                torch.testing.assert_close(
                    pool.set_kv_buffer.call_args.args[i], original_write.args[i]
                )
            torch.testing.assert_close(result[0], original[0][..., -64:])
            torch.testing.assert_close(result[1], latent[:, None, 512:])
            torch.testing.assert_close(
                result[2], torch.einsum("thd,hdl->thl", q[..., :128], m.w_kc)
            )
            torch.testing.assert_close(result[3], (latent[:, :512] * 2).unsqueeze(1))

    def test_mla_prepare_does_not_change_verify_or_decode(self):
        class OriginalPath(Exception):
            pass

        prepare = Mock(return_value="prefill")
        ns = load(
            [
                definition(
                    "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
                    "forward_mla_prepare_npu",
                )
            ],
            {
                "torch": torch,
                "envs": envs,
                "forward_mha_prepare_npu": prepare,
                "is_mla_preprocess_enabled": Mock(side_effect=OriginalPath),
            },
        )
        with envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(True):
            for mode in self.mode:
                batch = SimpleNamespace(forward_mode=mode)
                if mode.is_extend_without_speculative():
                    self.assertEqual(
                        ns["forward_mla_prepare_npu"](
                            None, None, None, batch, None, None
                        ),
                        "prefill",
                    )
                    self.assertTrue(prepare.call_args.kwargs["absorb_kv"])
                else:
                    with self.assertRaises(OriginalPath):
                        ns["forward_mla_prepare_npu"](
                            None, None, None, batch, None, None
                        )

    def test_backend_uses_prefill_lengths_and_preserves_verify_dispatch(self):
        ns = load(
            [
                definition(
                    "hardware_backend/npu/attention/ascend_backend.py",
                    "forward_extend",
                    scope="AscendAttnBackend",
                )
            ],
            {
                "torch": torch,
                "envs": envs,
                "RadixAttention": object,
                "ForwardBatch": object,
                "Optional": __import__("typing").Optional,
                "is_mla_preprocess_enabled": lambda: False,
                "is_fia_nz": lambda: True,
            },
        )
        cache = torch.empty(2, 128, 1, 512)
        rope_cache = torch.empty(2, 128, 1, 64)
        pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(cache, rope_cache)), set_kv_buffer=Mock()
        )
        backend = SimpleNamespace(
            use_mla=True,
            is_dllm_model=False,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            token_to_kv_pool=pool,
            page_size=128,
            fia_mask=object(),
            forward_mtp=Mock(return_value="verify"),
            forward_metadata=SimpleNamespace(
                seq_lens_cpu_int=torch.tensor([130, 7]),
                block_tables=torch.tensor([[0, 1], [1, 0]]),
            ),
        )
        q, qr = torch.empty(12, 4, 512), torch.empty(12, 4, 64)
        batch = SimpleNamespace(
            forward_mode=self.mode.EXTEND, extend_seq_lens_cpu=[3, 7]
        )
        layer = SimpleNamespace(layer_id=3, tp_q_head_num=4, scaling=0.1)
        with (
            envs.SGLANG_NPU_USE_FIAS_V2_PREFILL.override(True),
            patch.object(mla_prefill, "fias_v2_mla_prefill", return_value=q) as call,
        ):
            out = ns["forward_extend"](backend, q, None, None, layer, batch, q_rope=qr)
            self.assertEqual(out.shape, (12, 2048))
            self.assertEqual(call.call_args.kwargs["query_lens"], [3, 7])
            self.assertEqual(call.call_args.kwargs["kv_lens"], [130, 7])
            self.assertTrue(call.call_args.kwargs["is_nz"])
            pool.get_kv_buffer.assert_called_once_with(3)
            pool.set_kv_buffer.assert_not_called()
            for mode in (self.mode.TARGET_VERIFY, self.mode.DRAFT_EXTEND_V2):
                batch.forward_mode = mode
                self.assertEqual(
                    ns["forward_extend"](
                        backend, q, None, None, layer, batch, q_rope=qr
                    ),
                    "verify",
                )
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
