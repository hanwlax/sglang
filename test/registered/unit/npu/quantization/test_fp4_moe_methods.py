import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Initialize the quantization registry as in model-config loading.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluMxfp8Quant
from sglang.srt.hardware_backend.npu.quantization.block_fp8_moe import (
    NPUBlockFp8MoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
    NPUW4A8MXFP4FusedMoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    _pair_pack_mxfp_act_scale,
    w4a8_mxfp_gmm,
)
from sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils import (
    prepare_w4a8_mxfp_weight,
    process_npu_fp4_expert_weights,
    reshape_w4a8_mxfp_weight_scale_for_npu,
)
from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUMXFP8OnlineMoEMethod,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.test_utils import CustomTestCase


class TestFP4MethodGate(unittest.TestCase):
    def test_pre_arch35_keeps_fp8_moe_method(self):
        config = Fp8Config(is_fp4_experts=True)
        layer = FusedMoE.__new__(FusedMoE)

        with (
            patch("sglang.srt.layers.quantization.fp8.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35",
                return_value=False,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.experts")

        self.assertIsInstance(method, Fp8MoEMethod)

    def test_arch35_uses_ascend_runner_method(self):
        config = Fp8Config(is_fp4_experts=True)
        config.is_dsv4_fp4_experts = True
        layer = FusedMoE.__new__(FusedMoE)

        with (
            patch("sglang.srt.layers.quantization.fp8.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35",
                return_value=True,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.experts")

        self.assertIsInstance(method, NPUW4A8MXFP4FusedMoEMethod)


class TestNPUSwigluMxfp8Quant(unittest.TestCase):
    def test_translates_runner_conventions_onto_the_ascend_c_op(self):
        # The sgl-kernel-npu op numbers the group-list layouts the other way round
        # from sglang, so count layout arrives as 1 and must leave as 0. The clamp is
        # inert at 0.0, so a limit that missed clamp_value would silently disable it.
        activation = NPUSwigluMxfp8Quant(7.0)
        output = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.empty(2, 1, 2, dtype=torch.float8_e8m0fnu)
        group_list = torch.tensor([1, 1], dtype=torch.int64)
        hidden_states = torch.empty(2, 8)

        with patch.object(
            torch.ops.npu,
            "swiglu_group_quant",
            return_value=(output, scale, None),
            create=True,
        ) as kernel:
            actual_output, actual_scale = activation._apply_activation(
                hidden_states, group_list, group_list_type=1
            )

        self.assertIs(actual_output, output)
        self.assertIs(actual_scale, scale)
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["x"], hidden_states)
        self.assertIs(kwargs["group_index"], group_list)
        self.assertEqual(kwargs["quant_mode"], 2)
        self.assertEqual(kwargs["group_list_type"], 0)
        self.assertEqual(kwargs["clamp_value"], 7.0)

    def test_rejects_a_cumulative_group_list(self):
        # The op has no cusum layout and sums its group list as counts, so a cusum list
        # must fail here rather than derive a row count from the sum of prefix sums.
        activation = NPUSwigluMxfp8Quant(7.0)
        with self.assertRaises(ValueError):
            activation._apply_activation(
                torch.empty(2, 8),
                torch.tensor([1, 2], dtype=torch.int64),
                group_list_type=0,
            )


class TestReshapeMxfp4ScaleForNpu(unittest.TestCase):
    def test_packs_scale_to_gmm_layout(self):
        # [E, N, K/32] -> [E, K/64, N, 2] is the packed-pair layout the GMM reads;
        # getting the transpose axis wrong silently dequantizes with the wrong scale.
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        out = reshape_w4a8_mxfp_weight_scale_for_npu(scale)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 2))
        self.assertTrue(torch.equal(out, scale.view(1, 2, 2, 2).transpose(1, 2)))

    def test_rejects_odd_k_dim(self):
        with self.assertRaises(ValueError):
            reshape_w4a8_mxfp_weight_scale_for_npu(
                torch.zeros(1, 2, 3, dtype=torch.uint8)
            )


class TestPrepareW4A8MxfpWeight(CustomTestCase):
    def test_uses_shared_weight_and_scale_layout(self):
        # A wrong transpose or scale packing makes both ModelSlim W4A8 and
        # DeepSeek-V4 W4A8 read different blocks from the same checkpoint.
        weight = torch.arange(16, dtype=torch.uint8).view(1, 2, 8)
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        formatted_weight = torch.arange(16, dtype=torch.uint8).view(1, 2, 8)

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils.npu_format_cast",
            return_value=formatted_weight,
        ):
            prepared_weight, prepared_scale = prepare_w4a8_mxfp_weight(weight, scale)

        self.assertTrue(torch.equal(prepared_weight, formatted_weight.transpose(1, 2)))
        self.assertTrue(
            torch.equal(prepared_scale, scale.view(1, 2, 2, 2).transpose(1, 2))
        )

    def test_decodes_fp32_scales_without_rebiasing_raw_e8m0(self):
        # The same GMM layout must preserve byte-valued HF scales, decoded
        # powers of two, and ModelSlim / DeepSeek-V4's already encoded bytes.
        raw_bytes = torch.tensor(
            [0, 1, 126, 127, 128, 129, 254, 255], dtype=torch.uint8
        )
        exponents = torch.tensor([-126, -2, -1, 0, 1, 2, 3, 127])
        cases = (
            (raw_bytes, raw_bytes),
            (raw_bytes.float(), raw_bytes),
            (torch.exp2(exponents.float()), (exponents + 127).to(torch.uint8)),
        )
        weight = torch.zeros(2, 3, 64, dtype=torch.uint8)
        for scale, expected in cases:
            with self.subTest(dtype=scale.dtype, values=scale.tolist()):
                scale = scale.repeat(3).view(2, 3, 4)
                expected = expected.repeat(3).view(2, 3, 4)
                with patch(
                    "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils.npu_format_cast",
                    side_effect=lambda weight, **kwargs: weight,
                ):
                    _, prepared_scale = prepare_w4a8_mxfp_weight(weight, scale)

                self.assertEqual(prepared_scale.dtype, torch.uint8)
                self.assertEqual(tuple(prepared_scale.shape), (2, 2, 3, 2))
                for expert in range(2):
                    for n in range(3):
                        for k in range(4):
                            self.assertEqual(
                                prepared_scale[expert, k // 2, n, k % 2],
                                expected[expert, n, k],
                            )

    def test_preserves_packed_int8_bits_and_explicit_npu_format(self):
        weight = torch.tensor([-128, -1, 0, 127], dtype=torch.int8).view(1, 2, 2)
        scale = torch.ones(1, 2, 2, dtype=torch.uint8)

        def format_cast(tensor, acl_format, **kwargs):
            self.assertEqual(acl_format, 29)
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertEqual(kwargs["customize_dtype"], torch.float8_e4m3fn)
            self.assertIsNotNone(kwargs["input_dtype"])
            return tensor

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils.npu_format_cast",
            side_effect=format_cast,
        ):
            prepared_weight, _ = prepare_w4a8_mxfp_weight(weight, scale, npu_format=29)

        self.assertTrue(
            torch.equal(
                prepared_weight,
                torch.tensor([[[128, 0], [255, 127]]], dtype=torch.uint8),
            )
        )

    def test_rejects_missing_fp4_support(self):
        with (
            patch(
                "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils._get_float4_e2m1fn_x2_dtype",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "requires float4 support"),
        ):
            prepare_w4a8_mxfp_weight(
                torch.zeros(1, 2, 32, dtype=torch.uint8),
                torch.ones(1, 2, 2, dtype=torch.uint8),
            )

    def test_processes_both_experts_and_keeps_parameter_metadata(self):
        layer = torch.nn.Module()
        originals = {}
        for prefix, n in (("w13", 3), ("w2", 2)):
            weight = torch.nn.Parameter(
                torch.full((2, n, 64), -1, dtype=torch.int8), requires_grad=False
            )
            scale = torch.nn.Parameter(
                torch.tensor([125.0, 126.0, 127.0, 128.0]).repeat(2, n, 1),
                requires_grad=False,
            )
            scale.weight_loader = object()
            layer.register_parameter(f"{prefix}_weight", weight)
            layer.register_parameter(f"{prefix}_weight_scale_inv", scale)
            originals[prefix] = (weight, scale, scale.weight_loader)

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils.npu_format_cast",
            side_effect=lambda weight, **kwargs: weight,
        ):
            process_npu_fp4_expert_weights(layer)

        for prefix, n in (("w13", 3), ("w2", 2)):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale_inv")
            original_weight, original_scale, loader = originals[prefix]
            self.assertIs(weight, original_weight)
            self.assertIs(scale, original_scale)
            self.assertIs(scale.weight_loader, loader)
            self.assertTrue(
                torch.equal(weight, torch.full((2, 64, n), 255, dtype=torch.uint8))
            )
            self.assertTrue(
                torch.equal(
                    scale,
                    torch.tensor(
                        [[[[125, 126]] * n, [[127, 128]] * n]] * 2, dtype=torch.uint8
                    ),
                )
            )


class TestMxfp4ScaleWeightLoader(unittest.TestCase):
    def test_reinterprets_e8m0_scale_as_raw_uint8(self):
        loaded = []

        def weight_loader(param, loaded_weight, *args, **kwargs):
            loaded.append(loaded_weight.clone())

        layer = torch.nn.Module()
        method = NPUW4A8MXFP4FusedMoEMethod(prefix="test")
        method.create_weights(
            layer,
            num_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        checkpoint_scale = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float8_e8m0fnu)

        layer.w13_weight_scale_inv.weight_loader(
            layer.w13_weight_scale_inv,
            checkpoint_scale,
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
            "w1",
            0,
        )

        self.assertEqual(loaded[0].dtype, torch.uint8)
        self.assertTrue(torch.equal(loaded[0], checkpoint_scale.view(torch.uint8)))


class TestPairPackMxfpActScale(unittest.TestCase):
    def test_packs_as_view(self):
        # The GMM expects a pair-packed *view* of the per-token scale, not a copy;
        # materializing a copy here would break the kernel's aliasing contract.
        flat = torch.arange(8).view(2, 4)
        packed = _pair_pack_mxfp_act_scale(flat)
        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertEqual(packed.data_ptr(), flat.data_ptr())

    def test_rejects_odd_scale_dim(self):
        with self.assertRaises(ValueError):
            _pair_pack_mxfp_act_scale(torch.zeros(2, 3))


class _LowLatencyBuffer:
    """The MXFP8-era Buffer: bool flags, no quant_mode."""

    def __init__(self):
        self.kwargs = None

    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        use_mxfp4=False,
        use_mxfp8=False,
        **kwargs,
    ):
        self.kwargs = {
            "use_fp8": use_fp8,
            "use_mxfp4": use_mxfp4,
            "use_mxfp8": use_mxfp8,
            **kwargs,
        }
        return torch.empty(0), torch.empty(0), object(), object(), object()


class _LegacyLowLatencyBuffer:
    """A DeepEP API version that predates the use_mxfp8 flag."""

    def __init__(self):
        self.use_mxfp4 = None

    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        use_mxfp4=False,
        topk_weights,
        async_finish,
        return_recv_hook,
    ):
        self.use_mxfp4 = use_mxfp4
        return torch.empty(0), torch.empty(0), object(), object(), object()


class _CudaLowLatencyBuffer:
    """CUDA's Buffer API does not accept the NPU-only MXFP flags."""

    def __init__(self):
        self.use_fp8 = None

    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        round_scale=False,
        use_ue8m0=False,
        async_finish=False,
        return_recv_hook=False,
    ):
        self.use_fp8 = use_fp8
        return torch.empty(0), torch.empty(0), object(), object(), object()


class _CudaNormalBuffer:
    """CUDA's normal Buffer API receives an already-quantized input tuple."""

    def __init__(self):
        self.dispatched = False

    def get_dispatch_layout(self, *args, **kwargs):
        return (
            torch.ones(1, dtype=torch.int32),
            None,
            torch.ones(2, dtype=torch.int32),
            torch.ones(1, 1, dtype=torch.bool),
            None,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
        expert_alignment,
        config,
    ):
        self.dispatched = True
        return torch.empty(0), torch.empty(0), torch.empty(0), [], object(), object()


class _FlagNormalBuffer(_CudaNormalBuffer):
    def __init__(self):
        super().__init__()
        self.quantization_kwargs = None

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
        expert_alignment,
        config,
        use_fp8,
        use_mxfp4,
        use_mxfp8,
    ):
        self.dispatched = True
        self.quantization_kwargs = {
            "use_fp8": use_fp8,
            "use_mxfp4": use_mxfp4,
            "use_mxfp8": use_mxfp8,
        }
        return torch.empty(0), torch.empty(0), torch.empty(0), [], object(), object()


class _LegacyNormalBuffer:
    """The pre-bool-flags DeepEP normal-dispatch API used by CI."""

    def __init__(self):
        self.quant_mode = None

    def get_dispatch_layout(self, *args, **kwargs):
        return (
            torch.ones(1, dtype=torch.int32),
            None,
            torch.ones(2, dtype=torch.int32),
            torch.ones(1, 1, dtype=torch.bool),
            None,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        previous_event,
        async_finish,
        allocate_on_comm_stream,
        expert_alignment,
        config,
        quant_mode,
    ):
        self.quant_mode = quant_mode
        return torch.empty(0), torch.empty(0), torch.empty(0), [], object(), object()


class _OpaqueNormalBuffer(_CudaNormalBuffer):
    """The A3 pybind Buffer API whose dispatch signature hides quantization args."""

    def __init__(self):
        super().__init__()
        self.dispatch_kwargs = None

    def dispatch(self, *args, **kwargs):
        self.dispatched = True
        self.dispatch_kwargs = kwargs
        return torch.empty(0), torch.empty(0), torch.empty(0), [], object(), object()


class TestDeepEPLowLatencyMxfp8Dispatch(unittest.TestCase):
    def test_mxfp4_output_dtype_enables_only_mxfp4(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplBase)

        with patch.object(
            deepep,
            "get_deepep_output_dtype",
            return_value=deepep.DispatcherOutputDtype.MXFP4,
        ):
            dispatcher.set_deepep_dispatcher_dtype()

        self.assertFalse(dispatcher.use_fp8)
        self.assertTrue(dispatcher.use_mxfp4)
        self.assertFalse(dispatcher.use_mxfp8)

    @staticmethod
    def _dispatcher(quant_mode, buffer):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {}
        dispatcher.use_fp8 = False
        dispatcher.use_mxfp4 = False
        dispatcher.use_mxfp8 = quant_mode == "mxfp8"
        dispatcher.use_nvfp4 = False
        dispatcher.num_max_dispatch_tokens_per_rank = 2
        dispatcher.num_experts = 2
        dispatcher.return_recv_hook = False
        dispatcher._get_buffer = lambda: buffer
        return dispatcher

    def test_mxfp8_passes_the_mxfp8_flag_without_ue8m0(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher("mxfp8", buffer)

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertFalse(buffer.kwargs["use_fp8"])
        self.assertTrue(buffer.kwargs["use_mxfp8"])
        self.assertNotIn("use_ue8m0", buffer.kwargs)

    def test_mxfp4_passes_the_mxfp4_flag(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher("mxfp4", buffer)
        dispatcher.use_mxfp4 = True

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertFalse(buffer.kwargs["use_fp8"])
        self.assertTrue(buffer.kwargs["use_mxfp4"])
        self.assertFalse(buffer.kwargs["use_mxfp8"])

    def test_bf16_omits_unsupported_mxfp8_flag_for_legacy_buffer(self):
        buffer = _LegacyLowLatencyBuffer()
        dispatcher = self._dispatcher("bf16", buffer)

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertFalse(buffer.use_mxfp4)

    def test_normal_dispatch_passes_quantization_flags(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplNormal)
        dispatcher.num_experts = 2
        dispatcher.async_finish = False
        dispatcher.use_fp8 = False
        dispatcher.use_mxfp4 = False
        dispatcher.use_mxfp8 = True
        buffer = _FlagNormalBuffer()
        dispatcher._get_buffer = lambda: buffer

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(
                deepep.DeepEPConfig,
                "get_instance",
                return_value=SimpleNamespace(normal_dispatch_config=None),
            ),
            patch.object(
                deepep,
                "get_global_expert_distribution_recorder",
                return_value=MagicMock(),
            ),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
                None,
            )

        self.assertTrue(buffer.quantization_kwargs["use_mxfp8"])
        self.assertFalse(buffer.quantization_kwargs["use_fp8"])
        self.assertFalse(buffer.quantization_kwargs["use_mxfp4"])

    def test_normal_dispatch_uses_legacy_quant_mode_when_flags_are_unsupported(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplNormal)
        dispatcher.num_experts = 2
        dispatcher.async_finish = False
        dispatcher.use_fp8 = False
        dispatcher.use_mxfp4 = False
        dispatcher.use_mxfp8 = False
        buffer = _LegacyNormalBuffer()
        dispatcher._get_buffer = lambda: buffer

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(
                deepep.DeepEPConfig,
                "get_instance",
                return_value=SimpleNamespace(normal_dispatch_config=None),
            ),
            patch.object(
                deepep,
                "get_global_expert_distribution_recorder",
                return_value=MagicMock(),
            ),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
                None,
            )

        self.assertEqual(buffer.quant_mode, "bf16")

    def test_normal_dispatch_keeps_a3_legacy_path_for_opaque_signature(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplNormal)
        dispatcher.num_experts = 2
        dispatcher.async_finish = False
        dispatcher.use_fp8 = True
        dispatcher.use_mxfp4 = False
        dispatcher.use_mxfp8 = False
        buffer = _OpaqueNormalBuffer()
        dispatcher._get_buffer = lambda: buffer

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(
                deepep.DeepEPConfig,
                "get_instance",
                return_value=SimpleNamespace(normal_dispatch_config=None),
            ),
            patch.object(
                deepep,
                "get_global_expert_distribution_recorder",
                return_value=MagicMock(),
            ),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
                None,
            )

        self.assertTrue(buffer.dispatched)
        self.assertNotIn("use_fp8", buffer.dispatch_kwargs)
        self.assertNotIn("use_mxfp4", buffer.dispatch_kwargs)
        self.assertNotIn("use_mxfp8", buffer.dispatch_kwargs)
        self.assertNotIn("quant_mode", buffer.dispatch_kwargs)

    def test_cuda_normal_dispatch_omits_npu_quantization_flags(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplNormal)
        dispatcher.num_experts = 2
        dispatcher.async_finish = False
        dispatcher.use_fp8 = True
        dispatcher.use_mxfp4 = True
        dispatcher.use_mxfp8 = True
        buffer = _CudaNormalBuffer()
        dispatcher._get_buffer = lambda: buffer

        with (
            patch.object(deepep, "_is_npu", False),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(
                deepep.DeepEPConfig,
                "get_instance",
                return_value=SimpleNamespace(normal_dispatch_config=None),
            ),
            patch.object(
                deepep,
                "get_global_expert_distribution_recorder",
                return_value=MagicMock(),
            ),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
                None,
            )

        self.assertTrue(buffer.dispatched)

    def test_cuda_low_latency_dispatch_omits_npu_mxfp_flags(self):
        buffer = _CudaLowLatencyBuffer()
        dispatcher = self._dispatcher("mxfp8", buffer)
        dispatcher.use_fp8 = True
        dispatcher.use_mxfp4 = True

        with (
            patch.object(deepep, "_is_npu", False),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertTrue(buffer.use_fp8)


class TestW4A8MxfpGmmInputScale(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(2, 64)
        self.input_scale = torch.ones(2, 1, 2)
        self.weight = torch.empty(2, 64, 32, dtype=torch.uint8)
        self.weight_scale = torch.ones(2, 1, 32, 2, dtype=torch.uint8)
        self.group_list = torch.tensor([1, 1], dtype=torch.int32)

    def _call_gmm(self, input_scale):
        return w4a8_mxfp_gmm(
            input=self.input,
            input_scale=input_scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            group_list_type=1,
            group_list=self.group_list,
            output_dtype=torch.bfloat16,
        )

    def test_supplied_scale_skips_dynamic_quant(self):
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu, "npu_dynamic_mx_quant", create=True
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(self.input_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        call_kwargs = grouped_matmul.call_args.kwargs
        self.assertIs(call_kwargs["per_token_scale"][0], self.input_scale)
        self.assertEqual(call_kwargs["group_list"].dtype, torch.int64)
        self.assertTrue(torch.equal(call_kwargs["group_list"], self.group_list))

    def test_missing_scale_uses_dynamic_quant(self):
        quantized = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(quantized, quantized_scale),
                create=True,
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(None)

        dynamic_quant.assert_called_once()
        self.assertIs(output, expected)
        self.assertIs(
            grouped_matmul.call_args.kwargs["per_token_scale"][0], quantized_scale
        )


class TestRunnerDelegation(CustomTestCase):
    @staticmethod
    def _methods():
        return (
            NPUW4A8MXFP4FusedMoEMethod(prefix="test"),
            NPUBlockFp8MoEMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
                )
            ),
            Fp8MoEMethod(Fp8Config(is_fp4_experts=True)),
        )

    def test_apply_delegates_with_dsv4_scale_names(self):
        layer = SimpleNamespace(
            w13_weight=torch.zeros(1, 4, 4),
            w13_weight_scale_inv=torch.ones(1, 4, 1),
            w2_weight=torch.zeros(1, 4, 2),
            w2_weight_scale_inv=torch.full((1, 4, 1), 2.0),
            # These are different conventions; the checkpoint paths must keep
            # using scale_inv and must not start forwarding optional bias.
            w13_weight_scale=object(),
            w2_weight_scale=object(),
            w13_weight_bias=object(),
        )
        dispatch_output = SimpleNamespace(hidden_states=torch.zeros(1, 4))

        with patch("sglang.srt.layers.quantization.fp8._is_npu", True):
            for method in self._methods():
                with self.subTest(method=type(method).__name__):
                    if isinstance(method, NPUW4A8MXFP4FusedMoEMethod):
                        with self.assertRaisesRegex(
                            RuntimeError, "not been initialized"
                        ):
                            method.apply(layer, dispatch_output)
                    expected = object()
                    method.moe_runner_config = MoeRunnerConfig()
                    method.runner = MagicMock()
                    method.runner.run.return_value = expected

                    output = method.apply(layer, dispatch_output)

                    self.assertIs(output, expected)
                    method.runner.run.assert_called_once()
                    self.assertIs(method.runner.run.call_args.args[0], dispatch_output)
                    quant_info = method.runner.run.call_args.args[1]
                    self.assertIs(quant_info.w13_weight, layer.w13_weight)
                    self.assertIs(quant_info.w2_weight, layer.w2_weight)
                    self.assertIs(
                        quant_info.w13_weight_scale, layer.w13_weight_scale_inv
                    )
                    self.assertIs(quant_info.w2_weight_scale, layer.w2_weight_scale_inv)
                    self.assertIsNone(quant_info.w13_weight_bias)

    def test_create_runner_installs_internal_kernels_before_runner(self):
        # The real constructor reads config.layer.w2_kernel to select the
        # activation. Installing kernels after construction would fail here.
        with patch("sglang.srt.layers.quantization.fp8._is_npu", True):
            methods = (*self._methods(), NPUMXFP8OnlineMoEMethod())
            for method in methods:
                with self.subTest(method=type(method).__name__):
                    layer = SimpleNamespace()
                    config = MoeRunnerConfig()
                    method.create_moe_runner(layer, config)

                    self.assertIs(config.layer, layer)
                    self.assertIs(method.runner.config, config)
                    self.assertIs(method.runner.runner_backend, MoeRunnerBackend.ASCEND)
                    if isinstance(method, NPUW4A8MXFP4FusedMoEMethod):
                        self.assertIs(layer.w13_kernel, method.w13_kernel)
                        self.assertIs(layer.w2_kernel, method.w2_kernel)
                    if isinstance(method, NPUBlockFp8MoEMethod):
                        self.assertTrue(method._owns_moe_runner)
                        self.assertIs(method.moe_runner_config, config)
                        self.assertIsNotNone(method.runner.runner_core.activation)
                    elif type(method) is Fp8MoEMethod:
                        self.assertFalse(method._owns_moe_runner)
                    elif isinstance(method, NPUMXFP8OnlineMoEMethod):
                        self.assertIs(method.moe_runner_config, config)
                        self.assertIsNone(method._aiter_runner)
                        self.assertIsNotNone(layer.w13_kernel.hidden_states_quantizer)
                        self.assertIsNone(layer.w2_kernel.hidden_states_quantizer)

    def test_unsupported_backend_still_rejected_before_runner_creation(self):
        layer = SimpleNamespace()
        config = MoeRunnerConfig()
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            return_value=MoeRunnerBackend.TRITON,
        ):
            method = NPUW4A8MXFP4FusedMoEMethod()
            with self.assertRaisesRegex(ValueError, "requires the Ascend MoE runner"):
                method.create_moe_runner(layer, config)
            self.assertIsNone(method.runner)

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.block_fp8_moe.get_moe_runner_backend",
            return_value=MoeRunnerBackend.TRITON,
        ):
            with self.assertRaisesRegex(ValueError, "Block FP8 MoE on Ascend requires"):
                NPUBlockFp8MoEMethod(Fp8Config())

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.online_moe_methods.get_moe_runner_backend",
            return_value=MoeRunnerBackend.TRITON,
        ):
            method = NPUMXFP8OnlineMoEMethod()
            with self.assertRaisesRegex(ValueError, "MXFP8 MoE on Ascend requires"):
                method.create_moe_runner(layer, config)

        self.assertFalse(hasattr(layer, "w13_kernel"))
        self.assertFalse(hasattr(layer, "w2_kernel"))


class TestProcessWeightsAfterLoadingZeroScale(CustomTestCase):
    @staticmethod
    def _method():
        return NPUW4A8MXFP4FusedMoEMethod(prefix="test")

    def test_raises_when_w13_scales_never_loaded(self):
        # An all-zero scale is the signature of a checkpoint whose scale names
        # never matched; without this guard every routed expert computes silently
        # as zero instead of failing loudly.
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)

    def test_raises_when_w2_scales_never_loaded(self):
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.ones(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)

    def test_keeps_low_latency_dispatch_in_mxfp8(self):
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.arange(64, dtype=torch.uint8).view(1, 2, 32), requires_grad=False
            ),
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[[125, 126], [127, 128]]], dtype=torch.uint8),
                requires_grad=False,
            ),
            w2_weight=torch.nn.Parameter(
                torch.full((1, 64, 32), 255, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.full((1, 64, 2), 127, dtype=torch.uint8), requires_grad=False
            ),
            dispatcher=MagicMock(),
        )
        original_scales = (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)

        def format_cast(weight, acl_format, **kwargs):
            self.assertEqual(acl_format, 29)  # ACL_FORMAT_FRACTAL_NZ
            self.assertEqual(weight.dtype, torch.uint8)
            self.assertIsNotNone(kwargs["input_dtype"])
            return weight

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.mxfp4_weight_utils.npu_format_cast",
            side_effect=format_cast,
        ):
            self._method().process_weights_after_loading(layer)

        self.assertTrue(
            torch.equal(
                layer.w13_weight,
                torch.arange(64, dtype=torch.uint8).view(1, 2, 32).transpose(1, 2),
            )
        )
        self.assertTrue(
            torch.equal(
                layer.w2_weight, torch.full((1, 32, 64), 255, dtype=torch.uint8)
            )
        )
        self.assertTrue(
            torch.equal(
                layer.w13_weight_scale_inv,
                torch.tensor([[[[125, 126], [127, 128]]]], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                layer.w2_weight_scale_inv,
                torch.full((1, 1, 64, 2), 127, dtype=torch.uint8),
            )
        )
        for original, processed in zip(
            original_scales, (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)
        ):
            self.assertIsNot(original, processed)
            self.assertFalse(processed.requires_grad)

        layer.dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            }
        )


if __name__ == "__main__":
    unittest.main()
