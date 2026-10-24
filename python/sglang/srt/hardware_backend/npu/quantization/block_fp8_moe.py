"""Checkpoint-native 128x128 FP8 MoE on Ascend arch35 (950PR/DT)."""

import torch

from sglang.kernels.ops.gemm import npu_w8a8_block_fp8_grouped_matmul
from sglang.srt.hardware_backend.npu.quantization.moe_utils import (
    build_ascend_quant_info,
    create_ascend_moe_runner,
)
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


class _NPUBlockFp8MoEKernel:
    """One G-B grouped GEMM; the Ascend runner supplies SwiGLU between GEMMs."""

    def apply(
        self,
        quant_info,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        output_dtype,
        weight_prefix,
        group_list_type,
    ):
        return npu_w8a8_block_fp8_grouped_matmul(
            hidden_states,
            getattr(quant_info, f"{weight_prefix}_weight"),
            getattr(quant_info, f"{weight_prefix}_weight_scale"),
            expert_tokens,
            group_list_type,
            output_dtype,
        )


class NPUBlockFp8MoEMethod(Fp8MoEMethod):
    """Reuse FP8 checkpoint loading, with native Ascend G-B execution.

    No post-load transpose, format cast or requantization: the original
    parameter shapes also remain valid for packed expert loading and reloads.
    """

    def __init__(self, quant_config: Fp8Config):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):
            raise ValueError(
                "Block FP8 MoE on Ascend requires --moe-runner-backend auto "
                f"or ascend, got {backend.value!r}"
            )
        super().__init__(quant_config)

    def create_weights(self, layer, *args, with_bias=False, **kwargs):
        if with_bias:
            raise ValueError("Ascend block FP8 MoE does not support expert bias")
        super().create_weights(layer, *args, with_bias=False, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        self.runner = create_ascend_moe_runner(
            layer,
            moe_runner_config,
            w13_kernel=_NPUBlockFp8MoEKernel(),
            w2_kernel=_NPUBlockFp8MoEKernel(),
        )
        self._owns_moe_runner = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep both dispatch modes in BF16. GMM quantizes each routed row to
        # block-128 E4M3 + FP32 scale; the MX routing mode has a different ABI.
        layer.dispatcher.set_quant_config(
            {
                "dispatcher_output_dtype": "bf16",
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "bf16",
            }
        )

    def apply(self, layer, dispatch_output):
        return self.runner.run(
            dispatch_output,
            build_ascend_quant_info(layer),
        )
