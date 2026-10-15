"""NPU linear paths for unquantized Qwen2.5-VL prefill."""

from typing import Optional

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def prefill_gate_up(
    layer: torch.nn.Module,
    x: torch.Tensor,
    forward_mode: Optional[ForwardMode],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Use the execution mode: a multi-request decode also has more than one row.
    # Exact types preserve LoRA wrappers and custom/quantized linear methods.
    if (
        forward_mode in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL)
        and type(layer) is MergedColumnParallelLinear
        and type(layer.quant_method) is UnquantizedLinearMethod
        and layer.bias is None
        and not layer.gather_output
        and x.device.type == "npu"
        and x.ndim == 2
        and x.shape[0] > 0
        and x.dtype == layer.weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and layer.weight.is_contiguous()
    ):
        import torch_npu

        # A3 MatMulV2/V3 routing changes K-block accumulation order. GMM avoids
        # the legacy route that makes Qwen2.5-VL lose the second image description.
        return (
            torch_npu.npu_grouped_matmul(
                [x], [layer.weight.t()], split_item=0, group_type=-1
            )[0],
            None,
        )
    return layer(x)
