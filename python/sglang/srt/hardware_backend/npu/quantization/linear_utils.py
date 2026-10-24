"""Shared weight preparation and bias handling for NPU linear methods."""

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def create_unquantized_linear_weight(
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    from sglang.srt.layers.parameter import ModelWeightParameter

    output_size_per_partition = sum(output_partition_sizes)
    weight_loader = extra_weight_attrs.get("weight_loader")

    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    layer.orig_dtype = params_dtype

    # Load weights in original dtype; quantise later in process_weights_after_loading
    weight = ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=params_dtype,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight", weight)


def cache_linear_bias_fp32(layer: torch.nn.Module):
    # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
    if getattr(layer, "bias", None) is not None and layer.bias.dtype != torch.float32:
        layer.bias_fp32 = Parameter(
            layer.bias.data.to(torch.float32), requires_grad=False
        )
    else:
        layer.bias_fp32 = None


def get_linear_bias_fp32(layer: torch.nn.Module, bias):
    if bias is None:
        quant_bias = None
    elif (
        bias is getattr(layer, "bias", None)
        and getattr(layer, "bias_fp32", None) is not None
    ):
        quant_bias = layer.bias_fp32
    else:
        quant_bias = bias.to(torch.float32)
    return quant_bias


def prepare_w8a8_int8_linear_weights(layer: torch.nn.Module):
    layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
    layer.weight.data = npu_format_cast(layer.weight.data)

    layer.weight_scale.data = layer.weight_scale.data.flatten()
    # Compressed-tensors format doesn't have this field
    if hasattr(layer, "weight_offset"):
        layer.weight_offset.data = layer.weight_offset.data.flatten()
