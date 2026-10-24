"""Shared runner and quantization-payload assembly for Ascend MoE methods."""


def create_ascend_moe_runner(
    layer,
    moe_runner_config,
    *,
    w13_kernel,
    w2_kernel,
):
    """Install both kernels before the runner selects its activation."""
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    layer.w13_kernel = w13_kernel
    layer.w2_kernel = w2_kernel
    moe_runner_config.layer = layer
    return MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)


def build_ascend_quant_info(layer):
    """Build the Ascend payload for checkpoints using weight_scale_inv."""
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

    return AscendQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale_inv,
        w2_weight_scale=layer.w2_weight_scale_inv,
    )
