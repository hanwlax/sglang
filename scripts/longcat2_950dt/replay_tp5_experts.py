#!/usr/bin/env python3
"""Replay L1 TP5 row 0, col 7203 from two fixed-route captures on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from check_attn0_moe_boundary import SHAPE, read_full_bf16, round_bf16
from check_moe_branches import COL, REFERENCE, ROW, emit, fixed_routes
from route_lock_tools import require
from safetensors import safe_open

LAYER, TP, TP_SIZE, BLOCK = 1, 5, 8, 128
EXPERTS = (494, 465, 82, 300)
FP8 = torch.float8_e4m3fn


def checkpoint_map(model):
    index = model / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    mapping = {}
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for key in file.keys():  # noqa: SIM118 - safe_open exposes keys()
                require(key not in mapping, f"duplicate checkpoint tensor: {key}")
                mapping[key] = shard.name
    return mapping


def read_slice(model, mapping, name, shape, rows, cols, dtype):
    require(name in mapping, f"checkpoint missing {name}")
    with safe_open(model / mapping[name], framework="pt", device="cpu") as file:
        tensor = file.get_slice(name)
        require(tuple(tensor.get_shape()) == shape, f"wrong checkpoint shape: {name}")
        value = tensor[rows, cols]
    require(value.dtype == dtype, f"unexpected {name} dtype: {value.dtype}")
    require(torch.isfinite(value.float()).all().item(), f"nonfinite {name}")
    return value.double()


def projection(model, mapping, expert, proj, shape, rows, cols):
    prefix = f"model.layers.{LAYER}.mlp.experts.{expert}.{proj}_proj.weight"
    w = read_slice(model, mapping, prefix, shape, rows, cols, FP8)
    # The down projection reads one output row, which may start inside a block.
    sr = slice(rows.start // BLOCK, (rows.stop + BLOCK - 1) // BLOCK)
    sc = slice(cols.start // BLOCK, (cols.stop + BLOCK - 1) // BLOCK)
    scales = read_slice(
        model,
        mapping,
        prefix + "_scale_inv",
        tuple((n + BLOCK - 1) // BLOCK for n in shape),
        sr,
        sc,
        torch.float32,
    )
    row_index = torch.arange(rows.start, rows.stop) // BLOCK - sr.start
    col_index = torch.arange(cols.start, cols.stop) // BLOCK - sc.start
    return w * scales[row_index[:, None], col_index[None, :]]


def quant_dequant(x):
    # Ascend dynamic_block_quant defaults, E4M3FN with 1x128 activation blocks.
    blocks = x.float().reshape(x.shape[0], -1, BLOCK)
    limit = torch.finfo(FP8).max
    scale = blocks.abs().amax(dim=-1, keepdim=True) / limit
    divisor = torch.where(scale == 0, 1.0, scale)
    payload = (blocks / divisor).clamp(-limit, limit).to(FP8)
    return (payload.double() * scale.double()).reshape_as(x)


def capture(root, reference, digest):
    route = fixed_routes(root, reference, digest)[LAYER, TP]
    x = read_full_bf16(root, LAYER, "moe.input", TP, route)[ROW].clone()
    actual = read_full_bf16(root, LAYER, "moe.experts_output", TP, route)[ROW, COL]
    require(
        bool(torch.isfinite(x).all() and torch.isfinite(actual)), "nonfinite capture"
    )
    ordinary = [
        (i, w)
        for i, w in zip(route["ids"][ROW], route["weights"][ROW])
        if i < route["meta"]["num_experts"]
    ]
    require(
        tuple(i for i, _ in ordinary) == EXPERTS, f"unexpected row 0 route: {ordinary}"
    )
    return x, actual.item(), ordinary


def replay(args):
    require(args.a5.resolve() != args.a3.resolve(), "supply two distinct capture roots")
    config = json.loads((args.model / "config.json").read_text())
    quant = config["quantization_config"]
    require(
        quant["quant_method"] == "fp8"
        and quant["weight_block_size"] == [BLOCK, BLOCK]
        and quant["activation_scheme"] == "dynamic"
        and quant["fmt"] == "e4m3",
        "expected native dynamic block FP8 checkpoint",
    )
    hidden = config["hidden_size"]
    inner = config.get("expert_ffn_hidden_size", config.get("moe_intermediate_size"))
    require(
        hidden == SHAPE[1] and inner and inner % (TP_SIZE * BLOCK) == 0,
        "unexpected expert dimensions for block-aligned TP8",
    )
    width = inner // TP_SIZE
    shard = slice(TP * width, (TP + 1) * width)
    mapping = checkpoint_map(args.model)
    raw = REFERENCE.read_bytes()
    reference, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    a5, actual5, ordinary = capture(args.a5, reference, digest)
    a3, actual3, _ = capture(args.a3, reference, digest)
    x = quant_dequant(torch.stack([a5, a3]))
    emit(
        "REPLAY",
        layer=LAYER,
        tp=TP,
        row=ROW,
        col=COL,
        intermediate_slice=[shard.start, shard.stop],
        model=str(args.model.resolve()),
        route_sha256=digest,
        input_elements=hidden,
        input_different=int((a5 != a3).sum()),
        observed_950dt=actual5,
        observed_a3_int8=actual3,
    )
    terms = []
    for expert, weight in ordinary:
        w_gate = projection(
            args.model,
            mapping,
            expert,
            "gate",
            (inner, hidden),
            shard,
            slice(0, hidden),
        )
        w_up = projection(
            args.model, mapping, expert, "up", (inner, hidden), shard, slice(0, hidden)
        )
        w_down = projection(
            args.model,
            mapping,
            expert,
            "down",
            (hidden, inner),
            slice(COL, COL + 1),
            shard,
        )
        gate = round_bf16(x @ w_gate.T).float()
        up = round_bf16(x @ w_up.T).float()
        activation = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = round_bf16(quant_dequant(activation) @ w_down.T).flatten().float()
        # Standard no-A2A routing casts top-k weights to the hidden dtype.
        routing_weight = torch.tensor(weight, dtype=torch.bfloat16).item()
        weighted = down * routing_weight
        terms.append(weighted)
        emit(
            "EXPERT",
            expert=expert,
            route_weight_fp32=weight,
            route_weight_bf16=routing_weight,
            down_950dt_input=down[0].item(),
            down_a3_input=down[1].item(),
            weighted_950dt_input=weighted[0].item(),
            weighted_a3_input=weighted[1].item(),
            delta_950dt_minus_a3_input=(weighted[0] - weighted[1]).item(),
        )
    accum = torch.stack(terms).sum(dim=0, dtype=torch.float32)
    y5, y3 = accum.to(torch.bfloat16).tolist()
    emit(
        "TOTAL",
        sum_fp32_950dt_input=accum[0].item(),
        sum_fp32_a3_input=accum[1].item(),
        replay_bf16_950dt_input=y5,
        observed_950dt=actual5,
        replay_minus_observed=y5 - actual5,
        exact_950dt=y5 == actual5,
        replay_bf16_a3_input=y3,
        input_effect_950dt_minus_a3=y5 - y3,
        observed_platform_delta=actual5 - actual3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("a5", "a3", "model"):
        parser.add_argument("--" + opt, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.threads > 0, "expected positive thread count")
    torch.set_num_threads(args.threads)
    replay(args)
