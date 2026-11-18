#!/usr/bin/env python3
"""Cumulative CPU ablation of expert 494 on the same A3 input, before route weighting."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from check_attn0_moe_boundary import SHAPE, read_full_bf16, round_bf16
from check_moe_branches import REFERENCE, emit, fixed_routes
from replay_tp5_experts import (
    COL,
    LAYER,
    ROW,
    TP,
    checkpoint_map,
    projection,
    quant_dequant,
)
from route_lock_tools import require
from safetensors import safe_open

EXPERT = 494
SHARD = slice(1280, 1536)
FULL_H = slice(0, 8192)
MODEL_SHAPES = {"gate": (2048, 8192), "up": (2048, 8192), "down": (8192, 2048)}
# Prior CPU replays on the A3 input. Only the combined four-expert output was dumped.
PREVIOUS_REPLAY = {"int8": -0.828125, "fp8": -0.765625}


def int8_parameters(model, mapping, proj, shape, rows, cols):
    prefix = f"model.layers.{LAYER}.mlp.experts.{EXPERT}.{proj}_proj.weight"
    with safe_open(model / mapping[prefix], framework="pt", device="cpu") as f:
        t = f.get_slice(prefix)
        require(tuple(t.get_shape()) == shape, f"wrong weight shape: {prefix}")
        weight = t[rows, cols]
    key = prefix + "_scale"
    with safe_open(model / mapping[key], framework="pt", device="cpu") as f:
        t = f.get_slice(key)
        require(
            tuple(t.get_shape()) in ((shape[0],), (shape[0], 1)),
            f"wrong scale shape: {key}",
        )
        scale = t[rows].reshape(-1)
    require(
        weight.dtype == torch.int8
        and scale.is_floating_point()
        and torch.isfinite(scale).all(),
        f"invalid W8A8 weights: {prefix}",
    )
    return weight.double(), scale.double()


def int8_quant(x):
    x = x.float()
    scale = x.abs().amax() / 127
    payload = torch.round(x / torch.where(scale == 0, 1.0, scale)).clamp(-127, 127)
    return payload.double(), scale


def int8_rescale(params, quant):
    # Match the earlier A3 replay: exact integer dot, FP32 channel and token scales.
    w, scale = params
    payload, token_scale = quant
    return ((w @ payload).float() * scale.bfloat16().float() * token_scale).double()


def evaluate(fmt, model, row):
    mapping = checkpoint_map(model)
    weights, int8 = {}, {}
    for proj, shape in MODEL_SHAPES.items():
        rows, cols = (slice(COL, COL + 1), SHARD) if proj == "down" else (SHARD, FULL_H)
        if fmt == "fp8":
            weights[proj] = projection(model, mapping, EXPERT, proj, shape, rows, cols)
        else:
            w, scale = int8_parameters(model, mapping, proj, shape, rows, cols)
            int8[proj] = w, scale
            weights[proj] = w * scale[:, None]

    stages = {}

    def record(name, act, down=None):
        if down is None:
            down = weights["down"] @ act
        require(
            torch.isfinite(act).all() and torch.isfinite(down).all(),
            f"nonfinite {name}",
        )
        stages[name] = {"down": down.item(), "down_input": act.tolist()}

    def project_gate_up(x):
        return weights["gate"] @ x, weights["up"] @ x

    def activation(gate, up):
        return F.silu(gate) * up

    x = row.double()
    record("checkpoint_fp64", activation(*project_gate_up(x)))
    if fmt == "int8":
        weights = {p: w * s.bfloat16().double()[:, None] for p, (w, s) in int8.items()}
    record("runtime_weight_scales", activation(*project_gate_up(x)))

    if fmt == "fp8":
        x = quant_dequant(x[None])[0]
    else:
        q1 = int8_quant(x)
        x = q1[0] * q1[1].double()
    gate, up = project_gate_up(x)
    record("quantize_input_1", activation(gate, up))
    if fmt == "int8":
        gate, up = (int8_rescale(int8[p], q1) for p in ("gate", "up"))
    record("gemm1_rescale_fp32_int8", activation(gate, up))
    gate, up = round_bf16(gate).double(), round_bf16(up).double()
    record("gate_up_bf16", activation(gate, up))
    act = activation(gate.float(), up.float()).double()
    record("swiglu_fp32", act)
    if fmt == "fp8":
        # Native block FP8 materializes BF16 SwiGLU. A3 fuses SwiGLU and quantization.
        act = act.bfloat16().double()
    record("swiglu_output_bf16_fp8", act)
    if fmt == "fp8":
        act = quant_dequant(act[None])[0]
    else:
        q2 = int8_quant(act)
        act = q2[0] * q2[1].double()
    down = weights["down"] @ act
    record("quantize_input_2", act, down)
    if fmt == "int8":
        down = int8_rescale(int8["down"], q2)
    record("gemm2_rescale_fp32_int8", act, down)
    record("down_bf16", act, round_bf16(down))
    return stages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("fp8", "int8"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--a3", type=Path, required=True, help="fixed A3 dump on either platform"
    )
    p.add_argument(
        "--report",
        type=Path,
        required=True,
        help="new local JSON including all 256 down inputs per stage",
    )
    p.add_argument(
        "--a3-baseline", type=Path, help="compare FP8 stage scalars to the blue A3 run"
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    raw = REFERENCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    route = fixed_routes(args.a3, json.loads(raw), digest)[LAYER, TP]
    require(EXPERT in route["ids"][ROW], "expected fixed route to expert 494")
    row = read_full_bf16(args.a3, LAYER, "moe.input", TP, route)[ROW].contiguous()
    require(
        row.numel() == SHAPE[1] and torch.isfinite(row).all(), "invalid A3 input row"
    )
    scope = {
        "schema": 1,
        "layer": LAYER,
        "tp": TP,
        "expert": EXPERT,
        "row": ROW,
        "col": COL,
        "shard": [SHARD.start, SHARD.stop],
        "route_sha256": digest,
        "input_bf16_sha256": hashlib.sha256(
            bytes(row.view(torch.uint8).tolist())
        ).hexdigest(),
    }
    stages = evaluate(args.format, args.model, row)
    baseline = None
    if args.a3_baseline:
        base = json.loads(args.a3_baseline.read_text())
        require(
            args.format == "fp8"
            and base["format"] == "int8"
            and scope == base["scope"],
            "A3 baseline input, scope or format mismatch",
        )
        baseline = base["down_by_stage"]
        require(baseline.keys() == stages.keys(), "baseline stage mismatch")
    emit("INPUT", format=args.format, **scope)
    previous_gap = None
    for name, stage in stages.items():
        fields = {"stage": name, "down": stage["down"]}
        if baseline is not None:
            gap = stage["down"] - baseline[name]
            fields.update(
                a3=baseline[name],
                fp8_minus_int8=gap,
                gap_change=None if previous_gap is None else gap - previous_gap,
            )
            previous_gap = gap
        emit("STAGE", **fields)
    value = stages["down_bf16"]["down"]
    report = {
        "format": args.format,
        "scope": scope,
        "model": str(args.model.resolve()),
        "stages": stages,
    }
    with args.report.open("x") as out:
        json.dump(report, out, allow_nan=False)
        out.write("\n")
    emit(
        "ENDPOINT",
        down=value,
        previous_cpu_replay=PREVIOUS_REPLAY[args.format],
        matches_previous_cpu_replay=value == PREVIOUS_REPLAY[args.format],
        report=str(args.report),
    )


if __name__ == "__main__":
    main()
