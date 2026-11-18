#!/usr/bin/env python3
"""CPU elementwise checks of the two fixed-route TP8 MoE branches."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from check_attn0_moe_boundary import SHAPE, difference, read_full_bf16, sum_ranks
from route_lock_tools import load_run, require

REFERENCE = Path(__file__).parent / "green_route_20261117/reference/A3_reference.json"
METHODS = ("fp64_once", "bf16_serial_0_to_7", "bf16_serial_7_to_0", "bf16_pairwise")
ROW, COL = 0, 7203
REPLICATED = ("moe.input", "moe.identity", "moe.reduced", "moe.branch_output")


def emit(label, **data):
    print(label, json.dumps(data, allow_nan=False), flush=True)


def fixed_routes(root, reference, digest):
    routes = load_run(root / "routes")
    for (layer, tp), receipt in routes.items():
        expected = reference["layers"][str(layer)]
        require(
            receipt["mode"] == "ids_weights"
            and receipt["reference_sha256"] == digest
            and receipt["identity"] == reference["identity"]
            and receipt["meta"] == reference["meta"]
            and receipt["ids"] == expected["ids"]
            and receipt["weights"] == expected["weights"],
            f"{root}: route reference mismatch at L{layer} TP{tp}",
        )
    return routes


def identity_scales(route):
    ids = torch.tensor(route["ids"])
    weights = torch.tensor(route["weights"], dtype=torch.float32)
    mask = ids >= route["meta"]["num_experts"]
    # Weights in the receipt are already scaled, before identity-slot masking.
    return weights.masked_fill(~mask, 0).sum(dim=1, keepdim=True)


def check_group(platform, layer, ranks, tensors, scales, cpu_sums):
    def check(name, point, reference, **details):
        emit(
            "CHECK",
            platform=platform,
            layer=layer,
            ranks=ranks,
            check=name,
            point=point,
            **details,
            **difference(tensors[point], reference),
        )

    identity = (tensors["moe.input"].float() * scales).to(torch.bfloat16)
    check("identity", "moe.identity", identity)
    for method, total in cpu_sums.items():
        check("tp_sum", "moe.reduced", total, reduction=method)
    added = tensors["moe.reduced"] + tensors["moe.identity"]
    check("branch_add", "moe.branch_output", added)


def inspect_layer(root, platform, layer, routes):
    parts = []
    groups = []
    focus = []

    def read(point, tp):
        value = read_full_bf16(root, layer, point, tp, routes[layer, tp])
        if not torch.isfinite(value).all():
            bad = ~torch.isfinite(value)
            raise ValueError(
                f"{platform} L{layer} TP{tp} {point}: {int(bad.sum())} nonfinite; "
                f"first at {bad.nonzero()[0].tolist()}"
            )
        return value

    scales = identity_scales(routes[layer, 0])
    for tp in range(8):
        part = read("moe.experts_output", tp)
        parts.append(part)
        values = {point: read(point, tp) for point in REPLICATED}
        focus.append(
            {
                "moe.experts_output": part[ROW, COL].item(),
                **{name: value[ROW, COL].item() for name, value in values.items()},
            }
        )
        # Only share a report when every operand and output is elementwise equal.
        for tensors, ranks in groups:
            if all(torch.equal(tensors[p], values[p]) for p in REPLICATED):
                ranks.append(tp)
                break
        else:
            groups.append((values, [tp]))

    cpu_sums = {}
    for method in METHODS:
        # Bound the temporary FP64/nextafter buffers during BF16 rounding.
        cpu_sums[method] = torch.cat(
            [
                sum_ranks([p[start : start + 32] for p in parts], method)
                for start in range(0, SHAPE[0], 32)
            ]
        )

    for tensors, ranks in groups:
        check_group(platform, layer, ranks, tensors, scales, cpu_sums)
    return focus, {method: total[ROW, COL].item() for method, total in cpu_sums.items()}


def print_focus(layer, route, focus, cpu_sums):
    def compare(label, values, **details):
        if "950DT" in values:
            values["delta_950dt_minus_a3"] = values["950DT"] - values["A3"]
        emit(label, layer=layer, row=ROW, col=COL, **details, **values)

    ids = route["ids"][ROW]
    weights = route["weights"][ROW]
    emit(
        "FOCUS_ROUTE",
        layer=layer,
        row=ROW,
        ids=ids,
        weights=weights,
        identity_weight_sum_fp32=identity_scales(route)[ROW, 0].item(),
    )
    for tp in range(8):
        compare(
            "FOCUS_TP",
            {k: v[tp]["moe.experts_output"] for k, v in focus.items()},
            tp=tp,
        )
    compare(
        "FOCUS_SUM",
        {
            k: math.fsum(row["moe.experts_output"] for row in v)
            for k, v in focus.items()
        },
        reduction="unrounded_fp64",
    )
    for method in METHODS:
        compare(
            "FOCUS_SUM", {k: v[method] for k, v in cpu_sums.items()}, reduction=method
        )
    for point in REPLICATED:
        # Print once for matching scalar replicas, but retain discrepancies by rank.
        groups = {}
        for tp in range(8):
            key = tuple(v[tp][point] for v in focus.values())
            groups.setdefault(key, []).append(tp)
        for key, ranks in groups.items():
            compare("FOCUS_VALUE", dict(zip(focus, key)), point=point, ranks=ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3", type=Path, required=True, help="A3_fixed run root")
    parser.add_argument(
        "--a5", type=Path, help="950DT fixed run root, for a paired report"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.threads > 0, "expected positive thread count")
    torch.set_num_threads(args.threads)
    roots = {"A3": args.a3}
    if args.a5 is not None:
        require(
            args.a3.resolve() != args.a5.resolve(), "expected two distinct run roots"
        )
        roots["950DT"] = args.a5
    raw = REFERENCE.read_bytes()
    ref = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    routes = {k: fixed_routes(root, ref, digest) for k, root in roots.items()}
    emit(
        "ALIGNED",
        runs={k: str(v.resolve()) for k, v in roots.items()},
        mode="ids_weights",
        reference_sha256=digest,
        shape=SHAPE,
        layers=[0, 1],
        tp=8,
    )
    for layer in (1, 0):
        focus, sums = {}, {}
        for platform, root in roots.items():
            focus[platform], sums[platform] = inspect_layer(
                root, platform, layer, routes[platform]
            )
        print_focus(layer, routes["A3"][layer, 0], focus, sums)
    emit("COMPLETE", platforms=list(roots), layers=[1, 0])


if __name__ == "__main__":
    main()
