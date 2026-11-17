#!/usr/bin/env python3
"""Compare EVERY saved element of two full TP8/two-layer prefill captures.

CPU PyTorch only. No sampling, L2, or numerical pass/fail tolerance.
Inputs are experiment directories containing full/ and routes/ subdirectories.
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from route_lock_tools import load_run, require

POINTS = """attn0.input attn0.indexer attn0.local_output moe.input router.logits
router.expert_ids router.weights moe.dispatch_ids moe.dispatch_weights
moe.experts_output moe.identity moe.reduced moe.branch_output ffn0.input
ffn0.residual_in ffn0.local_output attn1.input attn1.residual attn1.local_output
ffn1.input ffn1.residual_in ffn1.local_output layer.output layer.residual""".split()
STAGES = [(-1, "embedding.oe_ids"), (-1, "embedding.output")]
STAGES += [(l, p) for l in range(2) for p in POINTS]
STAGES += [(-1, "final_norm.output"), (-1, "lm_head.logits")]
BINS = [
    "exact",
    "(0,0.0001)",
    "[0.0001,0.001)",
    "[0.001,0.01)",
    "[0.01,0.1)",
    "[0.1,inf)",
]


def files(root):
    found = {}
    for path in (Path(root) / "full").glob("full_*.pt"):
        match = re.fullmatch(r"full_L(-?\d+)_(.+)_TP(\d+)\.pt", path.name)
        require(match is not None, f"unknown full tensor filename {path}")
        key = int(match[1]), match[2], int(match[3])
        require(key not in found, f"duplicate tensor {key}")
        found[key] = path
    expected = {(l, p, t) for l, p in STAGES for t in range(8)}
    require(
        not expected - found.keys(),
        f"{root}: missing full tensors {sorted(expected - found.keys())}",
    )
    return found


def coordinate(index, shape):
    coord = []
    for width in reversed(shape):
        coord.append(index % width)
        index //= width
    return list(reversed(coord))


def scalar(value):
    value = value.item()
    return value if isinstance(value, int) or math.isfinite(value) else str(value)


def evidence(index, a, b, rows):
    coord = coordinate(index, a.shape)
    av, bv = a.reshape(-1)[index], b.reshape(-1)[index]
    return {
        "coordinate": coord,
        "token_row": rows[coord[0]],
        "a3": scalar(av),
        "a5": scalar(bv),
    }


def compare_values(a, b, rows, chunk=262144):
    require(a.shape == b.shape and a.dtype == b.dtype, "tensor shape/dtype mismatch")
    require(a.numel() > 0, "empty tensor is not valid for this fixed prefill")
    af, bf = a.reshape(-1), b.reshape(-1)
    floating = a.is_floating_point()
    result = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "elements": a.numel(),
        "different": 0,
        "nonfinite_a3": 0,
        "nonfinite_a5": 0,
        "nonfinite_pairs": 0,
        "first_difference": None,
        "first_nonfinite": None,
        "max_abs": None,
        "max_difference": None,
        "bins": dict.fromkeys(BINS, 0) if floating else None,
    }
    for start in range(0, a.numel(), chunk):
        x, y = af[start : start + chunk], bf[start : start + chunk]
        if floating:
            x, y = x.double(), y.double()
        fx, fy = torch.isfinite(x), torch.isfinite(y)
        valid = fx & fy
        bad = ~valid
        result["nonfinite_a3"] += int((~fx).sum())
        result["nonfinite_a5"] += int((~fy).sum())
        result["nonfinite_pairs"] += int(bad.sum())
        different = x != y
        result["different"] += int(different.sum())
        for mask, name in ((different, "first_difference"), (bad, "first_nonfinite")):
            if result[name] is None and mask.any():
                index = start + int(mask.nonzero()[0, 0])
                result[name] = evidence(index, a, b, rows)
        if not floating:
            continue  # Expert/token IDs are categorical, not floating error magnitudes.
        delta = (x - y).abs()
        v = delta[valid]
        result["bins"]["exact"] += int((v == 0).sum())
        result["bins"]["(0,0.0001)"] += int(((v > 0) & (v < 0.0001)).sum())
        for lo, hi, name in (
            (0.0001, 0.001, BINS[2]),
            (0.001, 0.01, BINS[3]),
            (0.01, 0.1, BINS[4]),
        ):
            result["bins"][name] += int(((v >= lo) & (v < hi)).sum())
        result["bins"][BINS[5]] += int((v >= 0.1).sum())
        if valid.any():
            finite_delta = delta.masked_fill(~valid, -1)
            index = int(finite_delta.argmax())
            maximum = finite_delta[index].item()
            if result["max_abs"] is None or maximum > result["max_abs"]:
                result["max_abs"] = maximum
                result["max_difference"] = evidence(start + index, a, b, rows)
    if floating:
        require(
            sum(result["bins"].values()) + result["nonfinite_pairs"]
            == result["elements"],
            "bin accounting failed",
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a3", required=True)
    p.add_argument("--a5", required=True)
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument(
        "--require-fixed",
        action="store_true",
        help="require equal applied IDs AND weights from one reference",
    )
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    ar, br = load_run(Path(args.a3) / "routes"), load_run(Path(args.a5) / "routes")
    require(
        ar[0, 0]["identity"] == br[0, 0]["identity"]
        and ar[0, 0]["meta"] == br[0, 0]["meta"],
        "routing input/config mismatch",
    )
    if args.require_fixed:
        for key in ar:
            x, y = ar[key], br[key]
            require(
                x["mode"] == y["mode"] == "ids_weights",
                "both runs must use ids_weights",
            )
            require(
                x["reference_sha256"] == y["reference_sha256"]
                and x["reference_sha256"],
                "different route references",
            )
            require(
                x["ids"] == y["ids"] and x["weights"] == y["weights"],
                f"forced routes differ: {key}",
            )
    ap, bp = files(args.a3), files(args.a5)
    require(ap.keys() == bp.keys(), "full tensor coverage differs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    with (out / "per_tensor.jsonl").open("w") as f:
        for key in sorted(ap):
            a = torch.load(ap[key], map_location="cpu", weights_only=True)
            b = torch.load(bp[key], map_location="cpu", weights_only=True)
            for side, payload, route in (("a3", a, ar[0, 0]), ("a5", b, br[0, 0])):
                require(
                    payload["schema"] == 1
                    and (payload["layer"], payload["point"], payload["tp"]) == key,
                    f"{side}: header mismatch {key}",
                )
                require(
                    payload["identity"] == route["identity"]
                    and payload["route_mode"] == route["mode"],
                    f"{side}: mixed experiment {key}",
                )
                require(payload["mode"] == "EXTEND", "expected prefill")
                expected_rows = (
                    [851] if key[1] == "lm_head.logits" else list(range(852))
                )
                require(
                    payload["rows"] == expected_rows
                    and payload["tensor"].shape[0] == len(expected_rows),
                    f"{side}: wrong row mapping {key}",
                )
            r = {
                "layer": key[0],
                "point": key[1],
                "tp": key[2],
                **compare_values(a["tensor"], b["tensor"], a["rows"]),
            }
            f.write(json.dumps(r, allow_nan=False) + "\n")
            f.flush()
            group = groups.setdefault(
                key[:2],
                {
                    "layer": key[0],
                    "point": key[1],
                    "ranks": [],
                    "elements": 0,
                    "different": 0,
                    "nonfinite_pairs": 0,
                    "bins": dict.fromkeys(BINS, 0) if r["bins"] is not None else None,
                    "max_abs": None,
                    "max_difference": None,
                    "first_difference": None,
                    "first_nonfinite": None,
                },
            )
            group["ranks"].append(key[2])
            for field in ("elements", "different", "nonfinite_pairs"):
                group[field] += r[field]
            if r["bins"] is not None:
                for name in BINS:
                    group["bins"][name] += r["bins"][name]
            for name in ("first_difference", "first_nonfinite"):
                if group[name] is None and r[name] is not None:
                    group[name] = {"tp": key[2], **r[name]}
            if r["max_abs"] is not None and (
                group["max_abs"] is None or r["max_abs"] > group["max_abs"]
            ):
                group["max_abs"] = r["max_abs"]
                group["max_difference"] = {"tp": key[2], **r["max_difference"]}
            print(
                "CHECKED",
                *key,
                "elements",
                r["elements"],
                "max_abs",
                r["max_abs"],
                flush=True,
            )
    ordered = [groups[s] for s in STAGES] + [
        v for k, v in groups.items() if k not in STAGES
    ]
    for group in ordered:
        require(sorted(group["ranks"]) == list(range(8)), "incomplete rank coverage")
    meta = {
        "a3": str(Path(args.a3).resolve()),
        "a5": str(Path(args.a5).resolve()),
        "fixed_routes_verified": args.require_fixed,
        "tensor_pairs": len(ap),
        "note": "Every stored element compared. Counts include all 8 ranks, including replicated values. No accuracy pass/fail threshold.",
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "stages": ordered}, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        json.dumps(meta),
        "stage | elements | different | max_abs | exact | (0,.0001) | [.0001,.001) | [.001,.01) | [.01,.1) | >=.1 | nonfinite_pairs",
    ]
    for g in ordered:
        counts = (
            " | ".join(str(g["bins"][name]) for name in BINS)
            if g["bins"] is not None
            else "integer IDs: exact comparison only"
        )
        lines.append(
            f"L{g['layer']} {g['point']} | {g['elements']} | {g['different']} | {g['max_abs']} | {counts} | {g['nonfinite_pairs']}"
        )
        lines.append("FIRST " + json.dumps(g["first_difference"]))
        if g["max_difference"] is not None:
            lines.append("MAX " + json.dumps(g["max_difference"]))
        if g["first_nonfinite"] is not None:
            lines.append("NONFINITE " + json.dumps(g["first_nonfinite"]))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("SAVED", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
