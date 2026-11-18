#!/usr/bin/env python3
"""Offline L0 TP8/DP1 boundary check from existing full captures; CPU only.

No model forward, no NPU, no sampling, no L2 and no numerical pass threshold.
HCCL's intermediate reduced tensor is absent: reduction orders below are
explicit hypotheses, NOT a reconstruction of the actual collective schedule.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from compare_full_route_tensors import compare_values
from route_lock_tools import load_run, require

WEIGHT_KEY = "model.layers.0.post_attention_layernorm.0.weight"
SHAPE = (852, 8192)
POINTS = {
    "embedding": (-1, "embedding.output"),
    "local": (0, "attn0.local_output"),
    "moe": (0, "moe.input"),
    "residual": (0, "ffn0.residual_in"),
    "ffn": (0, "ffn0.input"),
}


def weight_hash(weight):
    return hashlib.sha256(
        bytes(weight.contiguous().view(torch.uint8).tolist())
    ).hexdigest()


def round_bf16(value):
    """Nearest-even FP64 -> BF16, avoiding an intermediate FP32 double round."""
    value = value.double()
    center = value.to(torch.bfloat16)
    require(
        bool(torch.isfinite(value).all() and torch.isfinite(center).all()),
        "reference is nonfinite or outside BF16 range",
    )
    best = center.clone()
    distance = (best.double() - value).abs()
    for direction in (-float("inf"), float("inf")):
        neighbor = torch.nextafter(center, torch.full_like(center, direction))
        delta = (neighbor.double() - value).abs()
        even = (neighbor.view(torch.int16) & 1) == 0
        choose = (delta < distance) | ((delta == distance) & even)
        best = torch.where(choose, neighbor, best)
        distance = torch.where(choose, delta, distance)
    return best


def rms(value, weight, eps, dtype):
    x = value.to(dtype)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    y = y * weight.to(dtype)
    return round_bf16(y) if dtype == torch.float64 else y.to(torch.bfloat16)


def sum_ranks(parts, method):
    require(len(parts) == 8, "expected all eight rank contributions")
    if method == "fp64_once":
        total = torch.zeros_like(parts[0], dtype=torch.float64)
        for part in parts:
            total += part.double()
        return round_bf16(total)
    if method in ("bf16_serial_0_to_7", "bf16_serial_7_to_0"):
        order = parts if method.endswith("0_to_7") else parts[::-1]
        total = order[0].clone()
        for part in order[1:]:
            total = round_bf16(total.double() + part.double())
        return total
    require(method == "bf16_pairwise", "unknown sum hypothesis")
    work = parts
    while len(work) > 1:
        work = [
            round_bf16(work[i].double() + work[i + 1].double())
            for i in range(0, len(work), 2)
        ]
    return work[0]


def checkpoint_params(model):
    from safetensors import safe_open

    config = json.loads((model / "config.json").read_text())
    require(config["hidden_size"] == SHAPE[1], "expected hidden_size=8192")
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = [json.loads(index.read_text())["weight_map"][WEIGHT_KEY]]
    else:
        names = [p.name for p in sorted(model.glob("*.safetensors"))]
    found = []
    for name in names:
        with safe_open(str(model / name), framework="pt", device="cpu") as f:
            if WEIGHT_KEY in f.keys():
                found.append((name, f.get_tensor(WEIGHT_KEY)))
    require(len(found) == 1, "expected exactly one post-attention norm weight")
    filename, original = found[0]
    require(tuple(original.shape) == (SHAPE[1],), "wrong norm weight shape")
    require(
        original.dtype in (torch.bfloat16, torch.float16, torch.float32),
        "expected a floating norm weight",
    )
    require(bool(torch.isfinite(original).all()), "nonfinite checkpoint norm weight")
    # Captured runs used --dtype bfloat16. Runtime weights were NOT dumped.
    weight = original.to(torch.bfloat16)
    return {
        "schema": 1,
        "kind": "longcat_l0_post_attention_norm",
        "layer": 0,
        "hidden_size": SHAPE[1],
        "epsilon": config["rms_norm_eps"],
        "weight_key": WEIGHT_KEY,
        "source_model": str(model.resolve()),
        "source_shard": filename,
        "checkpoint_dtype": str(original.dtype),
        "assumed_runtime_dtype": "torch.bfloat16",
        "weight_bf16_sha256": weight_hash(weight),
        "weight": weight.float().tolist(),
        "runtime_weight_verified": False,
    }


def decode_params(data):
    require(
        data["schema"] == 1
        and data["kind"] == "longcat_l0_post_attention_norm"
        and data["layer"] == 0
        and data["hidden_size"] == SHAPE[1]
        and data["weight_key"] == WEIGHT_KEY
        and data["assumed_runtime_dtype"] == "torch.bfloat16",
        "wrong norm parameter metadata",
    )
    eps = data["epsilon"]
    require(
        isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0,
        "invalid RMSNorm epsilon",
    )
    weight = torch.tensor(data["weight"], dtype=torch.float64)
    require(
        tuple(weight.shape) == (SHAPE[1],) and bool(torch.isfinite(weight).all()),
        "invalid norm weight",
    )
    bf16 = weight.to(torch.bfloat16)
    require(
        torch.equal(bf16.double(), weight)
        and weight_hash(bf16) == data["weight_bf16_sha256"],
        "parameter values/hash do not describe the same BF16 weight",
    )
    return bf16, eps


def read_capture(root, point, tp, route):
    layer, name = POINTS[point]
    path = root / "full" / f"full_L{layer}_{name}_TP{tp}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload["schema"] == 1
        and (payload["layer"], payload["point"], payload["tp"]) == (layer, name, tp),
        f"wrong header: {path}",
    )
    require(
        payload["identity"] == route["identity"]
        and payload["route_mode"] == route["mode"]
        and payload["mode"] == "EXTEND"
        and payload["rows"] == list(range(SHAPE[0])),
        f"mixed capture/rows/mode: {path}",
    )
    value = payload["tensor"]
    require(
        tuple(value.shape) == SHAPE and value.dtype == torch.bfloat16,
        f"expected BF16 {SHAPE}: {path}",
    )
    return value


def difference(actual, reference):
    result = compare_values(actual, reference, list(range(actual.shape[0])))
    result["nonfinite_actual"] = result.pop("nonfinite_a3")
    result["nonfinite_reference"] = result.pop("nonfinite_a5")
    for field in ("first_difference", "first_nonfinite", "max_difference"):
        if result[field] is not None:
            item = result[field]
            item["actual"] = item.pop("a3")
            item["reference"] = item.pop("a5")
    return result


def analyze(root, params, out, focus_row=844, focus_col=0):
    out.mkdir(parents=True, exist_ok=False)
    records = []
    lines = []

    def emit(name, actual, reference, **meta):
        result = {"check": name, **meta, **difference(actual, reference)}
        records.append(result)
        brief = {
            k: result[k]
            for k in ("check", "elements", "different", "max_abs", "nonfinite_pairs")
        }
        brief.update(meta)
        print(json.dumps(brief, allow_nan=False), flush=True)
        lines.append(json.dumps(brief, allow_nan=False))
        if result["different"] or result["nonfinite_pairs"]:
            lines.append("BINS " + json.dumps(result["bins"]))
            lines.append("FIRST " + json.dumps(result["first_difference"]))
            lines.append("MAX " + json.dumps(result["max_difference"]))
        with (out / "checks.jsonl").open("a") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
        return result["different"] == 0 and result["nonfinite_pairs"] == 0

    try:
        routes = load_run(root / "routes")
        require(routes[0, 0]["meta"]["tp_size"] == 8, "expected TP8")
        weight, eps = decode_params(params)
        kept = {}
        replica_ok = True
        for point in ("embedding", "residual", "moe", "ffn"):
            base = read_capture(root, point, 0, routes[0, 0])
            require(bool(torch.isfinite(base).all()), f"nonfinite {point} TP0")
            for tp in range(1, 8):
                other = read_capture(root, point, tp, routes[0, tp])
                replica_ok &= emit("rank_agreement", other, base, point=point, tp=tp)
            kept[point] = base
        reuse_ok = emit("moe_ffn_input_reuse", kept["ffn"], kept["moe"])
        require(
            replica_ok and reuse_ok,
            "rank/input-reuse divergence; see checks.jsonl; TP0 cannot represent this capture",
        )
        del kept["ffn"]
        parts = [read_capture(root, "local", tp, routes[0, tp]) for tp in range(8)]
        for tp, part in enumerate(parts):
            require(
                bool(torch.isfinite(part).all()), f"nonfinite local contribution TP{tp}"
            )
        print(
            "ALIGN_OK 40 full tensors; all 852 rows and 8192 columns; TP0-7 checked",
            flush=True,
        )
        # This stored residual is BF16. The fused kernel can normalize the
        # unrounded sum, so these two checks alone are not a fused-norm oracle.
        for dtype in (torch.float32, torch.float64):
            emit(
                "norm_from_stored_residual",
                kept["moe"],
                rms(kept["residual"], weight, eps, dtype),
                arithmetic=str(dtype),
            )
        for method in (
            "fp64_once",
            "bf16_serial_0_to_7",
            "bf16_serial_7_to_0",
            "bf16_pairwise",
        ):
            reduced = sum_ranks(parts, method)
            added = kept["embedding"].double() + reduced.double()
            emit(
                "reduce_plus_residual",
                kept["residual"],
                round_bf16(added),
                reduction=method,
            )
            for dtype in (torch.float32, torch.float64):
                # Float32 addition is explicitly rounded before float32 norm.
                fused_input = kept["embedding"].to(dtype) + reduced.to(dtype)
                emit(
                    "fused_add_norm",
                    kept["moe"],
                    rms(fused_input, weight, eps, dtype),
                    reduction=method,
                    arithmetic=str(dtype),
                )
        focus = {
            "matrix_row": focus_row,
            "token_row": focus_row,
            "col_start": focus_col,
            "columns": list(range(focus_col, min(focus_col + 20, SHAPE[1]))),
        }
        for name, value in kept.items():
            focus[name] = value[focus_row, focus_col : focus_col + 20].float().tolist()
        focus["local_by_tp"] = [
            p[focus_row, focus_col : focus_col + 20].float().tolist() for p in parts
        ]
        (out / "focus.json").write_text(json.dumps(focus, indent=2) + "\n")
        result = {
            "status": "computed",
            "run": str(root.resolve()),
            "scope": "L0 TP8 DP1 EP1 prefill",
            "shape": list(SHAPE),
            "ranks": 8,
            "checks": len(records),
            "parameters": {k: v for k, v in params.items() if k != "weight"},
            "runtime_weight_verified": False,
            "note": "All elements checked, no L2 and no accuracy pass/fail. HCCL sum is not captured. "
            "Reduction orders are hypotheses, not exhaustive. The fused norm may use the "
            "unrounded residual sum. Checkpoint norm parameters are assumed, not runtime-verified.",
        }
        print("COMPUTED", json.dumps(result, allow_nan=False), flush=True)
        lines.insert(0, json.dumps(result, allow_nan=False))
        (out / "summary.json").write_text(
            json.dumps({**result, "results": records}, indent=2) + "\n"
        )
    except Exception as exc:
        lines.append("ERROR " + str(exc))
        (out / "error.json").write_text(
            json.dumps({"status": "error", "error": str(exc)}) + "\n"
        )
        raise
    finally:
        (out / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        required=True,
        help="one capture directory containing full/ and routes/",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=Path,
        help="local checkpoint supplying post-attention norm weight/eps",
    )
    source.add_argument(
        "--params",
        type=Path,
        help="small parameter JSON exported from that platform's checkpoint",
    )
    p.add_argument("--export-params", type=Path)
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--focus-row",
        type=int,
        default=844,
        help="extra numeric window only; does not restrict full comparison",
    )
    p.add_argument("--focus-col", type=int, default=0)
    args = p.parse_args()
    require(
        args.threads > 0
        and 0 <= args.focus_row < SHAPE[0]
        and 0 <= args.focus_col < SHAPE[1],
        "invalid thread count or focus coordinates",
    )
    torch.set_num_threads(args.threads)
    params = (
        checkpoint_params(args.model)
        if args.model
        else json.loads(args.params.read_text())
    )
    decode_params(params)
    if args.export_params:
        with args.export_params.open("x") as f:
            json.dump(params, f, indent=2, allow_nan=False)
            f.write("\n")
    analyze(args.run, params, args.out, args.focus_row, args.focus_col)


if __name__ == "__main__":
    main()
