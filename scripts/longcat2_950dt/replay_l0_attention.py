"""Offline L0/attention 0 projection comparison for the fixed TP8 Control capture."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import torch
from check_attn0_moe_boundary import read_full_bf16, round_bf16
from check_moe_branches import REFERENCE, emit, fixed_routes
from compare_full_route_tensors import compare_values
from replay_tp5_experts import BLOCK, FP8, checkpoint_map, quant_dequant
from route_lock_tools import require
from safetensors import safe_open

PREFIX = "model.layers.0.self_attn.0."
PARTS = {"q": ("q_a_proj", 1536), "kv": ("kv_a_proj_with_mqa", 576)}
ROW = 844


def digest(x):
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def compare(point, stage, left, right, names=("A3", "950DT"), **extra):
    stats = compare_values(left, right, range(left.size(0)))
    # compare_values uses a3/a5 keys internally, including for within-A5 checks.
    for key in ("first_difference", "max_difference", "first_nonfinite"):
        if (item := stats[key]) is not None:
            item["left"], item["right"] = item.pop("a3"), item.pop("a5")
    stats["nonfinite_left"] = stats.pop("nonfinite_a3")
    stats["nonfinite_right"] = stats.pop("nonfinite_a5")
    emit("COMPARE", point=point, stage=stage, operands=names, **extra, **stats)
    a, b = left[ROW, :20].double(), right[ROW, :20].double()
    emit(
        "ROW844",
        point=point,
        stage=stage,
        operands=names,
        **extra,
        cols=list(range(20)),
        left=a.tolist(),
        right=b.tolist(),
        right_minus_left=(b - a).tolist(),
    )


def weight(model, mapping, name, width, fp8, tp=None, captured=None):
    key = PREFIX + name + ".weight"
    cols = slice(None) if tp is None else slice(tp * 1024, (tp + 1) * 1024)

    def read(k, shape, col_slice):
        with safe_open(model / mapping[k], framework="pt", device="cpu") as f:
            t = f.get_slice(k)
            require(tuple(t.get_shape()) == shape, f"bad shape for {k}")
            return t[:, col_slice]

    raw = read(key, (width, 8192), cols)
    if not fp8:
        require(raw.dtype == torch.bfloat16, f"expected BF16 for {key}")
        return raw
    require(raw.dtype == FP8, f"expected E4M3FN for {key}")
    sc = slice(None) if tp is None else slice(tp * 8, (tp + 1) * 8)
    scales = read(key + "_scale_inv", ((width + 127) // 128, 64), sc)
    require(
        scales.dtype == torch.float32
        and torch.isfinite(scales).all()
        and (scales > 0).all(),
        f"invalid FP8 block scales for {key}",
    )
    if captured is not None:
        cw, cs = captured
        require(
            torch.equal(raw.view(torch.uint8), cw.view(torch.uint8))
            and cs.dtype == scales.dtype
            and torch.equal(scales, cs),
            f"captured weight or scale mismatch: {key}",
        )
    return (
        raw.double()
        * scales.repeat_interleave(BLOCK, 0)[:width]
        .repeat_interleave(BLOCK, 1)
        .double()
    )


def routes(root):
    data = REFERENCE.read_bytes()
    return fixed_routes(root, json.loads(data), hashlib.sha256(data).hexdigest())


def read_input(root, receipt):
    x = read_full_bf16(root, 0, "attn0.input", 0, receipt[0, 0])
    require(torch.isfinite(x).all(), "nonfinite fixed input")
    for tp in range(1, 8):
        require(
            torch.equal(x, read_full_bf16(root, 0, "attn0.input", tp, receipt[0, tp])),
            f"replicated input differs on TP{tp}",
        )
    return x


def o_inputs(root, receipt):
    """Inventory the aligned fixed capture and any old rank-0 attention-tail probe."""
    found = {}
    for tp in range(8):
        for point in ("attn0.o_proj.input", "attn0.o_input"):
            p = root / "full" / f"full_L0_{point}_TP{tp}.pt"
            if not p.exists():
                continue
            obj = torch.load(p, weights_only=True, map_location="cpu")
            require(
                (obj["layer"], obj["point"], obj["tp"], obj["mode"], obj["schema"])
                == (0, point, tp, "EXTEND", 1)
                and obj["identity"] == receipt[0, tp]["identity"]
                and obj["route_mode"] == receipt[0, tp]["mode"]
                and obj["rows"] == list(range(852)),
                f"wrong provenance: {p}",
            )
            x = obj["tensor"]
            require(
                x.shape == (852, 1024) and x.dtype == torch.bfloat16,
                f"bad O input: {p}",
            )
            if tp in found:
                require(torch.equal(x, found[tp]), "conflicting O input")
            found[tp] = x
    return found


def tail_inputs(root, receipt, selected, paths):
    expected = read_full_bf16(root, 0, "attn0.local_output", 0, receipt[0, 0])
    for p in paths:
        d = torch.load(p, weights_only=True, map_location="cpu")
        o, tail = d["o"], d["tail"]
        x = o["x"]
        aligned = (
            tail["layer_id"] == 0
            and x.shape == (852, 1024)
            and x.dtype == torch.bfloat16
            and torch.equal(x, tail["v_output"].reshape(852, 1024))
            and torch.equal(o["output"], expected)
            and torch.equal(tail["output"], expected)
        )
        emit("TAIL", path=str(p), aligned_by_full_tp0_output=aligned)
        if aligned:
            require(
                0 not in selected or torch.equal(x, selected[0]),
                "ambiguous TP0 O input",
            )
            selected[0] = x


def matched_qkv(path, x):
    if path is None:
        return None
    d = torch.load(path, weights_only=True, map_location="cpu")
    matched = d["x"].dtype == x.dtype and torch.equal(x, d["x"])
    emit("QKV_DUMP", path=str(path), same_full_input=matched)
    if not matched:
        return None
    for name, shape, dtype in (
        ("output", (852, 2112), torch.bfloat16),
        ("weight", (2112, 8192), FP8),
        ("weight_scale", (17, 64), torch.float32),
        ("qx", (852, 8192), FP8),
        ("x_scale", (852, 64), torch.float32),
    ):
        require(
            d[name].shape == shape and d[name].dtype == dtype, f"bad captured {name}"
        )
    return d


def check_model(model, fp8):
    cfg = json.loads((model / "config.json").read_text())
    require(
        not cfg.get("attention_bias")
        and cfg["hidden_size"] == 8192
        and cfg["q_lora_rank"] == 1536
        and cfg["kv_lora_rank"] == 512
        and cfg["qk_rope_head_dim"] == 64,
        "unexpected model config",
    )
    if fp8:
        q = cfg["quantization_config"]
        require(
            q["quant_method"] == "fp8"
            and q["fmt"] == "e4m3"
            and q["weight_block_size"] == [128, 128]
            and q["activation_scheme"] == "dynamic"
            and "self_attn" not in cfg.get("disable_quant_module", []),
            "expected native block FP8",
        )
    else:
        require(
            "self_attn" in cfg["disable_quant_module"],
            "expected A3 unquantized attention",
        )


def run(args):
    a = routes(args.a3)
    x = read_input(args.a3, a)
    scope = {
        "schema": 1,
        "shape": list(x.shape),
        "input_sha256": digest(x),
        "route_sha256": a[0, 0]["reference_sha256"],
        "input_hash": a[0, 0]["identity"]["input_hash"],
    }
    green = args.phase == "compare"
    check_model(args.model, green)
    mapping = checkpoint_map(args.model)
    if not green:
        result = {"scope": scope, "projections": {}, "o": {}}
        for part, (name, width) in PARTS.items():
            w = weight(args.model, mapping, name, width, False)
            y = x.double() @ w.double().T
            require(torch.isfinite(y).all(), "nonfinite A3 projection")
            result["projections"][part] = y
            emit(
                "BLUE",
                part=part,
                shape=list(y.shape),
                input_sha256=scope["input_sha256"],
                weight_dtype=str(w.dtype),
                weight_sha256=digest(w),
                reference_sha256=digest(y),
                row844_first20=y[ROW, :20].tolist(),
            )
        o = o_inputs(args.a3, a)
        tail_inputs(args.a3, a, o, args.tail)
        for tp, inp in o.items():
            result["o"][tp] = {
                "x": inp,
                "weight": weight(args.model, mapping, "o_proj", 8192, False, tp),
            }
        emit("O_INVENTORY", platform="A3", available_tp=sorted(o))
        buf = io.BytesIO()
        torch.save(result, buf)
        with args.reference.open("xb") as f:
            f.write(gzip.compress(buf.getvalue(), compresslevel=6, mtime=0))
        return
    b = routes(args.a5)
    require(torch.equal(x, read_input(args.a5, b)), "different A3/950DT attn0.input")
    with gzip.open(args.reference, "rb") as f:
        ref = torch.load(io.BytesIO(f.read()), weights_only=True, map_location="cpu")
    require(
        scope == ref["scope"],
        "A3 CPU reference has different input or route provenance",
    )
    emit("INPUT", **scope, equal_all_ranks=True)
    xq = quant_dequant(x)
    captured = matched_qkv(args.qkv, x)
    if captured is not None:
        captured_x = captured["qx"].double() * captured[
            "x_scale"
        ].double().repeat_interleave(128, 1)
        compare(
            "attn0.input",
            "quantizer_replay",
            captured_x,
            xq,
            ("950DT_captured_dequantized_x", "CPU_dequantized_x"),
        )
    for part, (name, width) in PARTS.items():
        base = ref["projections"][part]
        require(
            base.shape == (852, width) and base.dtype == torch.float64,
            "invalid reference",
        )
        begin = 0 if part == "q" else 1536
        end = begin + width
        params = (
            None
            if captured is None
            else (
                captured["weight"][begin:end],
                captured["weight_scale"][begin // 128 : (end + 127) // 128],
            )
        )
        w = weight(args.model, mapping, name, width, True, captured=params)
        yw = x.double() @ w.T
        yq = xq @ w.T
        compare(part, "checkpoint_fp64", base, yw)
        compare(part, "with_fp8_activation_quant", base, yq)
        compare(
            part,
            "fp8_activation_effect_only",
            yw,
            yq,
            ("950DT_unquantized_x", "950DT_quantized_x"),
        )
        compare(part, "output_bf16", round_bf16(base), round_bf16(yq))
        if captured is not None:
            observed = captured["output"][:, begin:end]
            compare(
                part,
                "A3_cpu_vs_950DT_observed",
                round_bf16(base),
                observed,
                ("A3_cpu_bf16", "950DT_observed"),
            )
            compare(
                part,
                "950DT_gemm_on_captured_operands",
                observed,
                round_bf16(captured_x @ w.T),
                ("950DT_observed", "CPU_bf16"),
            )
    o = o_inputs(args.a5, b)
    tail_inputs(args.a5, b, o, args.tail)
    common = sorted(ref["o"].keys() & o.keys())
    emit("O_INVENTORY", a3_tp=sorted(ref["o"]), a5_tp=sorted(o), paired_tp=common)
    for tp in range(8):
        observed = [
            read_full_bf16(root, 0, "attn0.local_output", tp, r[0, tp])
            for root, r in ((args.a3, a), (args.a5, b))
        ]
        compare("attn0.local_output", "observed", *observed, tp=tp)
        if tp not in common:
            continue
        a3_o = ref["o"][tp]
        w3 = a3_o["weight"].double()
        w5 = weight(args.model, mapping, "o_proj", 8192, True, tp)
        xa, xb = a3_o["x"], o[tp]
        aa, ab = (round_bf16(inp.double() @ w3.T) for inp in (xa, xb))
        ba, bb = (round_bf16(quant_dequant(inp) @ w5.T) for inp in (xa, xb))
        for stage, lhs, rhs, names in (
            ("A3_endpoint", observed[0], aa, ("A3_observed", "A3_replay")),
            ("950DT_endpoint", observed[1], bb, ("950DT_observed", "950DT_replay")),
            ("fixed_A3_operator", aa, ab, ("A3_input", "950DT_input")),
            ("fixed_950DT_operator", ba, bb, ("A3_input", "950DT_input")),
            ("fixed_A3_input", aa, ba, ("A3_operator", "950DT_operator")),
            ("fixed_950DT_input", ab, bb, ("A3_operator", "950DT_operator")),
        ):
            compare("o_proj", stage, lhs, rhs, names, tp=tp)
    emit(
        "COMPLETE",
        projection="checkpoint_cpu_replay",
        observed_local_output_ranks=8,
        o_cross_ranks=common,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["a3", "compare"])
    for arg in ("model", "a3", "reference"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--a5", type=Path)
    p.add_argument(
        "--qkv",
        type=Path,
        help="optional earlier rank0 FP8 fused Q/KV dump for the compare phase",
    )
    p.add_argument(
        "--tail",
        type=Path,
        action="append",
        default=[],
        help="optional same-side existing rank0 longcat_attention_tail dump",
    )
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.phase == "compare" and args.a5 is None:
        p.error("compare requires --a5")
    torch.set_num_threads(args.threads)
    run(args)
