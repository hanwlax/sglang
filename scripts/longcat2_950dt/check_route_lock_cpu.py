#!/usr/bin/env python3
"""CPU-only diagnostic self-check; needs --control with the fixed prompt IDs.

Uses synthetic router/activation tensors. Does not launch a server or use NPUs.
"""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control", required=True)
args = parser.parse_args()


def module(name):
    path = (
        root.parents[1] / "python/sglang/srt/debug_utils/longcat_route_lock.py"
        if name == "longcat_route_lock"
        else root / (name + ".py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


lock = module("longcat_route_lock")
tools = module("route_lock_tools")
saved = json.loads(Path(args.control).read_text())
ids = saved["response"]["choices"][0]["prompt_token_ids"]
assert len(ids) == 852
cfg = NS(
    renormalize=False,
    use_grouped_topk=False,
    correction_bias=torch.zeros(896),
    apply_routed_scaling_factor_on_output=False,
    routed_scaling_factor=None,
    scoring_func="softmax",
)
moe = NS(
    topk=NS(topk_config=cfg),
    tp_size=8,
    layer_id=0,
    num_experts=768,
    zero_expert_num=128,
    zero_expert_type="identity",
    top_k=12,
    routed_scaling_factor=256.0,
)


def trace(rank):
    return NS(
        mode="EXTEND",
        num_tokens=852,
        rank=rank,
        input_ids=torch.tensor(ids),
        positions=torch.arange(852),
    )


def natural(logits):
    top_ids = logits.softmax(-1).topk(12, sorted=False).indices
    return lock.weights_for_ids(logits, top_ids, "softmax", 256.0), top_ids


torch.manual_seed(41)
logits = torch.randn(852, 896)
logits[:, 838] += 4
weights, indices = natural(logits)
os.environ.pop("LC_ROUTE_MODE", None)
x, y = lock.apply_route_lock(moe, None, logits, weights, indices)
assert x is weights and y is indices
with tempfile.TemporaryDirectory(prefix="lc_route_cpu_") as td:
    td = Path(td)
    os.environ["LC_ROUTE_MODE"] = "capture"
    os.environ["LC_ROUTE_OUT"] = str(td / "capture")
    for layer in range(2):
        moe.layer_id = layer
        for rank in range(8):
            w, i = lock.apply_route_lock(
                moe, trace(rank), logits, weights.clone(), indices.clone()
            )
            assert torch.equal(w, weights) and torch.equal(i, indices)
    tools.pack(NS(routes=str(td / "capture"), out=str(td / "reference.json")))
    ref = json.loads((td / "reference.json").read_text())
    assert any(838 in row for row in ref["layers"]["0"]["ids"])
    os.environ["LC_ROUTE_REFERENCE"] = str(td / "reference.json")
    logits2 = -logits
    weights2, indices2 = natural(logits2)
    assert not torch.equal(indices, indices2)
    for mode in ["ids_weights", "ids"]:
        os.environ["LC_ROUTE_MODE"] = mode
        os.environ["LC_ROUTE_OUT"] = str(td / mode)
        for layer in range(2):
            moe.layer_id = layer
            for rank in range(8):
                w, i = lock.apply_route_lock(
                    moe, trace(rank), logits2, weights2, indices2
                )
                assert torch.equal(i, indices)
                expected = (
                    weights
                    if mode == "ids_weights"
                    else lock.weights_for_ids(logits2, indices, "softmax", 256.0)
                )
                assert torch.equal(w, expected)
                # Emulate the later identity kernel mutating its input buffers.
                i.fill_(0)
                w.zero_()
        tools.audit(NS(routes=str(td / mode), reference=str(td / "reference.json")))
    # Mismatched reference is rejected, and disabled/unrelated requests remain unchanged.
    os.environ["LC_ROUTE_MODE"] = "capture"
    os.environ["LC_ROUTE_OUT"] = str(td / "other")
    t = trace(0)
    t.input_ids[0] += 1
    w, i = lock.apply_route_lock(moe, t, logits, weights, indices)
    assert w is weights and i is indices and not (td / "other").exists()
    cfg.renormalize = True
    try:
        lock.apply_route_lock(moe, trace(0), logits, weights, indices)
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong TopK semantics accepted")
    cfg.renormalize = False
    missing = td / "capture" / "route_L1_TP7.json"
    missing_data = missing.read_text()
    missing.unlink()
    try:
        tools.pack(NS(routes=str(td / "capture"), out=str(td / "bad.json")))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing rank accepted")
    missing.write_text(missing_data)
    full_module = module("compare_full_route_tensors")
    delta = torch.tensor(
        [[0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, float("nan"), float("inf")]],
        dtype=torch.float64,
    )
    measured = full_module.compare_values(
        torch.zeros_like(delta), delta, [851], chunk=3
    )
    assert list(measured["bins"].values()) == [1, 1, 1, 1, 1, 2], measured
    assert measured["nonfinite_pairs"] == 2 and measured["max_abs"] == 1.0
    assert measured["max_difference"]["coordinate"] == [0, 6]
    assert measured["first_difference"]["coordinate"] == [0, 1]
    os.environ["LC_FULL_DUMP_OUT"] = str(td / "dump_alias")
    original = torch.ones(852, 4, dtype=torch.bfloat16)
    lock.dump_full_tensor(trace(0), 0, "moe.input", original)
    original.zero_()
    captured = torch.load(td / "dump_alias/full_L0_moe.input_TP0.pt", weights_only=True)
    assert captured["tensor"].count_nonzero() == 852 * 4
    import contextlib
    import io
    import shutil
    import sys

    for side in ["a", "b"]:
        (td / side / "routes").mkdir(parents=True)
        (td / side / "full").mkdir()
        for route_file in (td / "ids_weights").glob("route_*.json"):
            shutil.copyfile(route_file, td / side / "routes" / route_file.name)
        for layer, point in full_module.STAGES:
            rows = [851] if point == "lm_head.logits" else list(range(852))
            for rank in range(8):
                value = torch.zeros(len(rows), 2)
                if side == "b" and (layer, point, rank) == (0, "moe.branch_output", 3):
                    value[426, 1] = 0.125
                data = {
                    "schema": 1,
                    "layer": layer,
                    "point": point,
                    "tp": rank,
                    "identity": ref["identity"],
                    "mode": "EXTEND",
                    "rows": rows,
                    "route_mode": "ids_weights",
                    "tensor": value,
                }
                torch.save(
                    data, td / side / "full" / f"full_L{layer}_{point}_TP{rank}.pt"
                )
    sys.argv = [
        "compare_full_route_tensors.py",
        "--a3",
        str(td / "a"),
        "--a5",
        str(td / "b"),
        "--out",
        str(td / "comparison"),
        "--require-fixed",
        "--threads",
        "1",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        full_module.main()
    summary = json.loads((td / "comparison/summary.json").read_text())
    assert (
        summary["meta"]["tensor_pairs"] == 416
        and summary["meta"]["fixed_routes_verified"]
    )
    changed = [s for s in summary["stages"] if s["different"]]
    assert (
        len(changed) == 1
        and changed[0]["max_abs"] == 0.125
        and changed[0]["bins"]["[0.1,inf)"] == 1
    )
    assert changed[0]["max_difference"]["tp"] == 3 and changed[0]["max_difference"][
        "coordinate"
    ] == [426, 1]
    print(
        "PASS full tensors: copy-before-mutation, all 416 pairs, bin edges/chunking/nonfinite accounting, fixed-route equality, injected coordinate"
    )
print(
    "PASS: CPU capture/pack, all-rank equality, same IDs+weights, IDs-only expert-indexed weights, identity mutation isolation, wrong-input bypass, config and missing-rank rejection"
)
