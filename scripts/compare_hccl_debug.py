#!/usr/bin/env python3
"""Compare two HCCL debug directories offline, using only stdlib by default.

--tensors adds exact numerical comparison of saved CPU tensors (needs torch).
Exit 0: all captured tensors bitwise equal; 1: differences; 2: incomplete or
unalignable evidence. Equality is only for the captured window, not proof of
correctness. Never compare TP8 and TP32 dumps as a communication-mode A/B.
"""

import argparse
import json
from pathlib import Path


def load_run(directory):
    ranks = {}
    for path in sorted(Path(directory).rglob("events.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not records or records[0].get("type") != "manifest":
            raise ValueError(f"Missing manifest: {path}")
        manifest = records[0]
        rank = manifest["rank"]
        if rank in ranks:
            raise ValueError(f"Duplicate rank {rank}: use one run per directory")
        ranks[rank] = (path.parent, manifest, records[1:])
    if not ranks:
        raise ValueError(f"No events.jsonl found under {directory}")
    return ranks


def numeric_diff(left, right):
    import torch

    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"error": "shape or dtype mismatch"}
    result = {"num_different": int((a != b).sum())}
    if not a.is_floating_point():
        return result
    finite = torch.isfinite(a) & torch.isfinite(b)
    result["num_nonfinite_pairs"] = int((~finite).sum())
    av, bv = a[finite].double(), b[finite].double()
    if av.numel():
        delta = av - bv
        result["max_abs"] = float(delta.abs().max())
        denominator = float(torch.linalg.vector_norm(av))
        numerator = float(torch.linalg.vector_norm(delta))
        result["relative_l2"] = (
            numerator / denominator
            if denominator
            else (0.0 if not numerator else "inf")
        )
    # Only call this an argmax comparison: not every tensor is a logits tensor.
    if a.ndim >= 2 and a.shape[-1] and a.numel():
        result["last_dim_argmax_different"] = int((a.argmax(-1) != b.argmax(-1)).sum())
    return result


def event_key(event):
    return event["root"], event["scope"], event["edge"]


def compare_runs(left, right, *, tensors=False):
    a, b = load_run(left), load_run(right)
    report = {
        "left_ranks": sorted(a),
        "right_ranks": sorted(b),
        "ranks": [],
        "incomplete": [],
    }
    if a.keys() != b.keys():
        report["incomplete"].append("rank sets differ")
    for run_name, ranks in (("left", a), ("right", b)):
        world_sizes = {entry[1].get("world_size") for entry in ranks.values()}
        if len(world_sizes) != 1 or None in world_sizes:
            report["incomplete"].append(f"{run_name}: missing/inconsistent world_size")
        elif set(ranks) != set(range(next(iter(world_sizes)))):
            report["incomplete"].append(f"{run_name}: not all global ranks collected")
    any_difference = False
    for rank in sorted(a.keys() & b.keys()):
        adir, am, ae = a[rank]
        bdir, bm, be = b[rank]
        entry = {"rank": rank, "bitwise_different_tensors": 0, "first_difference": None}
        report["ranks"].append(entry)
        if am.get("world_size") != bm.get("world_size"):
            report["incomplete"].append(f"rank {rank}: different topology")
            continue
        for field in ("torch", "limits", "configuration", "graph_state"):
            if am.get(field) != bm.get(field):
                report["incomplete"].append(f"rank {rank}: manifest {field} differs")
        entry["requested_modes"] = [
            am.get("hccl_op_expansion_mode_requested"),
            bm.get("hccl_op_expansion_mode_requested"),
        ]
        if any(e["type"] == "limit" for e in ae + be):
            report["incomplete"].append(f"rank {rank}: event limit reached")
        ae = [e for e in ae if e["type"] == "snapshot"]
        be = [e for e in be if e["type"] == "snapshot"]
        for side, events in (("left", ae), ("right", be)):
            stack = []
            for event in events:
                if event["edge"] == "before":
                    stack.append((event["root"], event["scope"]))
                elif not stack or stack.pop() != (event["root"], event["scope"]):
                    report["incomplete"].append(f"rank {rank}: {side} unbalanced trace")
                    break
            if stack:
                report["incomplete"].append(f"rank {rank}: {side} unfinished scope")
        if not ae or not be or len(ae) != len(be):
            report["incomplete"].append(
                f"rank {rank}: empty or different event counts ({len(ae)}, {len(be)})"
            )
        for x, y in zip(ae, be):
            if event_key(x) != event_key(y) or x["metadata"] != y["metadata"]:
                report["incomplete"].append(
                    f"rank {rank}: event {x['event']} scope/metadata diverged; stop alignment"
                )
                break
            if x["tensors"].keys() != y["tensors"].keys():
                report["incomplete"].append(
                    f"rank {rank}: event {x['event']} tensor keys differ"
                )
                break
            for name, xs in x["tensors"].items():
                ys = y["tensors"][name]
                if all(xs[k] == ys[k] for k in ("shape", "dtype", "sha256")):
                    continue
                any_difference = True
                entry["bitwise_different_tensors"] += 1
                if entry["first_difference"] is None:
                    diff = {
                        "event": x["event"],
                        "root": x["root"],
                        "scope": x["scope"],
                        "edge": x["edge"],
                        "tensor": name,
                        "left": xs,
                        "right": ys,
                    }
                    if tensors and "file" in xs and "file" in ys:
                        diff["numerical"] = numeric_diff(
                            adir / xs["file"], bdir / ys["file"]
                        )
                    entry["first_difference"] = diff
    report["interpretation"] = (
        "First differences are per rank in execution order. Check the same token prefix, "
        "positions, masks, valid rows and ALL ranks' collective inputs before attributing "
        "a difference to HCCL. DeepEP dispatch order/padding can differ. "
        "Requested expansion mode does not prove the actual HCCL engine."
    )
    status = 2 if report["incomplete"] else 1 if any_difference else 0
    return report, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--tensors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report, status = compare_runs(args.left, args.right, tensors=args.tensors)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot compare: {exc}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
