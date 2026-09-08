#!/usr/bin/env python3
"""Compare two HCCL debug directories offline, using only stdlib by default.

--tensors adds exact numerical comparison of saved CPU tensors (needs torch).
Exit 0: all captured tensors bitwise equal; 1: differences; 2: incomplete or
unalignable evidence. Equality is only for the captured window, not proof of
correctness. Never compare TP8 and TP32 dumps as a communication-mode A/B.
"""

import argparse
import hashlib
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


def _saved_tensor(directory, event, name):
    import torch

    summary = event["tensors"][name]
    if "file" not in summary:
        raise ValueError(
            f"{directory}: event {event['event']} {name} has no saved tensor"
        )
    value = torch.load(
        directory / summary["file"], map_location="cpu", weights_only=True
    )
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if (
        list(value.shape) != summary["shape"]
        or str(value.dtype) != summary["dtype"]
        or hashlib.sha256(raw).hexdigest() != summary["sha256"]
    ):
        raise ValueError(f"{directory}: saved tensor does not match its JSON summary")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("AllReduce reference requires finite floating-point tensors")
    return value


def _allreduce_pair(run, event_id):
    directory, _, records = run
    after = next((e for e in records if e.get("event") == event_id), None)
    if (
        after is None
        or after["edge"] != "after"
        or after["scope"][-1] != "tp.all_reduce"
    ):
        raise ValueError(f"{directory}: event {event_id} is not an AllReduce output")
    before = next(
        (
            e
            for e in reversed(records)
            if (
                e.get("type") == "snapshot"
                and e["event"] < event_id
                and e["root"] == after["root"]
                and e["scope"] == after["scope"]
                and e["edge"] == "before"
            )
        ),
        None,
    )
    if before is None:
        raise ValueError(f"{directory}: missing AllReduce input event")
    return before, after


def allreduce_reference(left, right, event_id, rank=0):
    """Compare both modes to a CPU FP64 SUM of the SAME saved per-rank inputs.

    This diagnoses the selected SUM boundary, not end-to-end accept behavior.
    It neither runs NPU communication nor claims FP64 addition is exact.
    """
    import torch

    runs = {"left": load_run(left), "right": load_run(right)}
    if rank not in runs["left"]:
        raise ValueError(f"Missing reference rank {rank}")
    _, anchor = _allreduce_pair(runs["left"][rank], event_id)
    members = anchor["metadata"]["ranks"]
    if rank not in members or len(set(members)) != len(members):
        raise ValueError("Invalid AllReduce group members")
    pairs = {}
    reference = None
    for member in members:
        for side, run in runs.items():
            if member not in run:
                raise ValueError(f"{side}: missing group member {member}")
            before, after = _allreduce_pair(run[member], event_id)
            for event in (before, after):
                meta = event["metadata"]
                if (
                    event["root"] != anchor["root"]
                    or event["scope"] != anchor["scope"]
                    or meta["ranks"] != members
                    or meta["world_size"] != len(members)
                    or meta["rank_in_group"] != members.index(member)
                    or meta["group"] != anchor["metadata"]["group"]
                ):
                    raise ValueError(
                        f"{side} rank {member}: different collective identity"
                    )
            pairs[side, member] = before, after
        a = _saved_tensor(runs["left"][member][0], pairs["left", member][0], "input_")
        b = _saved_tensor(runs["right"][member][0], pairs["right", member][0], "input_")
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or pairs["left", member][0]["tensors"]["input_"]["sha256"]
            != pairs["right", member][0]["tensors"]["input_"]["sha256"]
        ):
            raise ValueError(
                f"rank {member}: inputs differ between modes; no common reference"
            )
        if reference is None:
            reference = torch.zeros_like(a, dtype=torch.float64)
            dtype = a.dtype
        if a.shape != reference.shape or a.dtype != dtype:
            raise ValueError("Group members have different input shapes/dtypes")
        reference.add_(a.double())
    rounded = reference.to(dtype)
    reference_norm = float(torch.linalg.vector_norm(reference))
    report = {
        "event": event_id,
        "root": anchor["root"],
        "scope": anchor["scope"],
        "group_ranks": members,
        "shape": list(reference.shape),
        "dtype": str(dtype),
        "all_rank_inputs_match_between_modes": True,
        "reference": "CPU FP64 SUM in group-rank order; also rounded once to input dtype",
        "modes": {},
    }
    for side, run in runs.items():
        outputs = []
        for member in members:
            _, after = pairs[side, member]
            output = _saved_tensor(run[member][0], after, "result")
            if output.shape != reference.shape or output.dtype != dtype:
                raise ValueError("AllReduce output shape/dtype differs from input")
            delta = output.double() - reference
            error_norm = float(torch.linalg.vector_norm(delta))
            outputs.append(
                {
                    "rank": member,
                    "sha256": after["tensors"]["result"]["sha256"],
                    "max_abs_vs_fp64": float(delta.abs().max())
                    if delta.numel()
                    else 0.0,
                    "relative_l2_vs_fp64": error_norm / reference_norm
                    if reference_norm
                    else (0.0 if not error_norm else "inf"),
                    "num_different_from_rounded_reference": int(
                        (output != rounded).sum()
                    ),
                    "numel": output.numel(),
                    "max_abs_vs_rounded_reference": float(
                        (output.double() - rounded.double()).abs().max()
                    )
                    if output.numel()
                    else 0.0,
                }
            )
        report["modes"][side] = {
            "requested_mode": run[rank][1].get("hccl_op_expansion_mode_requested"),
            "outputs_bitwise_equal_across_group": len({e["sha256"] for e in outputs})
            == 1,
            "outputs": outputs,
        }
    return report


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
        for field in ("torch", "limits", "configuration", "graph_state", "simulation"):
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
                entry["alignment_mismatch"] = {
                    "left_event": x["event"],
                    "right_event": y["event"],
                    "left_key": event_key(x),
                    "right_key": event_key(y),
                    "left_metadata": x["metadata"],
                    "right_metadata": y["metadata"],
                }
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


def inspect_scopes(directory, scopes, rank=0, limit=16):
    """Read each run independently; never align events across divergent states."""
    runs = load_run(directory)
    if rank not in runs:
        raise ValueError(f"{directory}: missing inspection rank {rank}")
    parent, manifest, records = runs[rank]
    selected = [
        e
        for e in records
        if e.get("type") == "snapshot" and any(s in e["scope"] for s in scopes)
    ]
    events = []
    for event in selected[:limit]:
        entry = {k: event[k] for k in ("event", "root", "scope", "edge", "metadata")}
        entry["tensors"] = {}
        for name, summary in event["tensors"].items():
            entry["tensors"][name] = {
                k: summary[k]
                for k in (
                    "shape",
                    "dtype",
                    "sha256",
                    "values",
                    "topk_indices",
                    "topk_values",
                    "dump_skipped",
                )
                if k in summary
            }
            entry["tensors"][name]["saved_file_exists"] = (
                (parent / summary["file"]).is_file() if "file" in summary else False
            )
        events.append(entry)
    simulation = manifest.get("simulation")
    acc_len = (simulation or {}).get("SGLANG_SIMULATE_ACC_LEN")
    return {
        "rank": rank,
        "simulation": simulation,
        "accept_length_evidence": (
            "unknown: old manifest; check launch environment"
            if acc_len is None
            else "simulated: final lengths are not real acceptance evidence"
            if acc_len > 0
            else "simulation disabled; still check request/state alignment"
        ),
        "matching_events": len(selected),
        "omitted_events": max(0, len(selected) - limit),
        "event_limit_reached": any(e.get("type") == "limit" for e in records),
        "events": events,
    }


def brief_report(report):
    """Keep all-rank status, omit repeated tensors and large expert counts."""
    result = {k: report[k] for k in ("left_ranks", "right_ranks", "incomplete")}
    result["rank_groups"] = []
    groups = {}
    for rank in report["ranks"]:
        entry = {k: rank[k] for k in ("rank", "bitwise_different_tensors")}
        diff = rank.get("first_difference")
        if diff:
            entry["first_difference"] = {
                k: diff[k] for k in ("event", "root", "scope", "edge", "tensor")
            }
            if "numerical" in diff:
                entry["first_difference"]["numerical"] = diff["numerical"]
        mismatch = rank.get("alignment_mismatch")
        if mismatch:
            entry["alignment_mismatch"] = {
                k: mismatch[k]
                for k in ("left_event", "right_event", "left_key", "right_key")
            }
            a, b = mismatch["left_metadata"], mismatch["right_metadata"]
            entry["alignment_mismatch"]["changed_metadata_fields"] = sorted(
                k for k in a.keys() | b.keys() if a.get(k) != b.get(k)
            )
        member = entry.pop("rank")
        key = json.dumps(entry, sort_keys=True)
        if key not in groups:
            groups[key] = {"ranks": [], **entry}
            result["rank_groups"].append(groups[key])
        groups[key]["ranks"].append(member)
    if "allreduce_reference" in report:
        ref = report["allreduce_reference"]
        result["allreduce_reference"] = {k: v for k, v in ref.items() if k != "modes"}
        result["allreduce_reference"]["modes"] = {}
        for name, mode in ref["modes"].items():
            # Collapse only AFTER the reference validated every group member.
            result["allreduce_reference"]["modes"][name] = {
                **mode,
                "outputs": mode["outputs"][:1]
                if mode["outputs_bitwise_equal_across_group"]
                else mode["outputs"],
            }
    for key in ("allreduce_reference_error", "scope_inspection", "interpretation"):
        if key in report:
            result[key] = report[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--tensors", action="store_true")
    parser.add_argument(
        "--allreduce-event",
        type=int,
        help="Also compare this AllReduce after-event to a saved-input FP64 SUM",
    )
    parser.add_argument(
        "--reference-rank",
        type=int,
        default=0,
        help="Rank whose group selects the AllReduce reference (default: 0)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Short console JSON; --output still saves full report",
    )
    parser.add_argument(
        "--inspect-scope",
        action="append",
        default=[],
        help="Exact scope component to inspect independently per run; repeatable",
    )
    parser.add_argument("--inspect-rank", type=int, default=0)
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=16,
        help="Max selected events per run (default: 16)",
    )
    args = parser.parse_args()
    if args.inspect_limit <= 0:
        parser.error("--inspect-limit must be positive")
    try:
        report, status = compare_runs(args.left, args.right, tensors=args.tensors)
        if args.inspect_scope:
            report["scope_inspection"] = {
                "interpretation": "Independent records, NOT aligned tensor pairs. Equal step numbers do not prove equal prefixes, cache state or valid rows.",
                "left": inspect_scopes(
                    args.left, args.inspect_scope, args.inspect_rank, args.inspect_limit
                ),
                "right": inspect_scopes(
                    args.right,
                    args.inspect_scope,
                    args.inspect_rank,
                    args.inspect_limit,
                ),
            }
        if args.allreduce_event is not None:
            try:
                report["allreduce_reference"] = allreduce_reference(
                    args.left, args.right, args.allreduce_event, args.reference_rank
                )
            except (ValueError, OSError, KeyError) as exc:
                report["allreduce_reference_error"] = str(exc)
                status = 2
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot compare: {exc}\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(
        json.dumps(brief_report(report), ensure_ascii=False, indent=2, allow_nan=False)
        if args.brief
        else encoded
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
