#!/usr/bin/env python3
"""Replay a saved, full-world single-node HCCL SUM without importing SGLang.

check/summarize need CPU torch; run needs torch_npu and torchrun.
Each mode MUST run in a fresh process group. No model, graph or custom kernels.
"""

import argparse
import hashlib
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
from compare_hccl_debug import (
    _allreduce_pair,
    _saved_tensor,
    allreduce_reference,
    load_run,
)


def digest(value):
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def prepare_case(left, right, event, rank=0):
    # Reuse the offline verifier: checks every member's identity, input equality,
    # shape/dtype, finite data and .pt digests, including both recorded outputs.
    report = allreduce_reference(left, right, event)
    members = report["group_ranks"]
    if len(members) < 2 or members != list(range(len(members))):
        raise ValueError("Replay requires a full-world group ordered [0, ..., N-1]")
    if rank not in members:
        raise ValueError(f"Replay rank {rank} not in saved group")
    runs = [load_run(left), load_run(right)]
    for run in runs:
        if set(run) != set(members):
            raise ValueError("Source must contain exactly the replay group")
        hosts = {entry[1].get("hostname") for entry in run.values()}
        if len(hosts) != 1 or None in hosts:
            raise ValueError(
                "Source must be a single-node capture with hostname metadata"
            )
        if any(entry[1].get("world_size") != len(members) for entry in run.values()):
            raise ValueError("Saved global world size differs from replay group")
    reference = None
    source = None
    input_hashes = []
    for member in members:
        directory, _, _ = runs[0][member]
        before, _ = _allreduce_pair(runs[0][member], event)
        value = _saved_tensor(directory, before, "input_")
        for run in runs:
            other_before, _ = _allreduce_pair(run[member], event)
            if other_before["tensors"]["input_"]["stride"] != list(value.stride()):
                raise ValueError(
                    "Replay currently requires originally contiguous inputs"
                )
        if not value.numel():
            raise ValueError("Empty inputs are not supported by this replay")
        input_hashes.append(digest(value))
        if reference is None:
            reference = torch.zeros_like(value, dtype=torch.float64)
        reference.add_(value.double())
        if member == rank:
            source = value
    return source, reference, report, input_hashes


def error_metrics(output, reference):
    nonfinite = int((~torch.isfinite(output)).sum())
    if nonfinite:
        return {
            "nonfinite": nonfinite,
            "max_abs_vs_fp64": None,
            "relative_l2_vs_fp64": None,
        }
    delta = output.double() - reference
    denominator = float(torch.linalg.vector_norm(reference))
    numerator = float(torch.linalg.vector_norm(delta))
    return {
        "nonfinite": 0,
        "max_abs_vs_fp64": float(delta.abs().max()),
        "relative_l2_vs_fp64": numerator / denominator
        if denominator
        else (0.0 if numerator == 0 else "inf"),
        "num_different_from_rounded_reference": int(
            (output != reference.to(output.dtype)).sum()
        ),
    }


def replay_iterations(
    source, iterations, warmup, *, allocate, synchronize, reduce, observe
):
    """Restore the CPU source before EVERY in-place SUM, including warmups."""
    work = allocate(source)
    for index in range(warmup + iterations):
        work.copy_(source)
        synchronize()
        reduce(work)
        synchronize()
        output = work.detach().to(device="cpu", copy=True).contiguous()
        observe(index, "warmup" if index < warmup else "measured", output)


def run_replay(args):
    mode = os.environ.get("HCCL_OP_EXPANSION_MODE")
    if mode != args.mode:
        raise ValueError(
            f"Set HCCL_OP_EXPANSION_MODE={args.mode} before starting torchrun; got {mode!r}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if int(os.environ["LOCAL_WORLD_SIZE"]) != world or local_rank != rank:
        raise ValueError("Use single-node torchrun with one process per visible NPU")
    source, reference, case, input_hashes = prepare_case(
        args.left, args.right, args.event, rank
    )
    if world != len(case["group_ranks"]):
        raise ValueError("torchrun world size differs from the saved group")
    directory = args.output / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    import torch_npu  # noqa: F401 -- registers torch.npu and HCCL

    torch.npu.set_device(local_rank)
    torch.npu.synchronize()
    report = {
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_name": torch.npu.get_device_name(local_rank),
        "requested_mode": mode,
        "actual_hccl_engine": "unknown; consult HCCL logs",
        "source_event": args.event,
        "source_root": case["root"],
        "source_scope": case["scope"],
        "group_ranks": case["group_ranks"],
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "all_input_sha256": input_hashes,
        "left": str(Path(args.left).resolve()),
        "right": str(Path(args.right).resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HCCL_BUFFSIZE",
                "HCCL_ALGO",
                "HCCL_DETERMINISTIC",
                "ASCEND_RT_VISIBLE_DEVICES",
                "HCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "HCCL_EXEC_TIMEOUT",
            )
        },
        "records": [],
        "completed": False,
    }
    saved = {}
    baselines = {
        side: {
            "requested_mode": mode_report["requested_mode"],
            "sha256": mode_report["outputs"][rank]["sha256"],
        }
        for side, mode_report in case["modes"].items()
    }
    report["model_outputs"] = baselines
    if rank == 0:
        print(
            "Saved inputs/outputs validated; initializing a fresh HCCL group",
            flush=True,
        )

    def observe(index, phase, output):
        sha = digest(output)
        if sha not in saved:
            filename = f"output-{index:03d}.pt"
            torch.save(output, directory / filename)
            saved[sha] = filename
        entry = {
            "iteration": index,
            "phase": phase,
            "sha256": sha,
            "file": saved[sha],
            "matches_model": {
                side: sha == value["sha256"] for side, value in baselines.items()
            },
            **error_metrics(output, reference),
        }
        report["records"].append(entry)
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    torch.distributed.init_process_group(
        "hccl", timeout=timedelta(seconds=args.timeout)
    )
    try:
        with torch.inference_mode():
            replay_iterations(
                source,
                args.iterations,
                args.warmup,
                allocate=lambda value: torch.empty_like(
                    value, device=f"npu:{local_rank}"
                ),
                synchronize=torch.npu.synchronize,
                reduce=lambda value: torch.distributed.all_reduce(
                    value, op=torch.distributed.ReduceOp.SUM
                ),
                observe=observe,
            )
        report["completed"] = True
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "requested_mode": mode,
                    "unique_outputs": len(saved),
                    "report": str(directory / "report.json"),
                }
            ),
            flush=True,
        )
    finally:
        torch.distributed.destroy_process_group()


def summarize(directory):
    reports = [
        json.loads(p.read_text())
        for p in sorted(Path(directory).glob("rank-*/report.json"))
    ]
    if not reports:
        raise ValueError(f"No replay reports in {directory}")
    anchor = reports[0]
    if sorted(r["rank"] for r in reports) != list(range(anchor["world_size"])):
        raise ValueError("Missing or duplicate replay ranks")
    for report in reports:
        for field in (
            "world_size",
            "requested_mode",
            "source_event",
            "source_root",
            "source_scope",
            "group_ranks",
            "all_input_sha256",
            "shape",
            "dtype",
            "warmup",
            "iterations",
        ):
            if report[field] != anchor[field]:
                raise ValueError(f"Replay manifest mismatch: {field}")
        records = report["records"]
        if (
            not report["completed"]
            or len(records) != report["warmup"] + report["iterations"]
        ):
            raise ValueError(f"rank {report['rank']}: incomplete replay")
        for index, entry in enumerate(records):
            if entry["iteration"] != index:
                raise ValueError("Replay iteration order mismatch")
            path = Path(directory) / f"rank-{report['rank']}" / entry["file"]
            value = torch.load(path, map_location="cpu", weights_only=True)
            if (
                digest(value) != entry["sha256"]
                or list(value.shape) != report["shape"]
                or str(value.dtype) != report["dtype"]
            ):
                raise ValueError(f"Replay tensor does not match report: {path}")
    count = anchor["warmup"] + anchor["iterations"]
    return {
        "directory": str(directory),
        "requested_mode": anchor["requested_mode"],
        "shape": anchor["shape"],
        "dtype": anchor["dtype"],
        "group_ranks": anchor["group_ranks"],
        "all_input_sha256": anchor["all_input_sha256"],
        "all_ranks_repeatable_including_warmup": all(
            len({e["sha256"] for e in r["records"]}) == 1 for r in reports
        ),
        "all_ranks_equal_each_iteration": all(
            len({r["records"][i]["sha256"] for r in reports}) == 1 for i in range(count)
        ),
        "all_outputs_match_model": {
            side: all(e["matches_model"][side] for r in reports for e in r["records"])
            for side in ("left", "right")
        },
        "any_nonfinite": any(e["nonfinite"] for r in reports for e in r["records"]),
        "first_output_per_rank": [
            {"rank": r["rank"], **r["records"][0]} for r in reports
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--left", required=True)
        sub.add_argument("--right", required=True)
        sub.add_argument("--event", type=int, default=3)
        if name == "run":
            sub.add_argument("--mode", required=True, choices=("AIV", "CCU_SCHED"))
            sub.add_argument("--output", required=True, type=Path)
            sub.add_argument("--iterations", type=int, default=10)
            sub.add_argument("--warmup", type=int, default=2)
            sub.add_argument("--timeout", type=int, default=300)
    sub = commands.add_parser("summarize")
    sub.add_argument("directories", nargs="+", type=Path)
    sub.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        _, _, report, hashes = prepare_case(args.left, args.right, args.event)
        print(
            json.dumps(
                {
                    "ready": True,
                    "group_ranks": report["group_ranks"],
                    "shape": report["shape"],
                    "dtype": report["dtype"],
                    "all_rank_inputs_match": True,
                    "input_sha256": hashes,
                },
                indent=2,
            )
        )
    elif args.command == "run":
        if args.iterations < 1 or args.warmup < 0 or args.timeout < 1:
            parser.error(
                "iterations/timeout must be positive; warmup must be nonnegative"
            )
        run_replay(args)
    else:
        reports = [summarize(p) for p in args.directories]
        for report in reports:
            if any(
                report[key] != reports[0][key]
                for key in ("all_input_sha256", "shape", "dtype", "group_ranks")
            ):
                raise ValueError("Replay runs used different inputs/layouts/groups")
        result = {
            "runs": reports,
            "interpretation": "Fresh single-node HCCL groups; default NPU stream, restored inputs and device synchronization. Matching model outputs supports independence from model scheduling for this boundary only. Requested mode does not prove actual engine.",
        }
        encoded = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(encoded + "\n")
        for report in reports:
            report.pop("all_input_sha256")
            if report["all_ranks_equal_each_iteration"]:
                report["first_output_per_rank"] = report["first_output_per_rank"][:1]
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
