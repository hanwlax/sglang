#!/usr/bin/env python3
"""Audit full Eliza capture coverage, payloads and same-platform repeats.

No cross-quantization tolerance or accuracy verdict is inferred from equality.
Supports a shared dump directory or disjoint per-host dump roots.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

from replay_fixed_decode_dp4 import load_inputs

REQUIRED = [
    "attn0.input",
    "attn0.residual",
    "attn0.local_output",
    "moe.input",
    "router.logits",
    "router.expert_ids",
    "router.weights",
    "moe.dispatch_ids",
    "moe.dispatch_weights",
    "moe.experts_output",
    "moe.reduced",
    "moe.branch_output",
    "ffn0.input",
    "ffn0.residual_in",
    "ffn0.local_output",
    "attn1.input",
    "attn1.residual",
    "attn1.local_output",
    "ffn1.input",
    "ffn1.residual_in",
    "ffn1.local_output",
    "layer.output",
    "layer.residual",
]
INTERNAL_STAGES = ("a_proj", "norm_qb", "qk_bmm", "rope", "sparse", "tail")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def locate(roots, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, str(relative))
    paths = [root / relative for root in roots if (root / relative).is_file()]
    require(len(paths) == 1, f"Expected exactly one {relative}, found {len(paths)}")
    return paths[0]


def check_payloads(item):
    import torch

    torch.set_num_threads(1)
    index_path, roots = item
    index = json.loads(index_path.read_text())
    count = nonfinite = 0
    for sample in index["samples"]:
        if sample["status"] != "full_valid":
            continue
        path = locate(roots, sample["values_file"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        value = payload["values"]
        require(
            payload["rid"] == index["rid"] and payload["mode"] == index["mode"],
            str(path),
        )
        require(payload["topology"] == index["topology"], str(path))
        for key in (
            "layer",
            "point",
            "rows",
            "positions",
            "token_ids",
            "value_shape",
            "dtype",
        ):
            require(payload[key] == sample[key], f"Payload metadata {key}: {path}")
        require(
            list(value.shape) == sample["value_shape"]
            and str(value.dtype) == sample["dtype"],
            str(path),
        )
        digest = hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy()
        ).hexdigest()
        require(
            digest == sample["tensor_sha256"] == payload["tensor_sha256"],
            f"Hash mismatch: {path}",
        )
        if value.is_floating_point():
            nonfinite += int((~torch.isfinite(value)).sum())
        count += 1
    return count, nonfinite


def check_internal(path):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    expected = f"{data['rid']}.tp{data['topology']['tp_rank']}.{data['mode']}.attn{data['attention_id']}.{data['stage']}.pt"
    require(
        expected == path.name and data["layer"] == data["attention_id"] // 2, str(path)
    )
    require(isinstance(data["values"], dict), str(path))
    # Raw cache includes unused slots; do not classify all raw cache as live data.
    return 1


def validate_launches(roots):
    launch_roots = list(dict.fromkeys(root.parent for root in roots))
    runtime_hashes = []
    for node in range(4):
        launch = json.loads(locate(launch_roots, f"launch_rank{node}.json").read_text())
        receipt = json.loads(
            locate(launch_roots, f"runtime_rank{node}.json").read_text()
        )
        argv, env = launch["args"], launch["env"]
        for key, value in {
            "--tp-size": "32",
            "--attn-dp-size": "2",
            "--ep-size": "32",
            "--node-rank": str(node),
            "--moe-a2a-backend": "deepep",
            "--deepep-mode": "auto",
            "--quantization": "fp8",
            "--cuda-graph-backend-prefill": "disabled",
            "--cuda-graph-backend-decode": "disabled",
        }.items():
            require(
                key in argv and argv[argv.index(key) + 1] == value,
                f"Launch {node}: {key}",
            )
        require(
            "--speculative-algorithm" not in argv and "--disable-radix-cache" in argv,
            f"Launch {node}: NEXTN/cache",
        )
        for key, value in {
            "HCCL_OP_EXPANSION_MODE": "AI_CPU",
            "MOE_ENABLE_TOPK_NEG_ONE": "1",
            "DEEPEP_HYBRID_DEPLOYMENT": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "0",
            "LC_INTERNAL_LAYERS": "0,7,37",
            "LC_INTERNAL_RANKS": ",".join(map(str, range(16))),
        }.items():
            require(env.get(key) == value, f"Launch {node}: {key}")
        require(
            receipt["mode"] == "trace"
            and receipt["internal_attention"]
            and receipt["capture"] == "full_valid_boundaries",
            f"Runtime {node}",
        )
        require(
            receipt["topology"]
            == {"tp_size": 32, "attn_dp_size": 2, "moe_ep_size": 32},
            f"Runtime topology {node}",
        )
        require(
            "python/sglang/srt/debug_utils/longcat_capture_profile.json"
            in receipt["sha256"],
            f"Runtime profile hash missing: {node}",
        )
        runtime_hashes.append(receipt["sha256"])
    require(
        all(x == runtime_hashes[0] for x in runtime_hashes),
        "Four nodes used different runtime sources",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("a3-dp4", "a5-dp2"), default="a5-dp2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dump", type=Path, nargs="+", required=True)
    parser.add_argument("--verify-payloads", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.workers > 0, "workers must be positive")
    roots = list(dict.fromkeys(path.resolve() for path in args.dump))
    dp_size = 2 if args.topology == "a5-dp2" else 4
    tp_size = dp_size * 16
    manifest = json.loads(args.manifest.read_text())
    prompt, prefix, _ = load_inputs(
        None, Path(__file__).parent / "fixtures/eliza_853.json", None
    )
    require(
        manifest["prompt_ids"] == prompt and manifest["prefix_ids"] == prefix,
        "Eliza fixture mismatch",
    )
    require(
        manifest["topology"]
        == {
            "tp_size": tp_size,
            "attn_dp_size": dp_size,
            "attn_tp_size": 16,
            "moe_ep_size": tp_size,
        },
        "Manifest topology mismatch",
    )
    require(
        manifest["dps"] == [0]
        and manifest["lengths"] == [1, 11]
        and manifest["repeat"] == 2,
        "Expected A3-matched DP0 / lengths1,11 / repeats2",
    )
    requests = manifest["requests"]
    require(len(requests) == 6, "Incomplete requests")
    require(
        {(x["length"], x["rep"]) for x in requests}
        == {(n, r) for n in (1, 11) for r in (0, 1, 2)},
        "Duplicate or missing request",
    )
    for req in requests:
        require(
            req["status"] == "ok"
            and req["dp"] == 0
            and req["warmup"] == (req["rep"] == 0),
            "Failed/pending/mismatched request",
        )
        n = len(prompt) + req["length"] - 1
        require(
            req["prompt_len"] == n and req["forced_id"] == prefix[req["length"] - 1],
            "Prefix mismatch",
        )
        require(
            len(req["output_ids"]) == 2 and req["output_ids"][0] == req["forced_id"],
            "Forced prefix verification failed",
        )
    for tp in range(tp_size):
        worker = json.loads(locate(roots, f"worker.tp{tp}.json").read_text())
        expected = {
            "tp_rank": tp,
            "tp_size": tp_size,
            "attn_dp_rank": tp // 16,
            "attn_dp_size": dp_size,
            "attn_tp_rank": tp % 16,
            "attn_tp_size": 16,
            "moe_ep_rank": tp,
            "moe_ep_size": tp_size,
            "moe_tp_size": 1,
            "moe_dp_size": 1,
            "attn_cp_size": 1,
            "pp_size": 1,
        }
        require(worker == expected, f"Worker topology mismatch: {tp}")
    if args.topology == "a5-dp2":
        validate_launches(roots)
    required = {
        (-1, name)
        for name in ("embedding.output", "final_norm.output", "lm_head.logits")
    }
    required |= {(layer, name) for layer in range(38) for name in REQUIRED}
    indexes, internal, compared = [], [], {}
    file_count = elements = byte_count = 0
    for req in (x for x in requests if not x["warmup"]):
        for mode in ("EXTEND", "DECODE"):
            ids = (
                prompt + prefix[: req["length"] - 1]
                if mode == "EXTEND"
                else [req["forced_id"]]
            )
            positions = (
                list(range(len(ids))) if mode == "EXTEND" else [req["prompt_len"]]
            )
            stages = collections.defaultdict(list)
            for tp in range(16):
                path = locate(roots, f"{req['rid']}.tp{tp}.{mode}.json")
                data = json.loads(path.read_text())
                require(
                    data["schema"] == 5 and data["selected_layers"] == list(range(38)),
                    str(path),
                )
                require(
                    data["rid"] == req["rid"]
                    and data["mode"] == mode
                    and data["tp"] == tp,
                    str(path),
                )
                require(
                    data["topology"]
                    == json.loads(locate(roots, f"worker.tp{tp}.json").read_text()),
                    str(path),
                )
                require(
                    data["input_ids"] == ids and data["positions"] == positions,
                    str(path),
                )
                samples = {(s["layer"], s["point"]): s for s in data["samples"]}
                require(
                    len(samples) == len(data["samples"]) and required <= samples.keys(),
                    f"Missing/duplicate boundary: {path}",
                )
                for key, sample in samples.items():
                    require(
                        sample["row_policy"] == "all_valid"
                        and sample["column_policy"] == "all",
                        str(path),
                    )
                    if sample["status"] == "absent":
                        require(
                            key not in required, f"Required tensor absent: {path} {key}"
                        )
                        continue
                    rows = sample["rows"]
                    require(
                        rows == sorted(set(rows))
                        and all(0 <= row < len(ids) for row in rows),
                        str(path),
                    )
                    require(
                        len(rows) == sample["live_rows"]
                        and sample["token_ids"] == [ids[i] for i in rows]
                        and sample["positions"] == [positions[i] for i in rows],
                        str(path),
                    )
                    if sample["status"] == "full_valid":
                        payload = locate(roots, sample["values_file"])
                        require(
                            payload.stat().st_size > 0
                            and sample["value_shape"][0] == len(rows),
                            str(payload),
                        )
                        require(
                            sample["elements"] == math.prod(sample["value_shape"]) > 0
                            and len(sample["tensor_sha256"]) == 64,
                            str(payload),
                        )
                        file_count += 1
                        elements += sample["elements"]
                        byte_count += payload.stat().st_size
                    else:
                        require(
                            sample["status"] == "no_live_rows"
                            and not rows
                            and sample["values_file"] is None,
                            str(path),
                        )
                    stages[key].append(rows)
                indexes.append(path)
                compared[req["length"], req["rep"], mode, tp] = samples
                if req["rep"] == 1:
                    for attention in (0, 1, 14, 15, 74, 75):
                        for stage in INTERNAL_STAGES:
                            internal.append(
                                locate(
                                    roots,
                                    f"internal/{req['rid']}.tp{tp}.{mode}.attn{attention}.{stage}.pt",
                                )
                            )
            for key, rows in stages.items():
                target = (
                    {len(ids) - 1}
                    if key[1] == "lm_head.logits"
                    else set(range(len(ids)))
                )
                require(
                    len(rows) == 16 and set().union(*map(set, rows)) == target,
                    f"Missing rank/token coverage: {req['rid']} {mode} {key}",
                )
    repeats = []
    for length in (1, 11):
        for mode in ("EXTEND", "DECODE"):
            equal = 0
            differences = []
            for tp in range(16):
                first, second = [compared[length, rep, mode, tp] for rep in (1, 2)]
                require(first.keys() == second.keys(), "Repeat stages mismatch")
                for order, (key, x) in enumerate(first.items()):
                    y = second[key]
                    require(
                        all(
                            x.get(k) == y.get(k)
                            for k in (
                                "status",
                                "rows",
                                "positions",
                                "dtype",
                                "value_shape",
                            )
                        ),
                        "Repeat layout mismatch",
                    )
                    if x["status"] != "full_valid":
                        continue
                    if x["tensor_sha256"] == y["tensor_sha256"]:
                        equal += 1
                    else:
                        differences.append(
                            {
                                "order": order,
                                "tp": tp,
                                "layer": key[0],
                                "point": key[1],
                                "r1_file": x["values_file"],
                                "r2_file": y["values_file"],
                            }
                        )
            differences.sort(key=lambda d: (d["order"], d["tp"]))
            item = {
                "length": length,
                "mode": mode,
                "equal_tensor_hashes": equal,
                "different_tensor_hashes": len(differences),
                "first_different": differences[0] if differences else None,
            }
            print("REPEAT", json.dumps(item), flush=True)
            repeats.append({**item, "differences": differences})
    nonfinite = None
    if args.verify_payloads:
        nonfinite = checked = 0
        with concurrent.futures.ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for i, (count, bad) in enumerate(
                pool.map(check_payloads, [(x, roots) for x in indexes]), 1
            ):
                checked += count
                nonfinite += bad
                if i % 16 == 0:
                    print(
                        "PAYLOAD_PROGRESS",
                        json.dumps(
                            {
                                "indexes_checked": i,
                                "total_indexes": len(indexes),
                                "files_checked": checked,
                                "nonfinite": nonfinite,
                            }
                        ),
                        flush=True,
                    )
            require(checked == file_count, "Payload count mismatch")
            require(
                sum(pool.map(check_internal, internal)) == len(internal),
                "Internal file count mismatch",
            )
    summary = {
        "status": "coverage_and_payloads_pass"
        if args.verify_payloads
        else "coverage_pass_payloads_unchecked",
        "topology": args.topology,
        "completed_indexes": len(indexes),
        "worker_receipts": tp_size,
        "boundary_files": file_count,
        "boundary_bytes": byte_count,
        "elements_including_replicas": elements,
        "internal_files": len(internal),
        "boundary_nonfinite": nonfinite,
        "payloads_verified": args.verify_payloads,
        "repeat_different_tensors": sum(x["different_tensor_hashes"] for x in repeats),
    }
    if nonfinite:
        summary["status"] = "nonfinite_boundary_data"
    args.out.write_text(
        json.dumps(
            {
                **summary,
                "repeats": repeats,
                "notes": [
                    "Repeat differences are diagnostic evidence, not an automatic accuracy failure.",
                    "Internal raw cache padding is excluded from finite-value verdicts.",
                    "No A3/A5 cross-quantization comparison is performed.",
                ],
            },
            indent=2,
        )
    )
    print("CAPTURE_SUMMARY", json.dumps(summary), flush=True)
    if nonfinite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
