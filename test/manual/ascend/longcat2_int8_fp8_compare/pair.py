#!/usr/bin/env python3
"""Freeze an existing working launcher and evaluate a paired GSM8K experiment."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import socket
import sys
import urllib.request
from pathlib import Path

MODEL_FLAGS = {
    "--model-path",
    "--tokenizer-path",
    "--speculative-draft-model-path",
    "--quantization",
    "--speculative-draft-model-quantization",
}
ENV_PREFIXES = (
    "SGLANG_",
    "HCCL_",
    "DEEPEP_",
    "MOE_",
    "ASCEND_",
    "TORCH_NPU_",
    "PYTORCH_NPU_",
    "GLOO_",
    "STREAMS_PER_DEVICE",
)
POLICY = (
    "tp_size",
    "attn_dp_size",
    "ep_size",
    "nnodes",
    "device",
    "dtype",
    "attention_backend",
    "kv_cache_dtype",
    "moe_a2a_backend",
    "deepep_mode",
    "deepep_dispatcher_output_dtype",
    "enable_dp_lm_head",
    "page_size",
    "mem_fraction_static",
    "max_total_tokens",
    "chunked_prefill_size",
    "max_running_requests",
    "context_length",
    "disable_radix_cache",
    "disable_cuda_graph",
    "cuda_graph_backend_decode",
    "cuda_graph_backend_prefill",
    "cuda_graph_bs_decode",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_num_draft_tokens",
    "speculative_eagle_topk",
    "speculative_draft_kv_cache_dtype",
    "random_seed",
    "disable_overlap_schedule",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def remove_flags(argv, flags):
    result = []
    i = 0
    while i < len(argv):
        key = argv[i].split("=", 1)[0]
        if key in flags:
            if "=" not in argv[i]:
                require(
                    i + 1 < len(argv) and not argv[i + 1].startswith("--"),
                    f"Missing value: {key}",
                )
                i += 1
        else:
            result.append(argv[i])
        i += 1
    return result


def value(argv, name):
    for i, arg in enumerate(argv):
        if arg == name:
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def prepare(args):
    source = args.run_sh.resolve()
    text = source.read_text()
    pattern = r"(?m)^(\s*)python3?\s+-m\s+sglang\.launch_server\b"
    require(
        len(re.findall(pattern, text)) == 1,
        "Expected one direct python[3] -m sglang.launch_server command in run.sh",
    )
    data = args.data.read_bytes()
    rows = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    require(len(rows) >= args.shots + args.count, "Dataset has too few rows")
    require(
        args.count > 0 and args.shots >= 0 and args.threads > 0,
        "Invalid evaluation settings",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "run.original.sh").write_text(text)
    replacement = 'python3 "$LC_PAIR_TOOL" launch --'
    wrapped = re.sub(pattern, lambda m: m[1] + replacement, text)
    (args.out / "run.wrapped.sh").write_text(wrapped)
    (args.out / "gsm8k_test.jsonl").write_bytes(data)
    write(
        args.out / "plan.json",
        {
            "original_run": str(source),
            "original_cwd": str(source.parent),
            "launcher_sha256": sha(text.encode()),
            "wrapped_sha256": sha(wrapped.encode()),
            "dataset_sha256": sha(data),
            "count": args.count,
            "shots": args.shots,
            "threads": args.threads,
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["Question", "Assistant:", "<|separator|>"],
            "api": "completion",
        },
    )
    print("PREPARED", args.out.resolve())


def start(args):
    root = args.pair.resolve()
    plan = read(root / "plan.json")
    require(
        sha((root / "run.wrapped.sh").read_bytes()) == plan["wrapped_sha256"],
        "Frozen launcher changed",
    )
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.run), "Invalid run name")
    model = args.model.resolve()
    require((model / "config.json").is_file(), "Missing model config.json")
    launch = root / "runs" / args.run / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        LC_PAIR_TOOL=str(Path(__file__).resolve()),
        LC_PAIR_ROOT=str(root),
        LC_PAIR_RUN=args.run,
        LC_PAIR_VARIANT=args.variant,
        LC_PAIR_MODEL=str(model),
        MODEL_PATH=str(model),
        LOG_DIR=str(launch / "serve_logs"),
    )
    os.chdir(plan["original_cwd"])
    os.execvpe("bash", ["bash", str(root / "run.wrapped.sh")], env)


def fingerprint():
    spec = importlib.util.find_spec("sglang")
    require(spec is not None and spec.origin, "Cannot locate actual SGLang source")
    source = Path(spec.origin).parent
    # Hash source bytes, including local changes/private runtimes, not only git HEAD.
    hashes = {
        str(p.relative_to(source)): sha(p.read_bytes())
        for p in sorted(source.rglob("*.py"))
    }
    dep = importlib.util.find_spec("deep_ep")
    require(dep is not None and dep.origin, "Cannot locate installed deep_ep")
    package = Path(dep.origin).parent
    binaries = set(package.rglob("*.so"))
    binaries.update(package.parent.glob("*deep*ep*.so"))
    require(
        binaries,
        "Cannot locate DeepEP extension; inspect installed package before comparing",
    )
    dep_files = sorted(set(package.rglob("*.py")) | binaries)
    dep_hashes = {
        str(p.relative_to(package.parent)): sha(p.read_bytes()) for p in dep_files
    }
    return {
        "sglang_path": str(source),
        "sglang_sha256": sha(json.dumps(hashes, sort_keys=True).encode()),
        "deepep_path": str(package),
        "deepep_files": dep_hashes,
        "deepep_sha256": sha(json.dumps(dep_hashes, sort_keys=True).encode()),
    }


def launch(args):
    root = Path(os.environ["LC_PAIR_ROOT"])
    variant = os.environ["LC_PAIR_VARIANT"]
    model = os.environ["LC_PAIR_MODEL"]
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    require(
        value(argv, "--speculative-algorithm") == "NEXTN", "Template must enable NEXTN"
    )
    argv = remove_flags(argv, MODEL_FLAGS)
    quant = "fp8" if variant == "fp8" else "compressed-tensors"
    argv += [
        "--model-path",
        model,
        "--tokenizer-path",
        model,
        "--quantization",
        quant,
        "--speculative-draft-model-path",
        model,
    ]
    if variant == "fp8":
        argv += ["--speculative-draft-model-quantization", "fp8"]
    for flag, expected in (
        ("--tp-size", "32"),
        ("--attn-dp-size", "2"),
        ("--ep-size", "32"),
        ("--moe-a2a-backend", "deepep"),
        ("--deepep-mode", "auto"),
    ):
        require(value(argv, flag) == expected, f"Template requires {flag} {expected}")
    require(
        os.environ.get("HCCL_OP_EXPANSION_MODE") == "AI_CPU",
        "Template must select AI_CPU",
    )
    require(
        "--disable-cuda-graph" not in argv
        and value(argv, "--cuda-graph-backend-decode") != "disabled",
        "Template disables decode graph",
    )
    require(
        value(argv, "--deepep-dispatcher-output-dtype") == "bf16",
        "Set --deepep-dispatcher-output-dtype bf16 in the shared template before prepare",
    )
    config = read(Path(model) / "config.json")
    require(
        config.get("num_layers", config.get("num_hidden_layers")) == 38,
        "Expected full 38-layer model",
    )
    rank = value(argv, "--node-rank")
    require(rank in ("0", "1", "2", "3"), "Expected node rank 0..3")
    receipt = root / "runs" / os.environ["LC_PAIR_RUN"] / "launch" / f"rank{rank}.json"
    metadata = {
        "variant": variant,
        "model": model,
        "hostname": socket.gethostname(),
        "node_rank": int(rank),
        "argv": argv,
        "common_argv": remove_flags(argv, MODEL_FLAGS),
        "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)},
        "config": config,
        "fingerprint": fingerprint(),
        "tokenizer_files": {
            p.name: sha(p.read_bytes())
            for p in sorted(Path(model).iterdir())
            if p.is_file()
            and (
                p.name.startswith("tokenizer")
                or p.name
                in (
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                )
            )
        },
    }
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(
        "PAIR_LAUNCH",
        shlex.join([sys.executable, "-m", "sglang.launch_server", *argv]),
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *argv])


def server_info(url):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(url.rstrip("/") + "/server_info", timeout=120) as response:
        return json.load(response)


def launches(run):
    items = [read(run / "launch" / f"rank{i}.json") for i in range(4)]
    for key in ("variant", "model"):
        require(len({x[key] for x in items}) == 1, f"Nodes disagree on {key}")
    require(
        all(x["tokenizer_files"] == items[0]["tokenizer_files"] for x in items),
        "Tokenizer differs across nodes",
    )
    for key in ("sglang_sha256", "deepep_sha256"):
        require(
            len({x["fingerprint"][key] for x in items}) == 1, f"Nodes disagree on {key}"
        )
    return items


def check_manifests(a, b):
    for key in ("plan", "evaluator_hashes", "policy"):
        require(
            a[key] == b[key],
            f"Comparison blocked: {key} differs; inspect manifest.json",
        )
    for x, y in zip(a["launches"], b["launches"]):
        for key in ("common_argv", "env", "tokenizer_files"):
            require(
                x[key] == y[key],
                f"Comparison blocked: rank{x['node_rank']} {key} differs",
            )
        for key in ("sglang_sha256", "deepep_sha256"):
            require(
                x["fingerprint"][key] == y["fingerprint"][key],
                f"Comparison blocked: {key}",
            )


def compare(args):
    left, right = args.left, args.right
    a, b = read(left / "manifest.json"), read(right / "manifest.json")
    for directory in (left, right):
        summary = read(directory / "summary.json")
        require(
            summary["complete"] and not summary["request_error_questions"],
            f"Incomplete run or request errors: {directory}",
        )
    check_manifests(a, b)

    def load_results(path):
        rows = [
            json.loads(s) for s in (path / "questions.jsonl").read_text().splitlines()
        ]
        result = {r["prompt_sha256"]: r for r in rows}
        require(
            len(result) == len(rows) == a["plan"]["count"],
            "Incomplete or duplicate results",
        )
        return result

    aa, bb = load_results(left), load_results(right)
    require(aa.keys() == bb.keys(), "Prompt sets differ")
    args.out.mkdir(parents=True, exist_ok=False)
    counts = {
        "both_correct": 0,
        "only_left_correct": 0,
        "only_right_correct": 0,
        "both_wrong": 0,
    }
    with (args.out / "paired.jsonl").open("w", encoding="utf-8") as f:
        for key, x in sorted(aa.items(), key=lambda kv: kv[1]["dataset_row"]):
            y = bb[key]
            require(x["gold"] == y["gold"], "Gold answers differ")
            group = (
                ("both_correct" if y["correct"] else "only_left_correct")
                if x["correct"]
                else ("only_right_correct" if y["correct"] else "both_wrong")
            )
            counts[group] += 1
            f.write(
                json.dumps(
                    {
                        "dataset_row": x["dataset_row"],
                        "prompt_sha256": key,
                        "group": group,
                        "left": x,
                        "right": y,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "left": str(left),
        "right": str(right),
        "counts": counts,
        "left_summary": read(left / "summary.json"),
        "right_summary": read(right / "summary.json"),
    }
    write(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-sh", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("start")
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--variant", choices=("int8", "fp8"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--run", required=True)
    p = sub.add_parser("launch")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("compare")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
