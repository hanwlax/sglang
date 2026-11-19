#!/usr/bin/env python3
"""Run the installed legacy run_eval, recording Completion requests and responses."""

import argparse
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from pair import (
    POLICY,
    check_manifests,
    launches,
    read,
    require,
    server_info,
    sha,
    write,
)


def text_flags(text):
    return {
        "empty": not text.strip(),
        "text_zero_loop": bool(re.search(r"0{64,}", text)),
        "think_tags": text.count("<longcat_think>") + text.count("</longcat_think>"),
        # Completion text cannot establish whether raw output token IDs are zero.
        "token_zero_run": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--url", default="http://141.61.45.105:10086")
    parser.add_argument("--model", default="LongCat-2.0")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Prior evaluation directory; reject deployment drift before requests",
    )
    args = parser.parse_args()
    require(
        re.fullmatch(r"[A-Za-z0-9_-]+", args.run)
        and re.fullmatch(r"[A-Za-z0-9_-]+", args.eval_id),
        "Invalid run/eval ID",
    )
    root = args.pair.resolve()
    plan = read(root / "plan.json")
    data = root / "gsm8k_test.jsonl"
    require(sha(data.read_bytes()) == plan["dataset_sha256"], "Frozen dataset changed")
    receipts = launches(root / "runs" / args.run)
    out = root / "runs" / args.run / args.eval_id
    out.mkdir(parents=True, exist_ok=False)
    info = server_info(args.url)
    write(out / "server_info.json", info)
    for key, expected in {
        "tp_size": 32,
        "attn_dp_size": 2,
        "ep_size": 32,
        "moe_a2a_backend": "deepep",
        "deepep_mode": "auto",
        "deepep_dispatcher_output_dtype": "bf16",
        "speculative_algorithm": "NEXTN",
    }.items():
        require(
            info.get(key) == expected,
            f"Effective {key}: {info.get(key)!r}, expected {expected!r}",
        )
    require(
        info.get("model_path") == receipts[0]["model"],
        "Running model differs from launch receipt",
    )
    expected_quant = "fp8" if receipts[0]["variant"] == "fp8" else "compressed-tensors"
    require(
        info.get("quantization") == expected_quant, "Effective quantization mismatch"
    )
    require(
        not info.get("disable_cuda_graph")
        and info.get("cuda_graph_backend_decode") in ("full", "piecewise"),
        "Expected enabled decode graph; inspect server_info.json",
    )
    override = info.get("json_model_override_args") or "{}"
    override = json.loads(override) if isinstance(override, str) else override
    require(
        not {"num_layers", "num_hidden_layers"}.intersection(override),
        "Layer override is not allowed",
    )

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    # Preserve dataset downloading/proxy behavior elsewhere; only bypass this local API.
    from urllib.parse import urlparse

    host = urlparse(args.url).hostname
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key), host]))
    from sglang.test import run_eval as evaluator
    from sglang.test import simple_eval_common as common
    from sglang.test import simple_eval_mixed_prefix_gsm8k as gsm

    modules = (evaluator, common, gsm)
    source_hashes = {m.__name__: sha(Path(m.__file__).read_bytes()) for m in modules}
    for name in ("pair.py", "eval_pair.py"):
        source_hashes[name] = sha(Path(__file__).with_name(name).read_bytes())
    dataset = [
        json.loads(s)
        for s in data.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    prefix = gsm.get_few_shot_examples(dataset, plan["shots"])
    prompts = {}
    prompt_rows = []
    for index in range(plan["shots"], plan["shots"] + plan["count"]):
        prompt = prefix + gsm.get_one_example(dataset, index, False)
        entry = {
            "dataset_row": index,
            "question": dataset[index]["question"],
            "gold": gsm.get_answer_value(dataset[index]["answer"]),
            "prompt": prompt,
            "prompt_sha256": sha(prompt.encode()),
        }
        require(
            entry["prompt_sha256"] not in prompts,
            "Duplicate prompts in selected dataset",
        )
        prompts[entry["prompt_sha256"]] = entry
        prompt_rows.append(entry)
    write(out / "prompts.json", prompt_rows)
    write(
        out / "manifest.json",
        {
            "plan": plan,
            "evaluator_hashes": source_hashes,
            "policy": {key: info.get(key) for key in POLICY},
            "launches": receipts,
        },
    )
    if args.reference:
        check_manifests(
            read(args.reference / "manifest.json"), read(out / "manifest.json")
        )
    lock = threading.Lock()
    local = threading.local()
    records = []

    def append(filename, obj):
        with lock, (out / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    class RecordedSampler(common.CompletionSampler):
        def __init__(self, *pos, **kwargs):
            super().__init__(*pos, **kwargs)
            original = self.client.completions.create

            def create(**request):
                tic = time.monotonic()
                try:
                    response = original(**request)
                    raw = response.model_dump(mode="json")
                    local.response = raw
                    append(
                        "attempts.jsonl",
                        {
                            "prompt_sha256": sha(request["prompt"].encode()),
                            "request": request,
                            "response": raw,
                            "seconds": time.monotonic() - tic,
                            "error": None,
                        },
                    )
                    return response
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    local.errors.append(error)
                    append(
                        "attempts.jsonl",
                        {
                            "prompt_sha256": sha(request["prompt"].encode()),
                            "request": request,
                            "response": None,
                            "seconds": time.monotonic() - tic,
                            "error": error,
                        },
                    )
                    raise

            self.client.completions.create = create

        def __call__(self, messages):
            prompt = "\n".join(
                m["content"] for m in messages if isinstance(m.get("content"), str)
            )
            key = sha(prompt.encode())
            require(key in prompts, "run_eval produced an unexpected prompt")
            local.response, local.errors = None, []
            tic = time.monotonic()
            text = ""
            try:
                text = super().__call__(messages)
                return text
            finally:
                raw = local.response or {}
                choice = (raw.get("choices") or [{}])[0]
                parsed = gsm.get_answer_value(text)
                row = dict(
                    prompts[key],
                    response=text,
                    parsed=parsed,
                    correct=parsed == prompts[key]["gold"],
                    finish_reason=choice.get("finish_reason"),
                    usage=raw.get("usage"),
                    response_id=raw.get("id"),
                    errors=local.errors,
                    seconds=time.monotonic() - tic,
                    **text_flags(text),
                )
                append("questions.jsonl", row)
                with lock:
                    records.append(row)
                print(
                    "QUESTION",
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "dataset_row",
                                "correct",
                                "parsed",
                                "gold",
                                "finish_reason",
                                "empty",
                                "text_zero_loop",
                            )
                        }
                    ),
                    flush=True,
                )

    # Only add observation hooks and relocate reports; keep the installed scorer,
    # prompt construction, HTTP sampling parameters and retry policy unchanged.
    common.CompletionSampler = RecordedSampler
    source = inspect.getsource(evaluator)
    for suffix in ("html", "json"):
        old = f'f"/tmp/{{file_stem}}.{suffix}"'
        require(
            source.count(old) == 1,
            "Unsupported run_eval report layout; no evaluation was started",
        )
        source = source.replace(old, f'str(__pair_out / f"{{file_stem}}.{suffix}")')
    (out / "run_eval_recorded.py").write_text(source, encoding="utf-8")
    command = [
        "run_eval_recorded.py",
        "--base-url",
        args.url.rstrip("/"),
        "--model",
        args.model,
        "--eval-name",
        "gsm8k",
        "--api",
        "completion",
        "--num-examples",
        str(plan["count"]),
        "--num-shots",
        str(plan["shots"]),
        "--num-threads",
        str(plan["threads"]),
        "--max-tokens",
        str(plan["max_tokens"]),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--repeat",
        "1",
        "--gsm8k-data-path",
        str(data),
    ]
    write(out / "eval_argv.json", command)
    sys.argv = command
    try:
        exec(  # noqa: S102 - executes the user's installed evaluator with relocated report paths
            compile(source, str(out / "run_eval_recorded.py"), "exec"),
            {"__name__": "__main__", "__file__": evaluator.__file__, "__pair_out": out},
        )
    finally:
        summary = {
            "responses": len(records),
            "expected": plan["count"],
            "correct": sum(r["correct"] for r in records),
            "score": sum(r["correct"] for r in records) / plan["count"],
            "complete": len(records) == plan["count"],
            "request_error_questions": sum(bool(r["errors"]) for r in records),
            "empty": sum(r["empty"] for r in records),
            "text_zero_loop": sum(r["text_zero_loop"] for r in records),
            "finish_length": sum(r["finish_reason"] == "length" for r in records),
        }
        write(out / "summary.json", summary)
        print("SUMMARY", json.dumps(summary), flush=True)
    require(summary["complete"], "Partial evaluation; see saved results")
    official = read(out / f"gsm8k_{args.model.replace('/', '_')}.json")
    require(
        abs(official["score"] - summary["score"]) < 1e-9,
        "Recorded score disagrees with run_eval",
    )
    require(
        not summary["request_error_questions"],
        "Request errors occurred; run is saved but not a clean baseline",
    )


if __name__ == "__main__":
    main()
