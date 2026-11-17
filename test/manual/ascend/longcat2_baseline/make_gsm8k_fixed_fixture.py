#!/usr/bin/env python3
"""Freeze an original GSM8K 5-shot prompt and an explicit decode prefix as IDs."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--question-prefix", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--prefix-text", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from sglang.srt.hf_transformers.tokenizer import get_tokenizer
    from sglang.test.simple_eval_mixed_prefix_gsm8k import (
        get_few_shot_examples,
        get_one_example,
    )

    lines = [
        json.loads(x)
        for x in a.dataset.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    indices = [
        i
        for i, row in enumerate(lines)
        if row["question"].startswith(a.question_prefix)
    ]
    if len(indices) != 1 or indices[0] < 5:
        raise ValueError("Need exactly one non-shot question")
    row = indices[0]
    prompt = get_few_shot_examples(lines, 5) + get_one_example(lines, row, False)
    tokenizer = get_tokenizer(str(a.model), trust_remote_code=True)
    ids = tokenizer.encode(prompt)
    prefix = tokenizer.encode(a.prefix_text, add_special_tokens=False)
    if not ids or not prefix:
        raise ValueError("Prompt and prefix must be nonempty")
    if tokenizer.decode(prefix, skip_special_tokens=False) != a.prefix_text:
        raise ValueError("Prefix does not roundtrip exactly")
    tokenizer_files = {
        f.name: digest(f.read_bytes())
        for f in sorted(a.model.glob("*token*"))
        if f.is_file()
    }
    result = {
        "schema": "longcat_fixed_input_v1",
        "name": a.name,
        "prompt_text": prompt,
        "prompt_ids": ids,
        "prompt_ids_sha256": digest(json.dumps(ids).encode()),
        "prefix_text": a.prefix_text,
        "prefix_ids": prefix,
        "prefix_ids_sha256": digest(json.dumps(prefix).encode()),
        "gold_answer": lines[row]["answer"],
        "provenance": {
            "dataset_sha256": digest(a.dataset.read_bytes()),
            "dataset_row": row,
            "few_shot_rows": list(range(5)),
            "model_path": str(a.model),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_loader": "sglang.srt.hf_transformers.tokenizer.get_tokenizer",
            "tokenizer_files_sha256": tokenizer_files,
            "prompt_builder": "sglang.test.simple_eval_mixed_prefix_gsm8k: 5 shots + unanswered question",
            "prefix_source": "explicit text for aligned comparison; not certified as original failing response token IDs",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "file": str(a.out),
                "name": a.name,
                "row": row,
                "prompt_length": len(ids),
                "prompt_ids_sha256": result["prompt_ids_sha256"],
                "prefix_ids": prefix,
                "prefix_length": len(prefix),
                "prefix_text": a.prefix_text,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
