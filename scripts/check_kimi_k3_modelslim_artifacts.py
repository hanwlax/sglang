#!/usr/bin/env python3
"""Static checks for Kimi-K3 ModelSlim artifacts.

This script intentionally avoids importing torch, transformers, or sglang so it
can run on a login/build host before NPU runtime smoke tests are possible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VISION_PREFIXES = (
    "vision_tower.",
    "model.vision_tower.",
    "mm_projector.",
    "model.mm_projector.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_vision_weight(name: str) -> bool:
    return name.startswith(VISION_PREFIXES)


def normalize_packed_key(name: str) -> str:
    return name.removesuffix("_packed") if name.endswith("_packed") else name


def describe_counter(counter: Counter[str], limit: int = 8) -> str:
    return ", ".join(f"{name}={count}" for name, count in counter.most_common(limit))


def check_config(model_dir: Path, errors: list[str], warnings: list[str]) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        errors.append("missing config.json")
        return

    config = load_json(config_path)
    text_config = config.get("text_config", {})

    if config.get("model_type") != "kimi_k3":
        errors.append(f"config.model_type must be kimi_k3, got {config.get('model_type')!r}")
    architectures = config.get("architectures") or []
    if "KimiK3ForConditionalGeneration" not in architectures:
        errors.append(
            "config.architectures must include KimiK3ForConditionalGeneration"
        )
    if text_config.get("model_type") != "kimi_linear":
        errors.append(
            "config.text_config.model_type must be kimi_linear, "
            f"got {text_config.get('model_type')!r}"
        )
    if config.get("quantization_config"):
        warnings.append(
            "top-level config.quantization_config is present; "
            "quant_model_description.json should win for --quantization modelslim"
        )
    if text_config.get("quantization_config"):
        warnings.append(
            "text_config.quantization_config is present; verify it is stale metadata "
            "and that quant_model_description.json is used for ModelSlim"
        )


def check_tokenizer_artifacts(
    model_dir: Path, errors: list[str], warnings: list[str]
) -> None:
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        errors.append("missing tokenizer_config.json")
        return

    tokenizer_config = load_json(tokenizer_config_path)
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if tokenizer_class != "TikTokenTokenizer":
        warnings.append(
            "tokenizer_config.tokenizer_class is not TikTokenTokenizer: "
            f"{tokenizer_class!r}"
        )

    auto_map = tokenizer_config.get("auto_map", {})
    auto_tokenizer = auto_map.get("AutoTokenizer") if isinstance(auto_map, dict) else None
    if not isinstance(auto_tokenizer, list) or not auto_tokenizer:
        warnings.append("tokenizer_config.auto_map.AutoTokenizer is missing or empty")
    else:
        tokenizer_impl = auto_tokenizer[0]
        if isinstance(tokenizer_impl, str) and "." in tokenizer_impl:
            warnings.append(
                "tokenizer uses local auto_map code; launch SGLang with "
                "--trust-remote-code unless tokenizer initialization is skipped"
            )
            module_name = tokenizer_impl.split(".", 1)[0]
            module_path = model_dir / f"{module_name}.py"
            if not module_path.exists():
                errors.append(
                    "tokenizer auto_map points to missing local module: "
                    f"{module_path.name}"
                )
        else:
            warnings.append(
                "tokenizer_config.auto_map.AutoTokenizer[0] has unexpected value: "
                f"{tokenizer_impl!r}"
            )

    for required_file in ("tiktoken.model",):
        if not (model_dir / required_file).exists():
            errors.append(f"missing tokenizer artifact {required_file}")

    for token_name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        if not tokenizer_config.get(token_name):
            warnings.append(f"tokenizer_config.{token_name} is missing")


def collect_weight_index(
    model_dir: Path, errors: list[str]
) -> tuple[list[str], list[str], Counter[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        errors.append("missing model.safetensors.index.json")
        return [], [], Counter()

    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        errors.append("model.safetensors.index.json must contain a weight_map object")
        return [], [], Counter()

    names = list(weight_map)
    language_names = [name for name in names if not is_vision_weight(name)]
    vision_names = [name for name in names if is_vision_weight(name)]
    suffixes = Counter(name.rsplit(".", 1)[-1] for name in language_names)
    return language_names, vision_names, suffixes


def check_weight_index(
    language_names: list[str],
    vision_names: list[str],
    suffixes: Counter[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not language_names:
        errors.append("no language_model weights found in safetensors index")
    if not vision_names:
        warnings.append("no vision/projector weights found; verify this is a text-only export")

    if suffixes.get("weight_packed", 0) == 0:
        warnings.append("no language *.weight_packed tensors found")
    if suffixes.get("weight_scale", 0) == 0:
        warnings.append("no language *.weight_scale tensors found")

    required_moe_suffixes = (
        ".block_sparse_moe.experts.0.w1.weight_packed",
        ".block_sparse_moe.experts.0.w2.weight_packed",
        ".block_sparse_moe.experts.0.w3.weight_packed",
    )
    missing_moe_samples = [
        suffix
        for suffix in required_moe_suffixes
        if not any(name.endswith(suffix) for name in language_names)
    ]
    if missing_moe_samples:
        warnings.append(
            "could not find representative Kimi MoE packed tensors: "
            + ", ".join(missing_moe_samples)
        )


def check_quant_description(
    model_dir: Path,
    language_names: list[str],
    allow_missing_quant: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    quant_path = model_dir / "quant_model_description.json"
    if not quant_path.exists():
        message = "missing quant_model_description.json"
        if allow_missing_quant:
            warnings.append(message)
        else:
            errors.append(message)
        return

    quant = load_json(quant_path)
    if not isinstance(quant, dict):
        errors.append("quant_model_description.json must be a JSON object")
        return

    quant_keys = {key for key in quant if isinstance(key, str)}
    normalized_quant_keys = {normalize_packed_key(key) for key in quant_keys}
    quant_weight_keys = {
        key
        for key in normalized_quant_keys
        if key.endswith(".weight") or key.endswith(".weight_scale")
    }

    if not quant_weight_keys:
        errors.append(
            "quant_model_description.json has no *.weight/*.weight_packed entries"
        )

    packed_weight_keys = [
        normalize_packed_key(name)
        for name in language_names
        if name.endswith(".weight_packed")
    ]
    if packed_weight_keys:
        covered = sum(1 for key in packed_weight_keys if key in normalized_quant_keys)
        coverage = covered / len(packed_weight_keys)
        if coverage == 0:
            warnings.append(
                "no checkpoint *.weight_packed keys are directly covered by "
                "quant_model_description.json after _packed normalization"
            )
        elif coverage < 0.95:
            warnings.append(
                "partial checkpoint packed-weight coverage in quant description: "
                f"{covered}/{len(packed_weight_keys)} ({coverage:.1%})"
            )

    vision_quant_keys = [key for key in quant_keys if is_vision_weight(key)]
    if vision_quant_keys:
        warnings.append(
            "quant_model_description.json contains vision/projector keys; "
            "Kimi-K3 SGLang path currently loads language weights only by default"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Kimi-K3 ModelSlim artifacts without loading the model."
    )
    parser.add_argument("model_dir", type=Path, help="Path to Kimi-K3 artifact dir")
    parser.add_argument(
        "--allow-missing-quant",
        action="store_true",
        help="Warn instead of failing when quant_model_description.json is absent.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not model_dir.is_dir():
        errors.append(f"model_dir is not a directory: {model_dir}")
    else:
        check_config(model_dir, errors, warnings)
        check_tokenizer_artifacts(model_dir, errors, warnings)
        language_names, vision_names, suffixes = collect_weight_index(model_dir, errors)
        check_weight_index(language_names, vision_names, suffixes, errors, warnings)
        check_quant_description(
            model_dir,
            language_names,
            args.allow_missing_quant,
            errors,
            warnings,
        )

        print(f"model_dir: {model_dir}")
        print(f"language weights: {len(language_names)}")
        print(f"vision/projector weights: {len(vision_names)}")
        if suffixes:
            print(f"language suffixes: {describe_counter(suffixes)}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
