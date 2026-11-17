#!/usr/bin/env python3
"""Validate the frozen full-38 Eliza collector, without importing SGLang/NPU."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(runtime, topology):
    dp = 2 if topology == "a5-dp2" else 4
    receipt = json.loads((runtime / "receipt.json").read_text())
    assert receipt["mode"] == "trace"
    assert receipt["capture"] == "full_valid_boundaries"
    assert receipt["internal_attention"] is True, (
        "Rebuild with --with-internal-attention"
    )
    assert receipt["topology"] == {
        "tp_size": dp * 16,
        "attn_dp_size": dp,
        "moe_ep_size": dp * 16,
    }
    relative = "python/sglang/srt/debug_utils/longcat_capture_profile.json"
    profile = json.loads((runtime / relative).read_text())
    assert profile["schema"] == 5 and profile["num_layers"] == 38
    assert profile["layers"] == list(range(38)), "Require all blocks 0..37"
    assert profile["attn_tp_size"] == 16 and profile["attn_dp_size"] == dp
    assert profile["tp_size"] == profile["moe_ep_size"] == dp * 16
    required = {
        relative,
        "python/sglang/srt/debug_utils/longcat_fixed_decode_dp4.py",
        "python/sglang/srt/debug_utils/longcat_full_boundary.py",
        "python/sglang/srt/debug_utils/capture_internal_dp4.py",
    }
    assert required <= receipt["sha256"].keys()
    for name, expected in receipt["sha256"].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == expected, (
            name
        )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--topology", choices=("a3-dp4", "a5-dp2"), default="a5-dp2")
    args = parser.parse_args()
    result = validate(args.runtime, args.topology)
    print(
        "CAPTURE_RUNTIME_OK",
        json.dumps(
            {
                "topology": result["topology"],
                "layers": 38,
                "internal_attention": True,
                "revision": result["revision"],
                "receipt_sha256": hashlib.sha256(
                    (args.runtime / "receipt.json").read_bytes()
                ).hexdigest(),
            }
        ),
    )
