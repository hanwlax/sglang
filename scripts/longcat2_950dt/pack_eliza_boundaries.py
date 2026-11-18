"""Lossless A3 Eliza boundary packs from a previously verified 38-layer archive.

First pack: every DECODE boundary and EXTEND layer.output/residual and logits.
Supplemental packs: remaining EXTEND boundaries at one layer (-1 for globals).
Original per-record semantics and row maps survive content deduplication.
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import torch

BACKBONE = {"layer.output", "layer.residual", "lm_head.logits"}


def selected(sample, mode, layer):
    if layer is None:
        return mode == "DECODE" or sample["point"] in BACKBONE
    return (
        mode == "EXTEND"
        and sample["layer"] == layer
        and sample["point"] not in BACKBONE
    )


def ensure(ok, reason):
    if not ok:
        raise ValueError(reason)


def pack(dump, out, layer):
    paths = sorted(dump.glob("*.EXTEND.json")) + sorted(dump.glob("*.DECODE.json"))
    ensure(len(paths) == 128, "expected 128 completed A3 Eliza indices")
    keys, seen = set(), set()
    logical, live, total = 0, 0, 0
    with zipfile.ZipFile(out, "x", allowZip64=True) as archive:
        for path in paths:
            idx = json.loads(path.read_bytes())
            rid, mode, tp = idx["rid"], idx["mode"], idx["tp"]
            m = re.fullmatch(r"lcfix4-([0-9a-f]+)-d0-p(853|863)-t(\d+)-r([12])", rid)
            ensure(
                m
                and path.name == f"{rid}.tp{tp}.{mode}.json"
                and idx["num_layers"] == 38
                and idx["selected_layers"] == list(range(38))
                and idx["schema"] == 5
                and idx["dp"] == 0
                and idx["topology"]["tp_size"] == 64
                and idx["topology"]["attn_tp_size"] == 16
                and tp in range(16),
                f"incompatible A3 index {path}",
            )
            ensure(
                (m[2], m[3]) in (("853", "78261"), ("863", "444")),
                "unexpected forced ID",
            )
            key = (m[1], m[2], m[4], tp, mode)
            ensure(key not in keys, "duplicate request/rank/mode")
            keys.add(key)
            samples = [s for s in idx["samples"] if selected(s, mode, layer)]
            idx["samples"] = samples
            for s in samples:
                logical += 1
                if s["values_file"] is None:
                    ensure(s["elements"] == 0, "missing live payload")
                    continue
                live += 1
                sha = s["tensor_sha256"]
                name = (
                    f"values/{s['dtype']}/{'x'.join(map(str, s['value_shape']))}/{sha}"
                )
                s["pack_member"] = name
                if name in seen:
                    continue
                p = (dump / s["values_file"]).resolve()
                ensure(dump.resolve() in p.parents, "payload outside dump root")
                d = torch.load(p, weights_only=True, map_location="cpu")
                ensure(
                    (
                        d["rid"],
                        d["mode"],
                        d["layer"],
                        d["point"],
                        d["topology"]["tp_rank"],
                    )
                    == (rid, mode, s["layer"], s["point"], tp),
                    f"bad payload header {p}",
                )
                v = d["values"].contiguous()
                raw = v.view(torch.uint8).numpy().tobytes()
                ensure(
                    str(v.dtype) == s["dtype"]
                    and list(v.shape) == s["value_shape"]
                    and v.numel() == s["elements"]
                    and hashlib.sha256(raw).hexdigest() == sha,
                    f"bad payload values {p}",
                )
                archive.writestr(name, raw)
                seen.add(name)
                total += len(raw)
            archive.writestr(
                "indices/" + path.name,
                json.dumps(idx, separators=(",", ":")),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        ensure(
            len({k[0] for k in keys}) == 1 and len(keys) == 128 and live > 0,
            "incomplete archive",
        )
        result = {
            "schema": 1,
            "scope": "first" if layer is None else f"extend_remainder_L{layer}",
            "indices": len(keys),
            "logical_records": logical,
            "live_records": live,
            "unique_payloads": len(seen),
            "unique_bytes": total,
            "internal_attention_included": False,
        }
        archive.writestr("complete.json", json.dumps(result))
    print("PACK", json.dumps(result), str(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new ZIP archive")
    p.add_argument("--extend-remainder-layer", type=int, choices=range(-1, 38))
    args = p.parse_args()
    torch.set_num_threads(1)
    pack(args.dump, args.out, args.extend_remainder_layer)
