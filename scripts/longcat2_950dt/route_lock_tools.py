#!/usr/bin/env python3
"""Pack full TP8 A3 routes, replay one prefill, or audit applied routes; stdlib only."""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_run(folder):
    records = {}
    for layer in range(2):
        for tp in range(8):
            path = Path(folder) / f"route_L{layer}_TP{tp}.json"
            r = json.loads(path.read_text())
            require(
                (r["schema"], r["layer"], r["tp"]) == (1, layer, tp),
                f"bad header {path}",
            )
            require(len(r["ids"]) == len(r["weights"]) == 852, f"bad rows {path}")
            require(
                all(
                    len(ids) == len(weights) == 12
                    and len(set(ids)) == 12
                    and all(type(i) is int and 0 <= i < 896 for i in ids)
                    and all(math.isfinite(w) and w >= 0 for w in weights)
                    for ids, weights in zip(r["ids"], r["weights"])
                ),
                f"bad routing {path}",
            )
            records[layer, tp] = r
    base = records[0, 0]
    for (layer, tp), r in records.items():
        require(
            r["identity"] == base["identity"]
            and r["meta"] == base["meta"]
            and r["mode"] == base["mode"]
            and r["reference_sha256"] == base["reference_sha256"],
            f"mixed experiment metadata L{layer} TP{tp}",
        )
        require(
            r["ids"] == records[layer, 0]["ids"],
            f"different applied expert IDs across ranks L{layer} TP{tp}",
        )
        require(
            r["weights"] == records[layer, 0]["weights"],
            f"different applied weights across ranks L{layer} TP{tp}",
        )
    identity = base["identity"]
    payload = json.dumps(
        [identity["input_ids"], identity["positions"]], separators=(",", ":")
    )
    require(
        hashlib.sha256(payload.encode()).hexdigest()[:16]
        == identity["input_hash"]
        == "fd85b1782b833ab4",
        "wrong fixed input",
    )
    require(identity["positions"] == list(range(852)), "wrong positions")
    return records


def pack(args):
    records = load_run(args.routes)
    base = records[0, 0]
    require(base["mode"] == "capture", "reference must come from natural A3 capture")
    ref = {
        "schema": 1,
        "kind": "longcat_route_reference",
        "identity": base["identity"],
        "meta": base["meta"],
        "layers": {
            str(layer): {k: records[layer, 0][k] for k in ("ids", "weights")}
            for layer in range(2)
        },
    }
    with Path(args.out).open("x") as f:
        json.dump(ref, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    print(
        "PACK_OK",
        json.dumps(
            {
                "ranks": 8,
                "layers": 2,
                "tokens": 852,
                "file": args.out,
                "sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
            }
        ),
    )


def audit(args):
    records = load_run(args.routes)
    ref_raw = Path(args.reference).read_bytes()
    ref = json.loads(ref_raw)
    sha = hashlib.sha256(ref_raw).hexdigest()
    for (layer, tp), r in records.items():
        require(
            r["identity"] == ref["identity"] and r["meta"] == ref["meta"],
            "reference metadata mismatch",
        )
        require(
            r["mode"] in ("ids", "ids_weights") and r["reference_sha256"] == sha,
            "not a forced run of this reference",
        )
        require(
            r["ids"] == ref["layers"][str(layer)]["ids"],
            f"forced ID mismatch L{layer} TP{tp}",
        )
        if r["mode"] == "ids_weights":
            require(
                r["weights"] == ref["layers"][str(layer)]["weights"],
                f"forced weight mismatch L{layer} TP{tp}",
            )
    print(
        "AUDIT_OK",
        json.dumps(
            {
                "mode": records[0, 0]["mode"],
                "records": 16,
                "reference_sha256": sha,
                "note": "route equality only, not an accuracy verdict",
            }
        ),
    )


def replay(args):
    saved = json.loads(Path(args.control).read_text())
    ids = saved["response"]["choices"][0]["prompt_token_ids"]
    require(
        len(ids) == 852
        and hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12] == "07bd4107b296",
        "expected original 852-token Control prompt",
    )
    log = Path(args.log)
    before = log.stat()
    offset = before.st_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    request = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
    }
    (out / "request.json").write_text(json.dumps(request) + "\n")
    req = urllib.request.Request(
        args.url.rstrip("/") + "/generate",
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
    )
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with client.open(req, timeout=args.timeout) as response:
            body = response.read()
        (out / "response.json").write_bytes(body)
    finally:
        # The HTTP reply can arrive before tee drains the worker log pipes.
        deadline = time.monotonic() + 5
        while True:
            after = log.stat()
            require(
                after.st_ino == before.st_ino and after.st_size >= offset,
                "service log rotated/truncated",
            )
            with log.open("rb") as f:
                f.seek(offset)
                excerpt = f.read().decode(errors="replace")
            records, locks, full = [], [], []
            for line in excerpt.splitlines(keepends=True):
                if not line.endswith("\n"):
                    continue
                if "LONGCAT_ACT " in line:
                    r = json.loads(line.split("LONGCAT_ACT ", 1)[1])
                    if r["mode"] == "EXTEND" and r["input_hash"] == "fd85b1782b833ab4":
                        records.append(r)
                if "LONGCAT_ROUTE_LOCK " in line:
                    locks.append(json.loads(line.split("LONGCAT_ROUTE_LOCK ", 1)[1]))
                if "LONGCAT_FULL_TENSOR " in line:
                    full.append(json.loads(line.split("LONGCAT_FULL_TENSOR ", 1)[1]))
            if (
                len(records) >= 416 and len(locks) >= 16
            ) or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        (out / "service_excerpt.log").write_text(excerpt)
        (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (out / "route_events.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in locks)
        )
        (out / "full_events.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in full)
        )
        print(
            "SAVED",
            out,
            "trace_records",
            len(records),
            "route_records",
            len(locks),
            "full_tensors",
            len(full),
        )
    require(
        len(records) == 416
        and len({(r["layer"], r["point"], r["tp"]) for r in records}) == 416,
        "expected 416 unique trace records; inspect service_excerpt.log",
    )
    require(
        len(locks) == 16
        and {(r["layer"], r["tp"]) for r in locks}
        == {(l, t) for l in range(2) for t in range(8)},
        "expected 16 routing receipts; probe may not have activated",
    )
    require(
        all(
            r["input_hash"] == "fd85b1782b833ab4" and r["tokens"] == 852 for r in locks
        ),
        "wrong routing receipts",
    )
    if args.require_full:
        require(
            len(full) == 416
            and {(r["layer"], r["point"], r["tp"]) for r in full}
            == {(r["layer"], r["point"], r["tp"]) for r in records},
            "full tensor coverage differs from trace",
        )
    print("REPLAY_OK", "prefill only; generated token is not fed back into decode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--routes", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=pack)
    p = sub.add_parser("audit")
    p.add_argument("--routes", required=True)
    p.add_argument("--reference", required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser("replay")
    p.add_argument("--control", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:10086")
    p.add_argument("--timeout", type=int, default=3000)
    p.add_argument("--require-full", action="store_true")
    p.set_defaults(func=replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
