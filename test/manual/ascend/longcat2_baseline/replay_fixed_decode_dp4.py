#!/usr/bin/env python3
"""Pin each fixed-prefix group to attention DP0/DP1/DP2/DP3; not accuracy samples."""

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:10096")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--lengths", type=int, nargs="+", default=[1])
    p.add_argument("--dps", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--prefix-ids", default="70115,838,15200,11683,597,444,241,239")
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    ids = json.loads(args.control.read_text())["response"]["choices"][0][
        "prompt_token_ids"
    ]
    if (
        len(ids) != 852
        or hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12] != "07bd4107b296"
    ):
        raise ValueError("Control prompt must match the saved 852-token prompt")
    prefix = [int(x) for x in args.prefix_ids.split(",")]
    if args.repeat < 2 or any(n < 1 or n > len(prefix) for n in args.lengths):
        raise ValueError("Need repeat>=2 and valid prefix lengths")
    if len(set(args.lengths)) != len(args.lengths):
        raise ValueError("Duplicate prefix lengths")
    if (
        not args.dps
        or len(set(args.dps)) != len(args.dps)
        or any(d not in (0, 1, 2, 3) for d in args.dps)
    ):
        raise ValueError("--dps must select distinct DP0/DP1/DP2/DP3")
    args.out.mkdir(parents=True, exist_ok=False)
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    run = uuid.uuid4().hex[:12]
    manifest = {
        "run": run,
        "repeat": args.repeat,
        "schema": 4,
        "topology": {"tp_size": 64, "attn_dp_size": 4, "attn_tp_size": 16, "moe_ep_size": 64},
        "dps": args.dps,
        "lengths": args.lengths,
        "prompt_ids": ids,
        "prefix_ids": prefix,
        "requests": [],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for dp in args.dps:
        for length in args.lengths:
            prompt = ids + prefix[: length - 1]
            forced = prefix[length - 1]
            # One explicit warm-up per prefix, excluded from repeat comparisons.
            for rep in range(args.repeat + 1):
                rid = f"lcfix4-{run}-d{dp}-p{len(prompt)}-t{forced}-r{rep}"
                item = {
                    "rid": rid,
                    "dp": dp,
                    "length": length,
                    "rep": rep,
                    "prompt_len": len(prompt),
                    "forced_id": forced,
                    "warmup": rep == 0,
                    "status": "pending",
                }
                manifest["requests"].append(item)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                body = {
                    "rid": rid,
                    "routed_dp_rank": dp,
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 2,
                        "ignore_eos": True,
                    },
                    "stream": False,
                }
                (args.out / f"{rid}.request.json").write_text(json.dumps(body))
                request = urllib.request.Request(
                    args.url.rstrip("/") + "/generate",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                started = time.monotonic()
                try:
                    with client.open(request, timeout=args.timeout) as response:
                        result = json.load(response)
                    (args.out / f"{rid}.response.json").write_text(json.dumps(result))
                    output = result.get("output_ids")
                    if (
                        not isinstance(output, list)
                        or len(output) != 2
                        or output[0] != forced
                    ):
                        raise RuntimeError(
                            f"Forced output verification failed: {output}"
                        )
                    item.update(
                        status="ok",
                        output_ids=output,
                        seconds=round(time.monotonic() - started, 3),
                    )
                except Exception as exc:
                    item.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    print(json.dumps(item), flush=True)
                    raise SystemExit(
                        "Stopped: do not submit another request until the failed/timed-out request is resolved"
                    ) from exc
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(json.dumps(item), flush=True)
    print("MANIFEST", manifest_path)


if __name__ == "__main__":
    main()
