#!/usr/bin/env python3
"""Full-model TP64/ADP4/EP64 origin-row repeat comparison, CPU only."""

import argparse
import json
from pathlib import Path

import torch
from longcat_fixed_decode_dp4 import select_rows


def need(ok, message):
    if not ok:
        raise ValueError(message)


def inspect_tensor(value):
    if value is None:
        return None
    finite = torch.isfinite(value)
    return {
        "nonfinite": int((~finite).sum()),
        "allzero": bool(
            value.numel() and finite.all() and (not torch.count_nonzero(value))
        ),
    }


def difference(left, right):
    a, b = (left.flatten(), right.flatten())
    bad = (a != b).nonzero().flatten()
    if not bad.numel():
        return None
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = a[finite].double() - b[finite].double()
    first = int(bad[0])

    def scalar(v):
        return v.item() if torch.isfinite(v) else str(v.item())

    return {
        "different_elements": bad.numel(),
        "elements": a.numel(),
        "max_abs_finite": delta.abs().max().item() if delta.numel() else None,
        "relative_l2_finite": (
            delta.norm() / a[finite].double().norm().clamp_min(1e-30)
        ).item(),
        "first_flat_index": first,
        "baseline": scalar(a[first]),
        "actual": scalar(b[first]),
    }


def check_workers(root):
    for rank in range(64):
        path = root / f"worker.tp{rank}.json"
        need(
            path.exists(),
            f"Missing worker receipt {path}; merge files from all four hosts first",
        )
        w = json.loads(path.read_text())
        need(
            (
                w["tp_rank"],
                w["tp_size"],
                w["attn_dp_rank"],
                w["attn_tp_rank"],
                w["attn_dp_size"],
                w["attn_tp_size"],
                w["moe_ep_size"],
                w["attn_cp_size"],
                w["pp_size"],
            )
            == (rank, 64, rank // 16, rank % 16, 4, 16, 64, 1, 1),
            f"Wrong topology: {path}",
        )
    print("WORKERS all 64 TP workers report TP64/ADP4/ATP16/EP64/CP1/PP1")


def load(root, item, mode, rank, prompt, forced, layers):
    path = root / f"{item['rid']}.tp{rank}.{mode}.pt"
    need(path.exists(), f"Missing active-rank capture {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    dp = item["dp"]
    need(
        data["schema"] == 4
        and data["rid"] == item["rid"]
        and data["tp"] == rank
        and data["dp"] == dp
        and data["mode"] == mode,
        f"Identity mismatch {path}",
    )
    need(
        data["topology"]["attn_dp_rank"] == dp
        and data["topology"]["attn_tp_rank"] == rank % 16,
        f"Wrong origin DP group {path}",
    )
    ids, pos = (
        (prompt, list(range(len(prompt))))
        if mode == "EXTEND"
        else ([forced], [len(prompt)])
    )
    need(
        data["input_ids"] == ids and data["positions"] == pos,
        f"Logical input mismatch {path}",
    )
    need(
        data["num_layers"] == layers,
        f"Expected full {layers} layers, got {data['num_layers']}",
    )
    raw = [0] * 4
    raw[dp] = len(ids)
    need(data["raw_counts"] == raw, "Other DP requests overlapped the diagnostic")
    records = data["samples"]
    keys = [(s["layer"], s["point"]) for s in records]
    need(len(keys) == len(set(keys)), "Duplicate trace point")
    required = {
        (-1, p) for p in ("embedding.output", "final_norm.output", "lm_head.logits")
    }
    for layer in range(layers):
        required |= {
            (layer, p)
            for p in (
                "attn0.input",
                "attn0.local_output",
                "moe.input",
                "router.logits",
                "router.expert_ids",
                "router.weights",
                "moe.experts_output",
                "moe.reduced",
                "moe.branch_output",
                "ffn0.input",
                "ffn0.local_output",
                "attn1.input",
                "attn1.local_output",
                "ffn1.input",
                "ffn1.local_output",
                "layer.output",
            )
        }
    need(required <= set(keys), f"Missing boundaries {required - set(keys)} in {path}")
    need(
        {k[0] for k in keys if k[0] >= 0} == set(range(layers)),
        "Layer coverage mismatch",
    )
    for s in records:
        if s["point"] == "lm_head.logits":
            expected = ([0], [len(ids) - 1], 1, s["shape"][0] - 1)
        else:
            expected = select_rows(
                s["axes"], data["padded_counts"], dp, rank % 16, len(ids), s["shape"][0]
            )
        need(
            (s["physical_rows"], s["rows"], s["live_rows"], s["padding_rows"])
            == expected,
            f"Bad row mapping {path}: {s['layer']}/{s['point']}",
        )
        sampled = bool(s["rows"])
        need(
            s["status"] == ("sampled" if sampled else "no_live_rows"),
            "Incorrect row status",
        )
        need(
            (s["values"] is not None) == sampled,
            "Missing values or unexpected padding values",
        )
        if sampled:
            need(
                s["values"].shape[0] == len(s["rows"])
                and list(s["values"].shape[1:]) == s["shape"][1:],
                "Sample shape mismatch",
            )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    m = json.loads(args.manifest.read_text())
    need(m["schema"] == 4, "Use the DP4 replay script")
    cfg = json.loads((args.model / "config.json").read_text())
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    need(isinstance(layers, int) and layers > 2, "Expected full model config")
    need(
        all(x["status"] == "ok" for x in m["requests"]),
        "Incomplete/failed replay; inspect manifest",
    )
    need(m["repeat"] >= 2 and m["dps"] and len(set(m["dps"])) == len(m["dps"])
         and all(dp in range(4) for dp in m["dps"]), "Invalid DP4 manifest")
    check_workers(args.dump_dir)
    details = []
    cross = {}
    for dp in m["dps"]:
        for length in m["lengths"]:
            items = [
                x
                for x in m["requests"]
                if x["dp"] == dp and x["length"] == length and not x["warmup"]
            ]
            need(len(items) == m["repeat"], "Incomplete repeat group")
            prompt = m["prompt_ids"] + m["prefix_ids"][: length - 1]
            forced = m["prefix_ids"][length - 1]
            for mode in ("EXTEND", "DECODE"):
                changed = set()
                zeros = set()
                no_live = set()
                nonfinite = []
                first = None
                tops = {}
                for rank in range(dp * 16, (dp + 1) * 16):
                    baseline = None
                    for item in items:
                        data = load(
                            args.dump_dir, item, mode, rank, prompt, forced, layers
                        )
                        records = data["samples"]
                        if baseline is None:
                            baseline = data
                            if rank == dp * 16:
                                cross[dp, length, mode] = {
                                    s["point"]: s["values"]
                                    for s in records
                                    if s["point"]
                                    in ("final_norm.output", "lm_head.logits")
                                }
                        need(
                            data["padded_counts"] == baseline["padded_counts"],
                            "DP padding changed across repeats",
                        )
                        refs = baseline["samples"]
                        need(
                            [(s["layer"], s["point"]) for s in records]
                            == [(s["layer"], s["point"]) for s in refs],
                            "Trace boundary order changed",
                        )
                        for order, (ref, cur) in enumerate(zip(refs, records)):
                            key = (rank, cur["layer"], cur["point"])
                            for field in (
                                "shape",
                                "dtype",
                                "axes",
                                "physical_rows",
                                "rows",
                                "live_rows",
                                "padding_rows",
                                "status",
                            ):
                                need(
                                    ref[field] == cur[field],
                                    f"Incompatible mapping {key}: {field}",
                                )
                            a, b = ref["values"], cur["values"]
                            if b is None:
                                no_live.add(key)
                                continue
                            stats = inspect_tensor(b)
                            if stats["nonfinite"]:
                                nonfinite.append(
                                    {
                                        "rid": item["rid"],
                                        "stage": key,
                                        "count": stats["nonfinite"],
                                    }
                                )
                            if stats["allzero"]:
                                zeros.add(key)
                            if cur["point"] == "lm_head.logits":
                                tops.setdefault(str(rank), []).append(
                                    None
                                    if stats["nonfinite"]
                                    else int(b.flatten().argmax())
                                )
                            diff = difference(a, b)
                            if diff is not None:
                                changed.add(key)
                                event = {
                                    "dp": dp,
                                    "length": length,
                                    "mode": mode,
                                    "tp": rank,
                                    "layer": cur["layer"],
                                    "point": cur["point"],
                                    "logical_rows": cur["rows"],
                                    "axes": cur["axes"],
                                    "rid": item["rid"],
                                    "baseline_rid": items[0]["rid"],
                                    **diff,
                                }
                                details.append(event)
                                sortkey = (order, rank, item["rep"])
                                if first is None or sortkey < first[0]:
                                    first = (sortkey, event)
                summary = {
                    "dp": dp,
                    "length": length,
                    "mode": mode,
                    "layers": layers,
                    "repeats": len(items),
                    "active_tp_ranks": [dp * 16, (dp + 1) * 16 - 1],
                    "decode_position": len(prompt),
                    "changed_rank_stages": len(changed),
                    "nonfinite_records": len(nonfinite),
                    "allzero_live_rank_stages": len(zeros),
                    "no_live_rank_stages": len(no_live),
                    "logits_top1_by_rank": tops,
                }
                print("SUMMARY", json.dumps(summary))
                print(
                    "FIRST_OBSERVED_DIFFERENCE", json.dumps(first[1] if first else None)
                )
                if nonfinite:
                    print("FIRST_NONFINITE", json.dumps(nonfinite[0]))
                details.append(
                    {
                        "summary": summary,
                        "allzero_stages": sorted(zeros),
                        "no_live_stages": sorted(no_live),
                        "nonfinite": nonfinite,
                    }
                )
    reference_dp = m["dps"][0]
    for other_dp in m["dps"][1:]:
        for length in m["lengths"]:
            for mode in ("EXTEND", "DECODE"):
                for point in ("final_norm.output", "lm_head.logits"):
                    a = cross[reference_dp, length, mode][point]
                    b = cross[other_dp, length, mode][point]
                    need(a is not None and b is not None and a.shape == b.shape,
                         "Cross-DP final tensor mismatch")
                    event = {"reference_dp": reference_dp, "other_dp": other_dp,
                             "length": length, "mode": mode, "point": point,
                             "difference": difference(a, b)}
                    print("CROSS_DP_FINAL", json.dumps(event))
                    details.append({"cross_dp_final": event})
    args.out.write_text(json.dumps(details, indent=2, allow_nan=False))
    print("DETAILS", args.out)
    print(
        "NOTE: sampled live prefill rows and complete live decode row; inactive/padding rows excluded; remote expert internals not captured; exact differences are not an accuracy verdict"
    )


if __name__ == "__main__":
    main()
