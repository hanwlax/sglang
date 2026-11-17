"""Temporary fixed-prefix probe for c0e9d9a0: TP64/attention-DP4/EP64.

Captures request-origin boundaries on its 16 Attention TP ranks. DeepEP runs
on all 64 ranks; remote expert-internal tensors are NOT captured here.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import torch

_active = None
_worker_written = False
_PATTERN = re.compile(r"lcfix4-([0-9a-f]{12})-d([0-3])-p(\d+)-t(\d+)-r(\d+)")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def topology():
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    fields = (
        "tp_rank",
        "tp_size",
        "attn_dp_rank",
        "attn_dp_size",
        "attn_tp_rank",
        "attn_tp_size",
        "attn_cp_size",
        "moe_ep_rank",
        "moe_ep_size",
        "moe_tp_size",
        "moe_dp_size",
        "pp_size",
    )
    result = {k: int(getattr(p, k)) for k in fields}
    need(
        (
            p.tp_size,
            p.attn_dp_size,
            p.attn_tp_size,
            p.moe_ep_size,
            p.attn_cp_size,
            p.pp_size,
        )
        == (64, 4, 16, 64, 1, 1),
        "Expected TP64/ADP4/ATP16/EP64/CP1/PP1",
    )
    need(p.moe_tp_size == p.moe_dp_size == 1, "Expected full-width experts with MoE TP1/DP1")
    need(p.tp_rank == p.attn_dp_rank * 16 + p.attn_tp_rank, "Unexpected rank mapping")
    return result


def request(batch):
    reqs = getattr(batch, "reqs", ()) if batch is not None else ()
    tagged = [r for r in reqs if str(r.rid).startswith("lcfix4-")]
    if not tagged:
        return None
    need(len(reqs) == len(tagged) == 1, "Probe requires one isolated request")
    rid = str(tagged[0].rid)
    m = _PATTERN.fullmatch(rid)
    need(m is not None, "Invalid lcfix4 RID")
    return {
        "rid": rid,
        "dp": int(m[2]),
        "prompt_len": int(m[3]),
        "forced_id": int(m[4]),
    }


def bind_fixed_decode(batch, forward_batch, *, speculative):
    global _active, _worker_written
    _active = request(batch)
    directory = os.environ.get("LONGCAT_FIXED_DECODE_DIR")
    if directory and not _worker_written:
        info = topology()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"worker.tp{info['tp_rank']}.json").write_text(json.dumps(info))
        print("LONGCAT_FIXED_V4_WORKER " + json.dumps(info), flush=True)
        _worker_written = True
    if _active is None:
        return
    need(bool(directory), "LONGCAT_FIXED_DECODE_DIR is required")
    info = topology()
    need(info["attn_dp_rank"] == _active["dp"], "Request reached wrong DP group")
    need(
        not speculative and forward_batch.forward_mode.name in ("EXTEND", "DECODE"),
        "Disable NEXTN; expected ordinary EXTEND/DECODE",
    )
    need(forward_batch.batch_size == 1, "Concurrent requests are unsupported")
    mode = forward_batch.forward_mode.name
    n = _active["prompt_len"] if mode == "EXTEND" else 1
    raw_counts = [int(x) for x in forward_batch.global_num_tokens_cpu]
    expected = [0] * 4
    expected[_active["dp"]] = n
    need(
        raw_counts == expected,
        f"Expected isolated raw DP counts {expected}, got {raw_counts}",
    )
    _active.update(mode=mode, saved=False, raw_counts=raw_counts, topology=info)


def select_rows(axes, counts, dp, local_rank, real_n, physical_n):
    """Map contiguous row domains used by layer_boundary.ops.move_rows.

    Return physical indices, logical request indices, real-row count and pad count.
    Only DP-local, global DP-gathered, and DP-local ATP-split layouts are supported.
    """
    need(len(counts) == 4 and 0 <= dp < 4 and 0 <= local_rank < 16, "Invalid DP4 row metadata")
    axes = frozenset(axes)
    need(
        axes
        in (frozenset(), frozenset({"ATTN_DP"}), frozenset({"ATTN_DP", "ATTN_TP"})),
        f"Unsupported row axes {axes}",
    )
    local_n = counts[dp]
    if "ATTN_DP" not in axes:
        expected_n = sum(counts)
        base = sum(counts[:dp])
        start, end = 0, real_n
    elif "ATTN_TP" in axes:
        q, r = divmod(local_n, 16)
        shard_start = local_rank * q + min(local_rank, r)
        expected_n = q + int(local_rank < r)
        start, end = shard_start, min(shard_start + expected_n, real_n)
        base = -shard_start
    else:
        expected_n = local_n
        start, end, base = 0, real_n, 0
        # Some final/logit-adjacent consumers have already removed DP padding.
        if physical_n == real_n:
            expected_n = real_n
    need(
        physical_n == expected_n,
        f"Row shape mismatch: axes={sorted(axes)} expected={expected_n} actual={physical_n}",
    )
    live = max(0, end - start)
    logical = [] if not live else sorted({start, start + live // 2, end - 1})
    return [base + i for i in logical], logical, live, physical_n - live


class FixedTrace:
    def __init__(self, input_ids, positions, batch, model):
        self.context = dict(_active)
        self.batch, self.model = batch, model
        self.mode = self.context["mode"]
        self.n = self.context["prompt_len"] if self.mode == "EXTEND" else 1
        need(
            input_ids.ndim == positions.ndim == 1 and input_ids.numel() >= self.n,
            "Unexpected model input layout",
        )
        self.ids = input_ids[: self.n].detach().clone()
        self.positions = positions[: self.n].detach().clone()
        self.counts = [int(x) for x in batch.global_num_tokens_cpu]
        dp = self.context["dp"]
        need(
            len(self.counts) == 4 and self.counts[dp] >= self.n,
            "Invalid padded DP counts",
        )
        self.samples = []
        self.num_layers = len(model.layers)

    def layout(self, layer, point):
        if layer == -1:
            return {"ATTN_DP"}
        decoder = self.model.layers[layer]
        if point.startswith(("moe.", "router.", "layer.")):
            plan = decoder.moe_boundary.plan
        elif point.startswith("attn0."):
            plan = decoder.attn_boundary.plan
        elif point.startswith("attn1."):
            plan = decoder.second_attn_boundary.plan
        elif point.startswith("ffn0."):
            plan = decoder.first_ffn_boundary.plan
        elif point.startswith("ffn1."):
            plan = decoder.second_ffn_boundary.plan
        else:
            raise RuntimeError(f"Unknown trace point: {point}")
        variant = plan.variant_for(self.batch)
        need(variant.name == "ORDINARY", f"Unsupported boundary variant {variant.name}")
        edges = plan.edges[variant]
        if point in (
            "moe.reduced",
            "moe.branch_output",
            "layer.output",
            "layer.residual",
        ):
            layout = plan.branch_rows(self.batch)[2]
        elif point.endswith(("residual", "residual_in")):
            layout = edges.incoming.residual_to
        elif point.endswith("local_output") or point == "moe.experts_output":
            layout = plan.path_for(self.batch).output.layout
        else:
            layout = edges.incoming.need.layout
        return {a.name for a in layout.sharded}

    def record(self, layer, point, tensor, *, last_token_only=False):
        if tensor is None:
            return
        # finish() may publish a deferred sum or a batch-dependent transport.
        # Compare its consumer input instead of guessing its output row domain.
        if point in ("ffn0.finish", "ffn1.finish"):
            return
        info = self.context["topology"]
        axes = self.layout(layer, point)
        if point == "lm_head.logits":
            need(last_token_only and tensor.shape[0] >= 1, "Missing request logits")
            physical, logical, live, padding = [0], [self.n - 1], 1, tensor.shape[0] - 1
        else:
            physical, logical, live, padding = select_rows(
                axes,
                self.counts,
                self.context["dp"],
                info["attn_tp_rank"],
                self.n,
                tensor.shape[0],
            )
        values = (
            torch.stack([tensor.detach()[i] for i in physical]) if physical else None
        )
        self.samples.append(
            {
                "layer": layer,
                "point": point,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "axes": sorted(axes),
                "physical_rows": physical,
                "rows": logical,
                "live_rows": live,
                "padding_rows": padding,
                "status": "sampled" if physical else "no_live_rows",
                "values": values,
            }
        )

    def finish(self):
        ids, pos = self.ids.cpu().tolist(), self.positions.cpu().tolist()
        if self.mode == "EXTEND":
            need(pos == list(range(self.n)), "Cold prefill positions mismatch")
        else:
            need(
                ids == [self.context["forced_id"]]
                and pos == [self.context["prompt_len"]],
                f"Decode input mismatch: {ids}, {pos}",
            )
        nonfinite = []
        no_live = 0
        for sample in self.samples:
            values = sample["values"]
            if values is None:
                no_live += 1
                continue
            sample["values"] = values.detach().cpu().clone()
            if not torch.isfinite(sample["values"]).all():
                nonfinite.append([sample["layer"], sample["point"]])
        payload = {
            **self.context,
            "schema": 4,
            "num_layers": self.num_layers,
            "tp": self.context["topology"]["tp_rank"],
            "input_ids": ids,
            "positions": pos,
            "padded_counts": self.counts,
            "samples": self.samples,
        }
        root = Path(os.environ["LONGCAT_FIXED_DECODE_DIR"])
        path = root / f"{payload['rid']}.tp{payload['tp']}.{self.mode}.pt"
        with path.open("xb") as file:
            torch.save(payload, file)
        need(
            _active is not None and _active["rid"] == payload["rid"],
            "Probe context changed",
        )
        _active["saved"] = True
        print(
            "LONGCAT_FIXED_V4_TRACE "
            + json.dumps(
                {
                    "rid": payload["rid"],
                    "dp": payload["dp"],
                    "tp": payload["tp"],
                    "mode": self.mode,
                    "layers": self.num_layers,
                    "tokens": self.n,
                    "last_id": ids[-1],
                    "last_position": pos[-1],
                    "input_hash": hashlib.sha256(
                        json.dumps([ids, pos]).encode()
                    ).hexdigest()[:16],
                    "padded_counts": self.counts,
                    "points": len(self.samples),
                    "no_live_points": no_live,
                    "nonfinite_points": nonfinite,
                    "file": str(path),
                }
            ),
            flush=True,
        )


def make_fixed_trace(input_ids, positions, forward_batch, model):
    if _active is None:
        return None
    return FixedTrace(input_ids, positions, forward_batch, model)


def finish_fixed_sample(batch, forward_batch, next_token_ids, *, can_run_graph):
    req = request(batch)
    if req is None:
        return
    need(
        not can_run_graph
        and _active is not None
        and _active["rid"] == req["rid"]
        and _active["saved"],
        "Missing aligned eager trace",
    )
    need(next_token_ids.numel() == 1, "Expected one unpadded sampled token")
    if _active["mode"] == "EXTEND":
        next_token_ids.fill_(req["forced_id"])
        if _active["topology"]["attn_tp_rank"] == 0:
            print("LONGCAT_FIXED_V4_FORCE " + json.dumps(req), flush=True)
