"""Schema 5: complete live boundary rows, streamed to immutable CPU files.

Only the request-origin Attention TP group records tensors. This is not an
all-expert dump or an all-EP reduction oracle. No new collectives are performed.
"""

import hashlib
import json
import os
from pathlib import Path

import torch

import longcat_fixed_decode_dp4 as probe


def cpu_rows(tensor, physical):
    """Copy the contiguous valid row interval, including every trailing column."""
    if not physical:
        return None
    probe.need(
        physical == list(range(physical[0], physical[-1] + 1)),
        "Expected a contiguous valid row interval",
    )
    return (
        tensor.detach()[physical[0] : physical[-1] + 1]
        .to("cpu", copy=True)
        .contiguous()
    )


def semantics(point):
    if point.endswith("residual") or point.endswith("residual_in"):
        return "independent_residual_state"
    if point == "layer.output":
        return "merged_branch_contribution_before_next_residual_update"
    if point.startswith("ffn") and point.endswith("local_output"):
        return "dense_tp_partial_do_not_compare_different_tp_shards_directly"
    if point.startswith("attn") and point.endswith("local_output"):
        return "attention_local_head_contribution_before_boundary_sum"
    if point == "moe.experts_output":
        return "expert_runner_output_on_origin_rows_before_branch_output"
    if point == "moe.reduced":
        return "ordinary_expert_branch_after_declared_sum_before_identity"
    if point == "moe.branch_output":
        return "ordinary_plus_identity_branch_not_residual_or_dense_ffn"
    if point == "moe.identity":
        return "identity_branch_zero_is_legal"
    if point == "router.expert_ids":
        return "global_expert_ids_before_identity_dispatch_rewrite"
    if point == "moe.dispatch_ids":
        return "dispatch_ids_after_identity_rewrite_sentinel_not_real_expert"
    if point.endswith("indexer"):
        return "indexer_slots_filter_sentinels_and_causal_validity_before_comparison"
    if point == "embedding.oe_ids":
        return "ngram_integer_ids"
    if point == "lm_head.logits":
        return "last_request_position_full_vocabulary_logits"
    return "as_consumed_boundary_tensor"


class FullBoundaryTrace(probe.FixedTrace):
    def __init__(self, input_ids, positions, batch, model):
        super().__init__(input_ids, positions, batch, model)
        self.profile = dict(probe.CAPTURE_PROFILE)
        probe.need(
            self.profile["schema"] == 5 and self.num_layers == 38,
            "Full boundary capture requires the complete 38-layer model",
        )
        self.layers = set(self.profile["layers"])
        # The consumer's residual gives an unambiguous view after applying the
        # previous block's contribution, without introducing a probe reduction.
        self.consumer_layers = {i + 1 for i in self.layers if i + 1 < self.num_layers}
        self.root = Path(os.environ["LONGCAT_FIXED_DECODE_DIR"])
        self.directory = self.root / (
            f"{self.context['rid']}.tp{self.context['topology']['tp_rank']}.{self.mode}.full"
        )
        self.warmup = self.context["rid"].endswith("-r0")
        if not self.warmup:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.ids_cpu = self.ids.cpu().tolist()
        self.pos_cpu = self.positions.cpu().tolist()
        self.seen = set()

    def selected(self, layer, point):
        return (
            layer == -1
            or layer in self.layers
            or (
                layer in self.consumer_layers
                and point in ("attn0.input", "attn0.residual")
            )
        )

    def stage_metadata(self, layer, point):
        info = self.context["topology"]
        meta = {"semantic": semantics(point), "origin_group_only": True}
        stream = getattr(self.batch, "residual_stream", None)
        pending = getattr(stream, "pending", None)
        owed = getattr(pending, "owed", None)
        meta["residual_stream"] = {
            "has_residual": getattr(stream, "residual", None) is not None,
            "has_pending_contribution": pending is not None,
            "pending_update_type": type(pending.update).__name__
            if pending is not None
            else None,
            "pending_completion_type": type(owed).__name__
            if owed is not None
            else None,
        }
        if layer < 0:
            return meta
        decoder = self.model.layers[layer]
        if point.startswith("attn"):
            index = int(point[4])
            attention = decoder.self_attn[index]
            heads = int(attention.num_local_heads)
            probe.need(
                int(attention.num_heads) == heads * 16,
                "Unexpected Attention head partition",
            )
            meta.update(
                attention_id=2 * layer + index,
                head_ids=list(
                    range(
                        info["attn_tp_rank"] * heads, (info["attn_tp_rank"] + 1) * heads
                    )
                ),
            )
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
            raise RuntimeError(f"Unknown boundary: {point}")
        output = plan.path_for(self.batch).output
        group = output.group
        meta["producer_declared_sum_group"] = group.name if group is not None else None
        if group is not None:
            from sglang.srt.layers.layer_boundary.layout import _sum_group

            meta["producer_declared_sum_ranks"] = list(_sum_group(group).ranks)
        # This describes the producer contract, not whether a later boundary
        # such as moe.reduced has already completed that operation.
        if point.startswith("ffn"):
            mlp = decoder.mlps[int(point[3])]
            meta["linear_partition"] = {
                name: {
                    key: int(getattr(module, key))
                    for key in (
                        "tp_rank",
                        "tp_size",
                        "input_size_per_partition",
                        "output_size_per_partition",
                    )
                    if hasattr(module, key)
                }
                for name, module in (
                    ("gate_up_proj", mlp.gate_up_proj),
                    ("down_proj", mlp.down_proj),
                )
            }
        if point.startswith(("moe.", "router.")):
            meta["ordinary_expert_count"] = int(decoder.mlp.num_experts)
            meta["expert_internal_capture"] = False
        return meta

    def record(self, layer, point, tensor, *, last_token_only=False):
        if (
            self.warmup
            or not self.selected(layer, point)
            or point in ("ffn0.finish", "ffn1.finish")
        ):
            return
        key = (layer, point)
        probe.need(key not in self.seen, f"Duplicate boundary {key}")
        self.seen.add(key)
        info = self.context["topology"]
        axes = self.layout(layer, point)
        sample = {
            "layer": layer,
            "point": point,
            "axes": sorted(axes),
            "row_policy": "all_valid",
            "column_policy": "all",
            "semantics": self.stage_metadata(layer, point),
        }
        if tensor is None:
            sample.update(status="absent", values_file=None, elements=0)
            self.samples.append(sample)
            return
        probe.need(
            isinstance(tensor, torch.Tensor) and tensor.ndim >= 1,
            f"Unsupported boundary object {key}: {type(tensor)}",
        )
        if point == "lm_head.logits":
            # DP LM head preserves local decode padding; the sampler consumes
            # the first live request row. Prefill has already pruned to one row.
            expected_rows = {1}
            if self.mode == "DECODE":
                expected_rows.add(self.counts[self.context["dp"]])
            probe.need(
                last_token_only and tensor.shape[0] in expected_rows,
                f"Unexpected request logits rows: {tensor.shape[0]}, "
                f"expected {sorted(expected_rows)} in {self.mode}",
            )
            physical, logical, live, padding = (
                [0],
                [self.n - 1],
                1,
                tensor.shape[0] - 1,
            )
        else:
            physical, logical, live, padding = probe.select_rows(
                axes,
                self.counts,
                self.context["dp"],
                info["attn_tp_rank"],
                self.n,
                tensor.shape[0],
                dp_size=info["attn_dp_size"],
                full=True,
            )
        value = cpu_rows(tensor, physical)
        sample.update(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            physical_rows=physical,
            rows=logical,
            live_rows=live,
            excluded_physical_rows=padding,
            token_ids=[self.ids_cpu[i] for i in logical],
            positions=[self.pos_cpu[i] for i in logical],
            status="full_valid" if physical else "no_live_rows",
            elements=value.numel() if value is not None else 0,
            values_file=None,
        )
        if value is not None:
            path = self.directory / f"layer{layer}.{point}.pt"
            sample["values_file"] = str(path.relative_to(self.root))
            sample["value_shape"] = list(value.shape)
            sample["tensor_sha256"] = hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            with path.open("xb") as out:
                torch.save(
                    {
                        "schema": 5,
                        "rid": self.context["rid"],
                        "mode": self.mode,
                        "topology": info,
                        **sample,
                        "values": value,
                    },
                    out,
                )
        # Keep metadata only. GPU/NPU temporaries and CPU tensors are released at
        # each boundary, rather than retaining 38 layers of device activations.
        self.samples.append(sample)

    def finish(self):
        if self.mode == "EXTEND":
            probe.need(
                self.pos_cpu == list(range(self.n)), "Cold prefill positions mismatch"
            )
        else:
            probe.need(
                self.ids_cpu == [self.context["forced_id"]]
                and self.pos_cpu == [self.context["prompt_len"]],
                "Decode input mismatch",
            )
        if self.warmup:
            probe.need(
                probe._active is not None
                and probe._active["rid"] == self.context["rid"],
                "Probe context changed",
            )
            probe._active["saved"] = True
            print(
                "LONGCAT_FULL_BOUNDARY "
                + json.dumps(
                    {
                        "rid": self.context["rid"],
                        "mode": self.mode,
                        "tp": self.context["topology"]["tp_rank"],
                        "status": "warmup_skipped",
                    }
                ),
                flush=True,
            )
            return
        required = {
            (-1, p) for p in ("embedding.output", "final_norm.output", "lm_head.logits")
        }
        for layer in self.layers:
            required.update(
                (layer, p)
                for p in (
                    "attn0.input",
                    "attn0.residual",
                    "attn0.local_output",
                    "moe.input",
                    "router.logits",
                    "router.expert_ids",
                    "router.weights",
                    "moe.dispatch_ids",
                    "moe.dispatch_weights",
                    "moe.experts_output",
                    "moe.reduced",
                    "moe.branch_output",
                    "ffn0.input",
                    "ffn0.residual_in",
                    "ffn0.local_output",
                    "attn1.input",
                    "attn1.residual",
                    "attn1.local_output",
                    "ffn1.input",
                    "ffn1.residual_in",
                    "ffn1.local_output",
                    "layer.output",
                    "layer.residual",
                )
            )
        required.update(
            (i, p)
            for i in self.consumer_layers
            for p in ("attn0.input", "attn0.residual")
        )
        probe.need(
            required <= self.seen,
            f"Missing full boundary calls: {required - self.seen}",
        )
        probe.need(
            all(
                s["status"] != "absent"
                for s in self.samples
                if (s["layer"], s["point"]) in required
            ),
            "Required boundary has no tensor",
        )
        payload = {
            **self.context,
            "schema": 5,
            "capture_profile": self.profile,
            "num_layers": self.num_layers,
            "selected_layers": sorted(self.layers),
            "consumer_layers": sorted(self.consumer_layers),
            "tp": self.context["topology"]["tp_rank"],
            "input_ids": self.ids_cpu,
            "positions": self.pos_cpu,
            "padded_counts": self.counts,
            "samples": self.samples,
            "skipped_points": ["ffn0.finish", "ffn1.finish"],
            "coverage": "origin Attention TP group; all live rows/all columns; no remote expert internals",
        }
        path = self.root / f"{payload['rid']}.tp{payload['tp']}.{self.mode}.pt"
        with path.open("xb") as out:
            torch.save(payload, out)
        # A completed index is the commit marker. A directory without it is an
        # interrupted capture and must not be treated as a complete request.
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2))
        probe.need(
            probe._active is not None and probe._active["rid"] == payload["rid"],
            "Probe context changed",
        )
        probe._active["saved"] = True
        print(
            "LONGCAT_FULL_BOUNDARY "
            + json.dumps(
                {
                    "rid": payload["rid"],
                    "tp": payload["tp"],
                    "dp": payload["dp"],
                    "mode": self.mode,
                    "layers": sorted(self.layers),
                    "points": len(self.samples),
                    "elements": sum(s["elements"] for s in self.samples),
                    "file": str(path),
                    "status": "complete",
                }
            ),
            flush=True,
        )
