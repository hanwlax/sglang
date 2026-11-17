#!/usr/bin/env python3
"""CPU capture tests: real row coverage, snapshots, DP2/4 and complete indexes."""

import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile

import torch

import longcat_fixed_decode_dp4 as probe
from longcat_full_boundary import FullBoundaryTrace, cpu_rows


def fails(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Expected rejection")


def layout(names):
    return NS(sharded=[NS(name=n) for n in names])


class Plan:
    def __init__(self, names):
        rows = layout(names)
        self.variant = NS(name="ORDINARY")
        self.edge = NS(incoming=NS(need=NS(layout=rows), residual_to=rows))
        self.output = NS(layout=rows, group=None)
        self.names = names

    @property
    def edges(self):
        class Edges:
            def __getitem__(s, key):
                return self.edge

        return Edges()

    def variant_for(self, batch):
        return self.variant

    def path_for(self, batch):
        return NS(output=self.output)

    def branch_rows(self, batch):
        return layout(self.names), layout(self.names), layout(["ATTN_DP"])


def decoder():
    attn = lambda: NS(plan=Plan(["ATTN_DP"]))
    ffn = lambda: NS(plan=Plan([]))
    linear = lambda: NS(tp_rank=0, tp_size=64, input_size_per_partition=32)
    return NS(
        attn_boundary=attn(),
        second_attn_boundary=attn(),
        first_ffn_boundary=ffn(),
        second_ffn_boundary=ffn(),
        moe_boundary=NS(plan=Plan(["ATTN_DP", "ATTN_TP"])),
        self_attn=[NS(num_heads=64, num_local_heads=4) for _ in range(2)],
        mlps=[NS(gate_up_proj=linear(), down_proj=linear()) for _ in range(2)],
        mlp=NS(num_experts=768),
    )


POINTS = (
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
    "moe.identity",
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


def main():
    with tempfile.TemporaryDirectory(prefix="lc-full-boundary-cpu-") as tmp:
        root = Path(tmp)
        os.environ["LONGCAT_FIXED_DECODE_DIR"] = str(root)
        model = NS(layers=[decoder() for _ in range(38)])
        for dp_size in (2, 4):
            probe.DP_SIZE = dp_size
            probe.CAPTURE_PROFILE = dict(schema=5, layers=[0, 1], attn_dp_size=dp_size)
            counts = [864 + 16 * i for i in range(dp_size)]
            for dp in range(dp_size):
                # All sixteen shards must cover each real token exactly once.
                union = []
                for local in range(16):
                    ph, rows, live, pad = probe.select_rows(
                        {"ATTN_DP", "ATTN_TP"},
                        counts,
                        dp,
                        local,
                        852,
                        counts[dp] // 16,
                        dp_size=dp_size,
                        full=True,
                    )
                    union += rows
                    assert len(rows) == live and all(
                        0 <= i < counts[dp] // 16 for i in ph
                    )
                assert union == list(range(852))
                for local in (0, 15):
                    for mode in ("EXTEND", "DECODE"):
                        n = 852 if mode == "EXTEND" else 1
                        padded = counts if n == 852 else [16] * dp_size
                        rank = dp * 16 + local
                        topo = dict(
                            tp_rank=rank,
                            tp_size=dp_size * 16,
                            attn_dp_rank=dp,
                            attn_dp_size=dp_size,
                            attn_tp_rank=local,
                            attn_tp_size=16,
                            moe_ep_rank=rank,
                            moe_ep_size=dp_size * 16,
                            moe_tp_size=1,
                            moe_dp_size=1,
                            attn_cp_size=1,
                            pp_size=1,
                        )
                        rid = f"lcfix{dp_size}-0123456789ab-d{dp}-p852-t70115-r1"
                        probe._active = dict(
                            rid=rid,
                            dp=dp,
                            prompt_len=852,
                            forced_id=70115,
                            mode=mode,
                            topology=topo,
                            raw_counts=[n if i == dp else 0 for i in range(dp_size)],
                        )
                        batch = NS(global_num_tokens_cpu=padded, residual_stream=None)
                        ids = torch.arange(852) if n == 852 else torch.tensor([70115])
                        positions = (
                            torch.arange(852) if n == 852 else torch.tensor([852])
                        )
                        trace = FullBoundaryTrace(ids, positions, batch, model)
                        keys = [(-1, "embedding.output")] + [
                            (l, p) for l in (0, 1) for p in POINTS
                        ]
                        keys += [
                            (2, "attn0.input"),
                            (2, "attn0.residual"),
                            (-1, "final_norm.output"),
                            (-1, "lm_head.logits"),
                        ]
                        for layer, point in keys:
                            axes = trace.layout(layer, point)
                            is_logits = point == "lm_head.logits"
                            size = (
                                (padded[dp] if mode == "DECODE" else 1)
                                if is_logits
                                else sum(padded)
                                if not axes
                                else padded[dp] // 16
                                if "ATTN_TP" in axes
                                else padded[dp]
                            )
                            x = torch.full(
                                (size, 5), float("nan"), dtype=torch.bfloat16
                            )
                            if is_logits:
                                ph, rows = [0], [n - 1]
                            else:
                                ph, rows, _, _ = probe.select_rows(
                                    axes,
                                    padded,
                                    dp,
                                    local,
                                    n,
                                    size,
                                    dp_size=dp_size,
                                    full=True,
                                )
                            for physical, logical in zip(ph, rows):
                                x[physical] = float(logical % 100)
                            trace.record(layer, point, x, last_token_only=is_logits)
                            # Later mutation must not alter the on-disk boundary.
                            x.zero_()
                        # Unselected block is ignored, not an accidental whole-model dump.
                        trace.record(3, "attn0.input", torch.ones((padded[dp], 5)))
                        trace.finish()
                        index = root / f"{rid}.tp{rank}.{mode}.pt"
                        meta = torch.load(index, weights_only=True)
                        assert meta["schema"] == 5 and len(meta["samples"]) == len(keys)
                        for sample in meta["samples"]:
                            assert sample["row_policy"] == "all_valid"
                            assert sample["elements"] == len(sample["rows"]) * 5
                            if sample["values_file"]:
                                payload = torch.load(
                                    root / sample["values_file"], weights_only=True
                                )
                                v = payload["values"]
                                assert v.isfinite().all(), (
                                    "Padding leaked into valid rows"
                                )
                                expected = torch.tensor(
                                    [r % 100 for r in sample["rows"]], dtype=v.dtype
                                )
                                assert torch.equal(v[:, 0], expected), (
                                    "Snapshot or row mapping corrupted"
                                )
                        if mode == "DECODE" and local == 15:
                            assert any(
                                s["status"] == "no_live_rows" for s in meta["samples"]
                            )
                        # Fail a missing required capture, instead of completing an index.
                        trace.seen.remove((0, "router.logits"))
                        fails(trace.finish)
        fails(
            lambda: probe.select_rows(
                set(), [16, 16], 0, 0, 17, 32, dp_size=2, full=True
            )
        )
        fails(
            lambda: probe.select_rows(
                {"ATTN_CP"}, [16, 16], 0, 0, 1, 16, dp_size=2, full=True
            )
        )
        # Nonfinite *valid* elements and integer IDs must be preserved verbatim.
        value = cpu_rows(torch.tensor([[float("nan"), float("inf")], [1.0, 2.0]]), [0])
        assert value[0, 0].isnan() and value[0, 1].isinf()
        ids = torch.tensor([[838, 39, -1]], dtype=torch.int64)
        assert torch.equal(cpu_rows(ids, [0]), ids)
        probe._active["rid"] = probe._active["rid"][:-1] + "0"
        trace = FullBoundaryTrace(
            torch.tensor([70115]), torch.tensor([852]), batch, model
        )
        trace.record(0, "attn0.input", torch.ones(16, 5))
        trace.finish()
        assert not trace.directory.exists() and probe._active["saved"]
    print(
        "FULL_BOUNDARY_CPU_PASS DP2/DP4 all valid rows, gathered offsets, padding, immutable snapshots, full columns, completion checks"
    )


if __name__ == "__main__":
    main()
