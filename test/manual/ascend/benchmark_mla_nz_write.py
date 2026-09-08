"""Paired eager/NPUGraph benchmarks of complete multi-layer MLA NZ writes.

No model/server is launched. Each sample covers both latent and RoPE writes
for all layers, including index generation and source packing. Results are
local write-chain timings, not serving TPOT or distributed critical-path time.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMLATokenToKVPool,
    _mla_fia_nz_scatter_indices,
)
from sglang.srt.runtime_context import get_forward


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def benchmark(args, rows):
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer, pool.page_size = 0, 128
    pool.kv_lora_rank, pool.qk_rope_head_dim = 512, 64
    pool.k_buffer, pool.v_buffer = [
        torch.zeros(
            args.layers, args.pages, 128, 1, d, device="npu", dtype=torch.bfloat16
        )
        for d in (512, 64)
    ]
    if rows > args.pages * 128:
        raise ValueError("rows exceed cache capacity")
    loc = torch.arange(rows, dtype=torch.int32, device="npu")
    values = torch.randn(rows, 1, 576, device="npu", dtype=torch.bfloat16)
    k, r = values.split([512, 64], dim=-1)

    def native():
        for layer in range(args.layers):
            for buf, src, dim in ((pool.k_buffer, k, 512), (pool.v_buffer, r, 64)):
                indices = _mla_fia_nz_scatter_indices(loc, dim, 128)
                torch_npu.npu_scatter_nd_update_(
                    buf[layer].view(-1, 16), indices, src.contiguous().view(-1, 16)
                )

    def write(shared):
        with get_forward().scoped(npu_mla_nz_indices={} if shared else None):
            for layer in range(args.layers):
                pool._set_fia_nz_kv_buffer(layer, loc, k, r)

    functions = {
        "native": native,
        "fused": lambda: write(False),
        "shared": lambda: write(True),
    }
    report = {"rows": rows, "layers": args.layers, "pages": args.pages, "modes": {}}
    for mode in ("eager", "graph"):
        runs = {}
        for name, fn in functions.items():
            for _ in range(args.warmup):
                fn()
            torch.npu.synchronize()
            if mode == "graph":
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    fn()
                runs[name] = graph.replay
            else:
                runs[name] = fn
        # Keep the same slots/values for each paired round. Rotate order to
        # distribute cache/frequency drift across all three implementations.
        samples = {name: [] for name in runs}
        for group in range(args.groups):
            names = list(runs)
            names = names[group % len(names) :] + names[: group % len(names)]
            for name in names:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                torch.npu.synchronize()
                start.record()
                wall_start = time.perf_counter()
                for _ in range(args.repeats):
                    runs[name]()
                submit_ms = (time.perf_counter() - wall_start) * 1000 / args.repeats
                end.record()
                end.synchronize()
                samples[name].append(
                    {
                        "device_ms": start.elapsed_time(end) / args.repeats,
                        "submit_ms": submit_ms,
                    }
                )
        report["modes"][mode] = {
            name: {
                "median_ms": statistics.median(s["device_ms"] for s in values),
                "p90_ms": percentile([s["device_ms"] for s in values], 0.9),
                "samples": values,
            }
            for name, values in samples.items()
        }
        print(
            json.dumps(
                {
                    "rows": rows,
                    "mode": mode,
                    "summary": {
                        name: {k: v for k, v in stats.items() if k != "samples"}
                        for name, stats in report["modes"][mode].items()
                    },
                }
            ),
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[48, 88, 128])
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        results = [benchmark(args, rows) for rows in args.rows]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
