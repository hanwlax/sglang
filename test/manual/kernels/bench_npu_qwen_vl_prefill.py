"""Compare the real NPU gate/up paths with FP64 samples and profiler timings.

Run with --output-dir pointing to a new directory. Synthetic inputs are seeded
on CPU, so the same commands can be compared across NPU platforms. Optionally
provide a trusted torch.save({'x': cpu_bf16_x, 'weight': cpu_bf16_weight}) capture.
This is a single-operator benchmark, not an end-to-end throughput measurement.
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.modules.linear import prefill_gate_up
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def digest(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def error_metrics(actual, reference):
    delta = actual.double() - reference
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
    }


def reference_rows(m):
    # Include historical K-shift row boundaries as well as the last token.
    return sorted(
        {0, m - 1}
        | {r for r in (1, 511, 512, 1023, 1024, 1535, 1536) if r < m}
        | set(torch.linspace(0, m - 1, min(m, 16)).long().tolist())
    )


def precision(x_cpu, weight_cpu, layer, x):
    rows = reference_rows(len(x_cpu))
    reference = x_cpu[rows].double() @ weight_cpu.double().t()
    outputs = {}
    result = {"fp64_rows": rows, "reference_elements": reference.numel()}
    for name, fn in (
        ("linear", lambda: layer(x)[0]),
        ("gmm", lambda: prefill_gate_up(layer, x, ForwardMode.EXTEND)[0]),
    ):
        y = fn()
        assert y.dtype == torch.bfloat16 and y.shape == (len(x), len(weight_cpu))
        outputs[name] = y.cpu()
        sample = outputs[name][rows].double()
        torch.testing.assert_close(sample, reference, rtol=0.005, atol=0.001)
        for _ in range(2):
            torch.testing.assert_close(fn().cpu(), outputs[name], rtol=0, atol=0)
        result[name] = {
            **error_metrics(sample, reference),
            "sha256": digest(outputs[name]),
            "three_runs_bitwise_equal": True,
        }
    result["different_output_elements"] = (
        (outputs["linear"] != outputs["gmm"]).sum().item()
    )
    result["output_elements"] = outputs["linear"].numel()
    result["max_linear_gmm_abs"] = (
        (outputs["linear"].float() - outputs["gmm"].float()).abs().max().item()
    )
    return result


def profile_path(fn, directory, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
    ):
        for _ in range(iterations):
            fn()
            torch.npu.synchronize()
    summaries = list(directory.rglob("op_statistic.csv"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one profiler summary: {summaries}")
    with summaries[0].open() as stream:
        records = list(csv.DictReader(stream))
    matmuls = [r for r in records if "matmul" in r.get("OP Type", "").lower()]
    if not matmuls or sum(int(r["Count"]) for r in matmuls) != iterations:
        raise RuntimeError(f"Unexpected matmul coverage: {records}")
    if directory.name.endswith("-gmm") and not all(
        "group" in r["OP Type"].lower() for r in matmuls
    ):
        raise RuntimeError(
            f"The GMM case fell back to another implementation: {matmuls}"
        )
    total = sum(float(r["Total Time(us)"]) for r in records)
    return {
        "device_us_per_call": total / iterations,
        "matmul_us_per_call": sum(float(r["Total Time(us)"]) for r in matmuls)
        / iterations,
        "op_statistic": str(summaries[0].relative_to(directory)),
        "operators": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 128, 512, 2526])
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=11008)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 2026])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if min(*args.m, args.k, args.n, args.iterations, args.rounds) < 1:
        parser.error("Dimensions and measurement counts must be positive")
    if args.warmup < 5:
        parser.error("Use at least five warmup iterations")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    publish(ServerArgs(model_path="dummy", device="npu"), role="test")
    report = {
        "device": str(torch.npu.get_device_properties(0)),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "source_sha256": digest_source(),
        "dtype": "bfloat16",
        "method": "torch_npu.profiler; same tensors; warmed; alternating order",
        "scope": "Sampled CPU FP64 rows; full-output repeatability; operator timings",
        "tolerance": {"rtol": 0.005, "atol": 0.001},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "cases": [],
    }

    def save():
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2))

    jobs = [(f"m{m}-seed{seed}", m, seed) for m in args.m for seed in args.seeds]
    if args.capture:
        jobs = [("capture", None, None)]
    with torch.inference_mode():
        for label, m, seed in jobs:
            if args.capture:
                captured = torch.load(
                    args.capture, map_location="cpu", weights_only=True
                )
                x_cpu, weight_cpu = captured["x"], captured["weight"]
            else:
                generator = torch.Generator().manual_seed(seed)
                x_cpu = torch.randn(m, args.k, generator=generator).bfloat16()
                weight_cpu = (
                    torch.randn(args.n, args.k, generator=generator) * 0.02
                ).bfloat16()
            assert x_cpu.dtype == weight_cpu.dtype == torch.bfloat16
            m, k = x_cpu.shape
            n, wk = weight_cpu.shape
            assert k == wk and n % 2 == 0
            layer = MergedColumnParallelLinear(
                k,
                [n // 2, n // 2],
                params_dtype=torch.bfloat16,
                tp_rank=0,
                tp_size=1,
                bias=False,
            ).to("npu")
            layer.weight.copy_(weight_cpu)
            x = x_cpu.to("npu")
            case = {
                "label": label,
                "shape_mkn": [m, k, n],
                "input_sha256": digest(x_cpu),
                "weight_sha256": digest(weight_cpu),
                "precision": precision(x_cpu, weight_cpu, layer, x),
                "profiles": [],
            }
            report["cases"].append(case)
            save()
            print(f"Precision passed: {label}", flush=True)
            # Profile one seed per shape. Other seeds expand numerical coverage.
            if args.capture or seed == args.seeds[0]:
                for repeat in range(args.rounds):
                    order = ["linear", "gmm"] if repeat % 2 == 0 else ["gmm", "linear"]
                    for name in order:
                        fn = (
                            (lambda layer=layer, x=x: layer(x)[0])
                            if name == "linear"
                            else (
                                lambda layer=layer, x=x: prefill_gate_up(
                                    layer, x, ForwardMode.EXTEND
                                )[0]
                            )
                        )
                        directory = args.output_dir / f"{label}-{repeat}-{name}"
                        timing = profile_path(
                            fn, directory, args.warmup, args.iterations
                        )
                        case["profiles"].append(
                            {"round": repeat, "path": name, **timing}
                        )
                        save()
                medians = {
                    name: statistics.median(
                        r["device_us_per_call"]
                        for r in case["profiles"]
                        if r["path"] == name
                    )
                    for name in ("linear", "gmm")
                }
                case["median_device_us"] = medians
                case["linear_over_gmm_speedup"] = medians["linear"] / medians["gmm"]
                save()
                print(json.dumps({"label": label, **medians}), flush=True)
            del layer, x
            torch.npu.empty_cache()
    report["successful"] = True
    save()
    reset_context()


def digest_source():
    import inspect

    return hashlib.sha256(
        Path(inspect.getfile(prefill_gate_up)).read_bytes()
    ).hexdigest()


if __name__ == "__main__":
    main()
