"""Bounded, opt-in eager snapshots for comparing HCCL execution modes.

No extra collectives, RNG calls, or tensor mutations. Copying to CPU DOES
synchronize the producing stream: these traces cannot exonerate timing bugs.
Set the environment before importing model/distributed modules; disabled
decorators return the original function, including on compile/graph paths.
"""

import functools
import hashlib
import inspect
import json
import logging
import os
import socket
import threading
from collections import Counter
from pathlib import Path

import torch
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def tensor_summary(tensor):
    """Return a detached CPU snapshot and stats over ALL logical elements."""
    cpu = tensor.detach().to(device="cpu", copy=True).contiguous()
    flat = cpu.reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes()
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if flat.numel():
        values = flat.double()
        finite = torch.isfinite(values)
        valid = values[finite]
        summary.update(
            nan=int(torch.isnan(values).sum()),
            posinf=int(torch.isposinf(values).sum()),
            neginf=int(torch.isneginf(values).sum()),
            min=float(valid.min()) if valid.numel() else None,
            max=float(valid.max()) if valid.numel() else None,
            mean=float(valid.mean()) if valid.numel() else None,
            l2=float(torch.linalg.vector_norm(valid)),
            sample=[
                v if finite[i] else str(v) for i, v in enumerate(flat[:8].tolist())
            ],
        )
        # Preserve all small token/index vectors in JSON, even without .pt dumps.
        if not cpu.is_floating_point() and flat.numel() <= 256:
            summary["values"] = cpu.tolist()
    return cpu, summary


def _resolve(values, path):
    value = values
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, (tuple, list)) and part.isdigit():
            value = value[int(part)] if int(part) < len(value) else None
        else:
            value = getattr(value, part, None)
        if value is None:
            break
    return value


class HcclDebugRecorder:
    def __init__(self, directory, *, rank, metadata=None):
        self.directory = Path(directory) / f"rank-{rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        # Refuse to append another process/run to an existing rank's evidence.
        with self.path.open("x") as stream:
            stream.write(
                json.dumps({"type": "manifest", "rank": rank, **(metadata or {})})
                + "\n"
            )
        self.event_ct = 0
        self.dump_bytes = 0
        self.steps = Counter()
        self.local = threading.local()
        self.max_steps = envs.SGLANG_DEBUG_HCCL_MAX_STEPS.get()
        self.skip_steps = envs.SGLANG_DEBUG_HCCL_SKIP_STEPS.get()
        self.max_events = envs.SGLANG_DEBUG_HCCL_MAX_EVENTS.get()
        self.layers = set(envs.SGLANG_DEBUG_HCCL_LAYERS.get())
        self.save_tensors = envs.SGLANG_DEBUG_HCCL_SAVE_TENSORS.get()
        self.max_dump_bytes = envs.SGLANG_DEBUG_HCCL_MAX_DUMP_MB.get() * 1024**2
        if (
            min(self.max_steps, self.max_events) <= 0
            or min(self.skip_steps, self.max_dump_bytes) < 0
        ):
            raise ValueError(
                "HCCL debug limits must be positive (skip/dump may be zero)"
            )
        logger.warning(
            "HCCL debug snapshots: %s; touch %s/START after warmup to begin; "
            "CPU copies synchronize streams; "
            "eager numerical comparison only, not a timing/performance test",
            self.path,
            self.directory.parent,
        )

    @property
    def stack(self):
        if not hasattr(self.local, "stack"):
            self.local.stack = []
        return self.local.stack

    def emit(self, edge, values, paths, metadata):
        if self.event_ct >= self.max_events:
            return
        event = {
            "type": "snapshot",
            "event": self.event_ct,
            "root": self.stack[0]["root"],
            "scope": [entry["name"] for entry in self.stack],
            "edge": edge,
            "metadata": metadata,
            "tensors": {},
        }
        for path in paths:
            tensor = _resolve(values, path)
            if tensor is None:
                continue
            # Collections are explicit paths, not arbitrary model object walks.
            if not isinstance(tensor, torch.Tensor):
                if isinstance(tensor, (str, bool, int, float)) or (
                    isinstance(tensor, (tuple, list))
                    and len(tensor) <= 256
                    and all(isinstance(x, int) for x in tensor)
                ):
                    event["metadata"][path] = tensor
                continue
            cpu, summary = tensor_summary(tensor)
            if "logits" in path and cpu.ndim >= 2 and cpu.shape[-1] > 0:
                rows = cpu.reshape(-1, cpu.shape[-1])[:8].float()
                top = rows.topk(min(5, rows.shape[-1]), dim=-1)
                summary["topk_indices"] = top.indices.tolist()
                # Strings retain masked -inf without nonstandard JSON numbers.
                summary["topk_values"] = [
                    [str(x) for x in row] for row in top.values.tolist()
                ]
            size = cpu.numel() * cpu.element_size()
            if self.save_tensors and self.dump_bytes + size <= self.max_dump_bytes:
                name = f"{self.event_ct:06d}-{len(event['tensors']):03d}.pt"
                torch.save(cpu, self.directory / name)
                summary["file"] = name
                self.dump_bytes += size
            elif self.save_tensors:
                summary["dump_skipped"] = "byte_budget"
            event["tensors"][path] = summary
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.event_ct += 1
        if self.event_ct == self.max_events:
            with self.path.open("a") as stream:
                stream.write(
                    json.dumps({"type": "limit", "max_events": self.max_events}) + "\n"
                )
            logger.warning("HCCL debug event limit reached: %s", self.path)


def _get_recorder(create_if_missing=True):
    # Lazy imports avoid a parallel_state -> runtime_context import cycle.
    from sglang.srt.runtime_context import get_buffer, get_resources, get_server_args

    if not create_if_missing:
        return get_resources().buffers.get("hccl_debug_recorder")

    def create():
        args = get_server_args()
        graph = args.cuda_graph_config
        if (
            graph is None
            or graph.decode.backend != "disabled"
            or graph.prefill.backend != "disabled"
            or args.enable_torch_compile
        ):
            raise RuntimeError(
                "SGLANG_DEBUG_HCCL_DIR requires both "
                "--cuda-graph-backend-decode disabled and "
                "--cuda-graph-backend-prefill disabled, without torch.compile"
            )
        dist = torch.distributed
        rank = dist.get_rank() if dist.is_initialized() else 0
        metadata = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "torch": torch.__version__,
            "hccl_op_expansion_mode_requested": os.getenv("HCCL_OP_EXPANSION_MODE"),
            "actual_hccl_engine": "unknown; consult HCCL logs",
            "configuration": {
                name: getattr(args, name, None)
                for name in (
                    "model_path",
                    "tokenizer_path",
                    "dtype",
                    "quantization",
                    "tp_size",
                    "dp_size",
                    "nnodes",
                    "ep_size",
                    "enable_dp_attention",
                    "enable_dp_lm_head",
                    "moe_a2a_backend",
                    "deepep_mode",
                    "speculative_algorithm",
                    "speculative_draft_model_path",
                    "speculative_dspark_block_size",
                    "speculative_eagle_topk",
                    "mamba_ssm_dtype",
                    "chunked_prefill_size",
                    "max_running_requests",
                )
            },
            "limits": {
                name: getattr(envs, name).get()
                for name in (
                    "SGLANG_DEBUG_HCCL_MAX_STEPS",
                    "SGLANG_DEBUG_HCCL_SKIP_STEPS",
                    "SGLANG_DEBUG_HCCL_MAX_EVENTS",
                    "SGLANG_DEBUG_HCCL_LAYERS",
                    "SGLANG_DEBUG_HCCL_SAVE_TENSORS",
                    "SGLANG_DEBUG_HCCL_MAX_DUMP_MB",
                )
            },
        }
        return HcclDebugRecorder(
            envs.SGLANG_DEBUG_HCCL_DIR.get(), rank=rank, metadata=metadata
        )

    return get_buffer("hccl_debug_recorder", create)


def hccl_trace(
    label, *, inputs=(), outputs=(), root=False, layer=False, collective=False
):
    """Trace selected arguments/returns, preserving in-place input snapshots.

    Root steps are counted separately by label and forward mode. Nested scopes
    share their outer root's budget. A layer filter also suppresses collectives
    in excluded layers. Collectives outside a selected root are never traced.
    """

    def decorate(fn):
        if not envs.SGLANG_DEBUG_HCCL_DIR.get():
            return fn
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            recorder = _get_recorder(create_if_missing=root)
            if recorder is None or recorder.event_ct >= recorder.max_events:
                return fn(*args, **kwargs)
            if not root and not recorder.stack:
                return fn(*args, **kwargs)
            if (
                not recorder.stack
                and not (recorder.directory.parent / "START").exists()
            ):
                return fn(*args, **kwargs)
            values = signature.bind(*args, **kwargs).arguments
            obj = values.get("self")
            batch = values.get("forward_batch", values.get("batch"))
            mode = getattr(getattr(batch, "forward_mode", None), "name", "unknown")
            name = label
            enabled = not recorder.stack or recorder.stack[-1]["enabled"]
            if layer:
                layer_idx = str(obj.layer_idx)
                name += f"[{layer_idx}]"
                enabled = enabled and (
                    not recorder.layers or layer_idx in recorder.layers
                )
            metadata = {"mode": mode}
            if collective:
                metadata.update(
                    group=obj.unique_name,
                    ranks=obj.ranks,
                    rank_in_group=obj.rank_in_group,
                    world_size=obj.world_size,
                )
            if not recorder.stack:
                key = f"{label}:{mode}"
                step = recorder.steps[key]
                recorder.steps[key] += 1
                enabled = (
                    recorder.skip_steps
                    <= step
                    < recorder.skip_steps + recorder.max_steps
                )
                root_key = [key, step]
            else:
                root_key = recorder.stack[0]["root"]
            recorder.stack.append({"name": name, "root": root_key, "enabled": enabled})
            try:
                if enabled:
                    recorder.emit("before", values, inputs, metadata.copy())
                result = fn(*args, **kwargs)
                if enabled:
                    recorder.emit(
                        "after", {**values, "result": result}, outputs, metadata.copy()
                    )
                return result
            finally:
                recorder.stack.pop()

        return wrapped

    return decorate
