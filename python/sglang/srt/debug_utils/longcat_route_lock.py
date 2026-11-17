"""Opt-in two-layer TP8/DP1/EP1 LongCat routing experiment, no graphs/NEXTN.

Call after routed_scaling_factor multiplication and BEFORE identity processing.
LC_ROUTE_MODE: off (default), capture, ids, ids_weights.
Only the fixed 852-token cold prefill is eligible; all other forwards are untouched.
This is a diagnostic intervention, not an accuracy fix.
"""

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

LOG = logging.getLogger(__name__)
TARGET = "fd85b1782b833ab4"


def require(ok, message):
    if not ok:
        raise RuntimeError("LONGCAT_ROUTE_LOCK: " + message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def trace_identity(trace):
    identity = getattr(trace, "_lc_route_identity", None)
    if identity is None:
        token_ids = trace.input_ids.cpu().tolist()
        positions = trace.positions.cpu().tolist()
        identity = {
            "input_ids": token_ids,
            "positions": positions,
            "input_hash": digest([token_ids, positions])[:16],
        }
        trace._lc_route_identity = identity
    return identity


def dump_full_tensor(trace, layer, point, tensor, *, last_token_only=False):
    """Save every element BEFORE the caller/kernel can mutate the tensor.

    Synchronous diagnostic copies deliberately change timing. Use identical
    instrumentation for natural and forced runs on both platforms.
    """
    folder = os.environ.get("LC_FULL_DUMP_OUT")
    if (
        not folder
        or tensor is None
        or trace.mode != "EXTEND"
        or trace.num_tokens != 852
    ):
        return
    identity = trace_identity(trace)
    if identity["input_hash"] != TARGET:
        return
    import torch

    require(
        identity["positions"] == list(range(852)) and trace.rank in range(8),
        "full dump requires cold TP8 prefill",
    )
    require(tensor.ndim >= 1, "full dump needs row-major activations")
    if tensor.shape[0] == 852:
        rows = list(range(852))
    elif last_token_only and tensor.shape[0] == 1:
        rows = [851]
    else:
        raise RuntimeError(
            f"unaligned full dump L{layer} {point}: {tuple(tensor.shape)}"
        )
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"full_L{layer}_{point}_TP{trace.rank}.pt"
    value = tensor.detach().to(device="cpu", copy=True).contiguous()
    payload = {
        "schema": 1,
        "layer": layer,
        "point": point,
        "tp": trace.rank,
        "identity": identity,
        "mode": "EXTEND",
        "rows": rows,
        "route_mode": os.environ.get("LC_ROUTE_MODE", "off"),
        "tensor": value,
    }
    with dest.open("xb") as f:
        torch.save(payload, f)
    LOG.warning(
        "LONGCAT_FULL_TENSOR %s",
        json.dumps(
            {
                "layer": layer,
                "point": point,
                "tp": trace.rank,
                "shape": list(value.shape),
                "file": str(dest),
            },
            separators=(",", ":"),
        ),
    )


@lru_cache(maxsize=1)
def reference(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    require(
        data.get("schema") == 1 and data.get("kind") == "longcat_route_reference",
        "bad reference schema",
    )
    require(set(data["layers"]) == {"0", "1"}, "reference needs both layers")
    return data, hashlib.sha256(raw).hexdigest()


def weights_for_ids(logits, ids, scoring, factor):
    import torch.nn.functional as F

    # c0e9d9a0cc6b fused_topk_torch_native, correction_bias present:
    # bias affects choice only; the weights come from the unbiased scores.
    if scoring == "softmax":
        scores = logits.softmax(dim=-1)
    elif scoring == "sigmoid":
        scores = logits.sigmoid()
    elif scoring == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    else:
        raise RuntimeError(f"unsupported scoring function {scoring}")
    return scores.gather(1, ids.long()) * factor


def apply_route_lock(moe, trace, logits, weights, ids):
    mode = os.environ.get("LC_ROUTE_MODE", "off")
    require(mode in {"off", "capture", "ids", "ids_weights"}, f"invalid mode {mode}")
    if (
        mode == "off"
        or trace is None
        or trace.mode != "EXTEND"
        or trace.num_tokens != 852
    ):
        return weights, ids
    identity = trace_identity(trace)
    if identity["input_hash"] != TARGET:
        return weights, ids

    import torch

    cfg = moe.topk.topk_config
    require(
        identity["positions"] == list(range(852)), "requires cold contiguous prefill"
    )
    require(moe.tp_size == 8 and trace.rank in range(8), "requires TP8")
    require(moe.layer_id in (0, 1), "requires two-layer run_trace model")
    require(
        moe.num_experts == 768 and moe.zero_expert_num == 128 and moe.top_k == 12,
        "requires 768 ordinary + 128 identity experts, topk12",
    )
    require(moe.zero_expert_type == "identity", "requires identity expert semantics")
    require(
        tuple(logits.shape) == (852, 896)
        and tuple(ids.shape) == tuple(weights.shape) == (852, 12),
        "requires full unsharded 852-row routing tensors",
    )
    require(
        logits.dtype == weights.dtype == torch.float32,
        "requires FP32 router scores/weights",
    )
    require(
        not cfg.renormalize
        and not cfg.use_grouped_topk
        and cfg.correction_bias is not None,
        "unexpected TopK configuration",
    )
    require(
        not cfg.apply_routed_scaling_factor_on_output
        and cfg.routed_scaling_factor is None,
        "TopK already folds scaling; inspect source before continuing",
    )
    require(
        torch.isfinite(logits).all().item() and torch.isfinite(weights).all().item(),
        "nonfinite natural routing data",
    )
    natural_ids = ids.detach().cpu().tolist()
    natural_weights = weights.detach().cpu().tolist()
    meta = {
        "num_experts": 768,
        "zero_expert_num": 128,
        "zero_expert_type": moe.zero_expert_type,
        "top_k": 12,
        "scoring_func": cfg.scoring_func,
        "routed_scaling_factor": float(moe.routed_scaling_factor),
        "weight_semantics": "post_scaling_before_identity",
        "tp_size": 8,
    }
    ref_sha = None
    if mode != "capture":
        ref, ref_sha = reference(os.environ["LC_ROUTE_REFERENCE"])
        require(
            ref["identity"] == identity and ref["meta"] == meta,
            "reference input/positions/router configuration mismatch",
        )
        saved = ref["layers"][str(moe.layer_id)]
        forced_ids = saved["ids"]
        require(
            len(forced_ids) == 852
            and all(
                len(row) == 12
                and len(set(row)) == 12
                and all(type(i) is int and 0 <= i < 896 for i in row)
                for row in forced_ids
            ),
            "invalid forced expert IDs",
        )
        require(
            len(saved["weights"]) == 852
            and all(len(row) == 12 for row in saved["weights"]),
            "invalid reference weights shape",
        )
        new_ids = torch.tensor(forced_ids, dtype=ids.dtype, device=ids.device)
        if mode == "ids_weights":
            new_weights = torch.tensor(
                saved["weights"], dtype=weights.dtype, device=weights.device
            )
        else:
            original = weights_for_ids(
                logits, ids, cfg.scoring_func, moe.routed_scaling_factor
            )
            require(
                torch.equal(original, weights),
                "native weight reconstruction differs; do not reuse slot weights",
            )
            new_weights = weights_for_ids(
                logits, new_ids, cfg.scoring_func, moe.routed_scaling_factor
            )
        require(
            torch.isfinite(new_weights).all().item()
            and (new_weights >= 0).all().item(),
            "invalid forced routing weights",
        )
        weights, ids = new_weights.contiguous(), new_ids.contiguous()
    applied_ids = ids.detach().cpu().tolist()
    applied_weights = weights.detach().cpu().tolist()
    payload = {
        "schema": 1,
        "mode": mode,
        "tp": trace.rank,
        "layer": moe.layer_id,
        "identity": identity,
        "meta": meta,
        "reference_sha256": ref_sha,
        "natural_ids": natural_ids,
        "natural_weights": natural_weights,
        "ids": applied_ids,
        "weights": applied_weights,
    }
    out = Path(os.environ["LC_ROUTE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"route_L{moe.layer_id}_TP{trace.rank}.json"
    # One matching forward per output directory. Repeated requests must use a fresh service/run directory.
    with dest.open("x") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    LOG.warning(
        "LONGCAT_ROUTE_LOCK %s",
        json.dumps(
            {
                "mode": mode,
                "layer": moe.layer_id,
                "tp": trace.rank,
                "input_hash": TARGET,
                "tokens": 852,
                "reference_sha256": ref_sha,
                "ids_sha256": digest(applied_ids),
                "weights_sha256": digest(applied_weights),
                "changed_id_slots": sum(
                    i != j
                    for a, b in zip(natural_ids, applied_ids)
                    for i, j in zip(a, b)
                ),
                "file": str(dest),
            },
            separators=(",", ":"),
        ),
    )
    return weights, ids
