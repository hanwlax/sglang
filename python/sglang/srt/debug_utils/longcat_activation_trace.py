"""Sample row-aligned LongCat activations without synchronizing between layers."""

import hashlib
import json
import logging

import torch

logger = logging.getLogger(__name__)


def _row_summary(row):
    row = row.flatten()
    width = row.numel()
    count = min(width, 12 if row.is_floating_point() else 16)
    cols = [i * (width - 1) // max(count - 1, 1) for i in range(count)]
    if not row.is_floating_point():
        return {"cols": cols, "values": row[cols].tolist()}

    row = row.to(torch.float64)
    finite = torch.isfinite(row)
    values = row[finite]
    return {
        "cols": cols,
        "values": [
            value if valid else None
            for value, valid in zip(row[cols].tolist(), finite[cols].tolist())
        ],
        "nan": int(torch.isnan(row).sum()),
        "posinf": int(torch.isposinf(row).sum()),
        "neginf": int(torch.isneginf(row).sum()),
        "nonzero_finite": int(torch.count_nonzero(values)),
        "mean": values.mean().item() if values.numel() else None,
        "rms": values.square().mean().sqrt().item() if values.numel() else None,
        "absmax": values.abs().max().item() if values.numel() else None,
    }


class LongCatActivationTrace:
    """Keep independent samples on device until the forward has finished."""

    def __init__(self, rank, mode, input_ids, positions):
        self.rank = rank
        self.mode = mode
        self.input_ids = input_ids.detach().clone()
        self.positions = positions.detach().clone()
        self.num_tokens = input_ids.numel()
        self.rows = tuple(sorted({0, self.num_tokens // 2, self.num_tokens - 1}))
        self.samples = []

    def record(self, layer, point, tensor, *, last_token_only=False):
        if tensor is None:
            return
        if tensor.shape[0] == self.num_tokens:
            rows = self.rows
            values = torch.stack([tensor.detach()[row] for row in rows])
        elif last_token_only and tensor.shape[0] == 1:
            rows = (self.num_tokens - 1,)
            values = tensor.detach().clone()
        else:
            # A padded or sharded layout may not share input token row numbers.
            rows = ()
            values = None
        self.samples.append(
            (layer, point, tuple(tensor.shape), str(tensor.dtype), rows, values)
        )

    def finish(self):
        input_ids = self.input_ids.cpu().tolist()
        positions = self.positions.cpu().tolist()
        payload = json.dumps([input_ids, positions], separators=(",", ":"))
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
        for layer, point, shape, dtype, rows, values in self.samples:
            record = {
                "input_hash": fingerprint,
                "mode": self.mode,
                "tp": self.rank,
                "tokens": self.num_tokens,
                "layer": layer,
                "point": point,
                "shape": shape,
                "dtype": dtype,
                "rows": [],
            }
            if values is not None:
                for row, value in zip(rows, values.cpu()):
                    summary = _row_summary(value)
                    if point == "lm_head.logits":
                        logits = value.flatten().float()
                        logits = logits.masked_fill(~torch.isfinite(logits), -torch.inf)
                        top = logits.topk(min(5, logits.numel()))
                        summary["top_logits"] = [
                            [index, logit]
                            for index, logit in zip(
                                top.indices.tolist(), top.values.tolist()
                            )
                            if logit != -torch.inf
                        ]
                    record["rows"].append(
                        {
                            "row": row,
                            "position": positions[row],
                            "input_id": input_ids[row],
                            **summary,
                        }
                    )
            logger.info("LONGCAT_ACT %s", json.dumps(record, separators=(",", ":")))
