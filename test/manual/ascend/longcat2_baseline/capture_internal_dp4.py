"""Eager, request-tagged CPU captures; no new collectives or NPU allocations.

Raw cache includes unused/padded slots: only consume entries selected by the
saved block tables, sequence lengths and sparse indices when doing references.
"""
import json
import os
from pathlib import Path

import torch
import longcat_fixed_decode_dp4 as probe


def capture_internal(attention_id, stage, **values):
    ctx = probe._active
    if ctx is None or not ctx['rid'].endswith('-r1'):
        return
    tp = ctx['topology']['tp_rank']
    ranks = {int(v) for v in os.environ.get('LC_INTERNAL_RANKS', '0,16,32,48').split(',')}
    layers = os.environ.get('LC_INTERNAL_LAYERS', '0,1,37')
    if tp not in ranks or (layers != 'all' and attention_id // 2 not in {int(v) for v in layers.split(',')}):
        return
    def cpu(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().clone()
        if isinstance(v, dict):
            return {k: cpu(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cpu(x) for x in v]
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        raise TypeError(type(v))
    root = Path(os.environ['LONGCAT_FIXED_DECODE_DIR']) / 'internal'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{ctx['rid']}.tp{tp}.{ctx['mode']}.attn{attention_id}.{stage}.pt"
    payload = {**ctx, 'attention_id': attention_id, 'layer': attention_id // 2,
               'attention_in_layer': attention_id % 2, 'stage': stage,
               'values': cpu(values)}
    with path.open('xb') as out:
        torch.save(payload, out)
    print('LONGCAT_INTERNAL ' + json.dumps({'rid': ctx['rid'], 'tp': tp,
          'mode': ctx['mode'], 'attention_id': attention_id, 'stage': stage,
          'file': str(path)}), flush=True)
