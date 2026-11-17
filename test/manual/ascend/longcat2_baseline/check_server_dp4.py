#!/usr/bin/env python3
"""Read effective server configuration before evaluating/capturing."""
import argparse
import json
import urllib.request
from pathlib import Path


def validate(info, trace=False):
    expected = {'tp_size':64, 'attn_dp_size':4, 'ep_size':64,
                'moe_a2a_backend':'deepep', 'deepep_mode':'auto',
                'cuda_graph_backend_prefill':'disabled', 'cuda_graph_backend_decode':'disabled'}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f'{key}: expected {value!r}, got {info.get(key)!r}')
    if info.get('speculative_algorithm') not in (None, ''):
        raise ValueError('NEXTN must be off')
    override = info.get('json_model_override_args') or '{}'
    if isinstance(override, str):
        override = json.loads(override)
    if any(k in override for k in ('num_layers', 'num_hidden_layers')):
        raise ValueError('Layer-count override is not a full-model baseline')
    if trace and int(info['chunked_prefill_size']) < 1024:
        raise ValueError('Fixed-prefix capture needs at least 1024 tokens per DP chunk')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(args.url.rstrip('/')+'/server_info',timeout=120) as r:
        info=json.load(r)
    args.out.write_text(json.dumps(info,indent=2))
    validate(info,args.trace)
    print('SERVER_CONFIG_OK TP64/ADP4/ATP16/EP64 graph=off NEXTN=off')
