#!/usr/bin/env python3
"""Build a private frozen runtime; never patch the shared source checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REV = 'c0e9d9a0cc6b1a37e41677faf0c961fedc668af7'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', default='/home/hanwlax/workspace/sglang')
p.add_argument('--mode', choices=('eval', 'trace'), required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--full-boundary-layers', type=int, nargs='+', help='Save all live rows/all columns of selected blocks (full 38-layer model)')
p.add_argument('--with-internal-attention', action='store_true', help='Also install internal Attention hooks; LC_INTERNAL_LAYERS/RANKS select captures')
p.add_argument('--trace-topology', choices=('a3-dp4', 'a5-dp2'), default='a3-dp4')
a = p.parse_args()
if a.with_internal_attention and a.mode != 'trace':
    p.error('--with-internal-attention requires trace mode')
if a.full_boundary_layers is not None:
    if a.mode != 'trace' or not a.full_boundary_layers or len(set(a.full_boundary_layers)) != len(a.full_boundary_layers) or any(l < 0 or l >= 38 for l in a.full_boundary_layers):
        p.error('Full boundaries require trace mode and distinct block IDs 0..37')
if a.trace_topology == 'a5-dp2' and a.full_boundary_layers is None:
    p.error('A5 DP2 currently requires --full-boundary-layers')
dp_size = 2 if a.trace_topology == 'a5-dp2' else 4
bundle = Path(__file__).resolve().parent
a.out.mkdir(parents=True, exist_ok=False)
archive = subprocess.Popen(['git', '-C', a.source, 'archive', REV, 'python'], stdout=subprocess.PIPE)
with tarfile.open(fileobj=archive.stdout, mode='r|') as tar:
    tar.extractall(a.out, filter='data')
if archive.wait() != 0:
    raise RuntimeError('git archive failed; remove only this incomplete private runtime before retrying')
root = a.out / 'python/sglang/srt'

def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f'Anchor count mismatch: expected {count}: {old[:100]}')
    return text.replace(old, new)

def edit(rel, transform):
    f = root / rel
    s = transform(f.read_text())
    ast.parse(s)
    f.write_text(s)

def worker(s):
    anchor = '        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)'
    s = replace(s, anchor, '        from sglang.srt.debug_utils.longcat_fixed_decode_dp4 import bind_fixed_decode, finish_fixed_sample\n'
                '        bind_fixed_decode(batch, forward_batch, speculative=self.enable_spec)\n' + anchor)
    for indent in (20, 16):
        pad = ' ' * indent
        anchor = (pad + 'batch_result.next_token_ids = self.model_runner.sample(\n' +
                  pad + '    logits_output, forward_batch\n' + pad + ')')
        s = replace(s, anchor, anchor + '\n' + pad +
                    'finish_fixed_sample(batch, forward_batch, batch_result.next_token_ids, can_run_graph=can_run_cuda_graph)')
    return s

def model(s):
    if a.full_boundary_layers is not None:
        anchor = '            trace.record(self.layer_id, "attn0.input", hidden_states)'
        s = replace(s, anchor, anchor + '\n            trace.record(self.layer_id, "attn0.residual", forward_batch.residual_stream.residual)')
    anchor = '        trace = None\n        mode = forward_batch.forward_mode\n        if (\n            self.trace_activations'
    return replace(s, anchor,
        '        from sglang.srt.debug_utils.longcat_fixed_decode_dp4 import make_fixed_trace\n'
        '        trace = make_fixed_trace(input_ids, positions, forward_batch, self.model)\n'
        '        mode = forward_batch.forward_mode\n        if (\n'
        '            trace is None and self.trace_activations')

def attention(s):
    # Only modify the DSA prepare/core functions used by LongCat.
    start = s.index('def forward_dsa_prepare_npu(')
    end = s.index('\ndef npu_mla_preprocess(', start)
    t = s[start:end]
    anchor = '        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]'
    t = replace(t, anchor, anchor + '\n        from sglang.srt.debug_utils.capture_internal_dp4 import capture_internal\n'
        '        capture_internal(m.layer_id, "a_proj", input=hidden_states, output=fused_qkv_a_proj_out, positions=positions)')
    anchor = '        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)'
    t = replace(t, anchor, anchor + '\n'
        '        capture_internal(m.layer_id, "norm_qb", q_lora=q_lora, k_nope=k_nope, k_pe=k_pe, q=q, q_nope=q_nope, q_pe=q_pe)')
    anchor = '        if is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:'
    t = replace(t, anchor,
        '        capture_internal(m.layer_id, "qk_bmm", output=q_nope_out, w_kc=m.w_kc)\n' + anchor)
    anchor = '        if dsa_use_prefill_cp(forward_batch):\n            # support allgather+rerrange'
    t = replace(t, anchor,
        '        capture_internal(m.layer_id, "rope", q_pe=q_pe, k_pe=k_pe, positions=positions)\n' + anchor)
    anchor = '    output, _ = m.o_proj(attn_bmm_output)'
    t = replace(t, anchor, anchor + '\n    from sglang.srt.debug_utils.capture_internal_dp4 import capture_internal\n'
        '    capture_internal(m.layer_id, "tail", attn_output=attn_output, v_bmm=attn_bmm_output, w_vc=m.w_vc, output=output)')
    return s[:start] + t + s[end:]

def backend(s):
    start = s.index('    def forward_sparse(')
    end = s.index('\n    def _load_mla_prefix_cache(', start)
    t = s[start:end]
    anchor = '        return attn_out'
    t = replace(t, anchor,
        '        from sglang.srt.debug_utils.capture_internal_dp4 import capture_internal\n'
        '        capture_internal(layer.layer_id, "sparse", q=q_nope, q_pe=q_pe, cache_k=k_nope, cache_pe=k_pe, '
        'indices=topk_indices, block_tables=self.forward_metadata.block_tables, '
        'query_lengths=actual_seq_qlen, kv_lengths=actual_seq_lengths_kv, scale=layer.scaling, output=attn_out)\n' + anchor)
    return s[:start] + t + s[end:]

if a.mode == 'trace':
    edit('managers/tp_worker.py', worker)
    edit('models/longcat_flash.py', model)
    # Physical-cache/internal hooks are opt-in with full boundaries.
    if a.full_boundary_layers is None or a.with_internal_attention:
        edit('hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py', attention)
        edit('hardware_backend/npu/attention/ascend_backend.py', backend)
    # Fully qualified imports prevent a same-named bundle helper in cwd from
    # shadowing the frozen runtime's copy/profile.
    helpers = ['longcat_fixed_decode_dp4.py', 'capture_internal_dp4.py']
    if a.full_boundary_layers is not None:
        helpers.append('longcat_full_boundary.py')
    for name in helpers:
        text = (bundle / name).read_text()
        text = text.replace('import longcat_fixed_decode_dp4 as probe',
                            'from sglang.srt.debug_utils import longcat_fixed_decode_dp4 as probe')
        text = text.replace('from longcat_full_boundary import FullBoundaryTrace',
                            'from sglang.srt.debug_utils.longcat_full_boundary import FullBoundaryTrace')
        ast.parse(text)
        (root / 'debug_utils' / name).write_text(text)
    if a.full_boundary_layers is not None:
        profile = {'schema': 5, 'kind': 'full_valid_boundaries', 'num_layers': 38,
                   'layers': sorted(a.full_boundary_layers), 'attn_dp_size': dp_size,
                   'attn_tp_size': 16, 'tp_size': dp_size * 16, 'moe_ep_size': dp_size * 16,
                   'revision': REV}
        (root / 'debug_utils/longcat_capture_profile.json').write_text(json.dumps(profile, indent=2))
# The evaluator copy only relocates its two reports; sampling and scoring stay unchanged.
eval_source = (a.out / 'python/sglang/test/run_eval.py').read_text()
for ext in ('html', 'json'):
    eval_source = replace(eval_source, 'f"/tmp/{file_stem}.' + ext + '"',
                          'os.path.join(os.environ["LC_EVAL_OUTPUT_DIR"], f"{file_stem}.' + ext + '")')
ast.parse(eval_source)
(a.out / 'run_eval_to_dir.py').write_text(eval_source)
paths = [root / r for r in ('managers/tp_worker.py', 'models/longcat_flash.py',
    'hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py',
    'hardware_backend/npu/attention/ascend_backend.py')]
paths.append(a.out / 'run_eval_to_dir.py')
if a.mode == 'trace':
    paths += [root / 'debug_utils' / name for name in ('longcat_fixed_decode_dp4.py', 'capture_internal_dp4.py')]
if a.full_boundary_layers is not None:
    paths += [root / 'debug_utils/longcat_full_boundary.py', root / 'debug_utils/longcat_capture_profile.json']
receipt = {'mode': a.mode, 'topology': {'tp_size':16*dp_size, 'attn_dp_size':dp_size, 'moe_ep_size':16*dp_size}, 'capture': 'full_valid_boundaries' if a.full_boundary_layers is not None else 'legacy', 'internal_attention': a.mode == 'trace' and (a.full_boundary_layers is None or a.with_internal_attention), 'revision': REV, 'source': a.source, 'runtime': str(a.out.resolve()),
           'sha256': {str(f.relative_to(a.out)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
(a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
