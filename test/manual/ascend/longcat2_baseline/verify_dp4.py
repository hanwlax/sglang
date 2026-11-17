#!/usr/bin/env python3
"""CPU-only tests for DP4 mapping, capture gating, completeness and eval separation."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

import longcat_fixed_decode_dp4 as probe
from capture_internal_dp4 import capture_internal
from compare_fixed_decode_dp4 import check_workers, difference, load
from check_server_dp4 import validate

root = Path(__file__).resolve().parent
for f in root.glob('*dp4*.py'):
    ast.parse(f.read_text(), str(f))
for f in ('run_a3_full_baseline.sh','run_gsm8k.sh','run_fixed_prefix_dp4.sh'):
    subprocess.run(['bash','-n',str(root/f)],check=True)

def fails(call):
    try:
        call()
    except (ValueError,RuntimeError):
        return
    raise AssertionError('Expected rejection')

def topo(rank):
    return dict(tp_rank=rank,tp_size=64,attn_dp_rank=rank//16,attn_dp_size=4,
                attn_tp_rank=rank%16,attn_tp_size=16,attn_cp_size=1,
                moe_ep_rank=rank,moe_ep_size=64,moe_tp_size=1,moe_dp_size=1,pp_size=1)

# Unequal padded DP counts catch hard-coded DP0/DP1 offsets.
counts=[1008,1024,1040,1056]
for dp in range(4):
    p,logical,live,padding=probe.select_rows(set(),counts,dp,0,852,sum(counts))
    assert p==[sum(counts[:dp])+i for i in (0,426,851)] and logical==[0,426,851]
    all_live=[]
    for rank in range(16):
        n=counts[dp]//16
        physical,logical,live,pad=probe.select_rows({'ATTN_DP','ATTN_TP'},counts,dp,rank,852,n)
        assert all(0<=i<n for i in physical) and all(i<852 for i in logical)
        all_live.append(live)
    assert sum(all_live)==852
    for rank in range(16):
        physical,logical,live,pad=probe.select_rows({'ATTN_DP','ATTN_TP'},[16]*4,dp,rank,1,1)
        assert live==int(rank==0)
fails(lambda:probe.select_rows(set(),[1024]*2,1,0,852,2048))

with tempfile.TemporaryDirectory(prefix='lc-dp4-verify-') as tmp:
    tmp=Path(tmp)
    os.environ['LONGCAT_FIXED_DECODE_DIR']=str(tmp)
    os.environ['LC_INTERNAL_LAYERS']='0,1,37'
    os.environ['LC_INTERNAL_RANKS']='0,16,32,48'
    for dp in range(4):
        probe._active={'rid':f'lcfix4-0123456789ab-d{dp}-p852-t70115-r1',
                       'dp':dp,'mode':'EXTEND','topology':topo(dp*16)}
        x=torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
        capture_internal(0,'test',x=x,nested=[None,{'y':x}])
        x.zero_()
    files=list((tmp/'internal').glob('*.pt')); assert len(files)==4
    for f in files:
        data=torch.load(f,weights_only=True)
        assert data['values']['x'].sum()>0 and data['topology']['attn_dp_rank']==data['dp']
    probe._active['rid']=probe._active['rid'][:-1]+'2'
    capture_internal(0,'skip_repeat',x=x)
    probe._active=None; capture_internal(0,'skip_idle',x=x)
    assert len(list((tmp/'internal').glob('*.pt')))==4

    dump=tmp/'dumps'; dump.mkdir()
    for rank in range(64):
        (dump/f'worker.tp{rank}.json').write_text(json.dumps(topo(rank)))
    check_workers(dump)
    last=dump/'worker.tp63.json'; saved=last.read_text(); last.unlink()
    fails(lambda:check_workers(dump)); last.write_text(saved)
    bad=topo(63); bad['attn_dp_rank']=1
    last.write_text(json.dumps(bad)); fails(lambda:check_workers(dump)); last.write_text(saved)

    # Full 38-layer schema through the real comparator, four DPs, both modes.
    points=['attn0.input','attn0.local_output','moe.input','router.logits',
            'router.expert_ids','router.weights','moe.experts_output','moe.reduced',
            'moe.branch_output','ffn0.input','ffn0.local_output','attn1.input',
            'attn1.local_output','ffn1.input','ffn1.local_output','layer.output']
    keys=[(-1,'embedding.output')]+[(l,p) for l in range(38) for p in points]+[
        (-1,'final_norm.output'),(-1,'lm_head.logits')]
    prompt=[1]*852; forced=70115
    manifest={'schema':4,'dps':[0,1,2,3],'lengths':[1],'repeat':2,
              'prompt_ids':prompt,'prefix_ids':[forced],'requests':[]}
    for dp in range(4):
        for rep in (1,2):
            item={'rid':f'lcfix4-0123456789ab-d{dp}-p852-t{forced}-r{rep}',
                  'dp':dp,'rep':rep,'length':1,'warmup':False,'status':'ok'}
            manifest['requests'].append(item)
            for mode in ('EXTEND','DECODE'):
                n=852 if mode=='EXTEND' else 1
                raw=[0]*4; raw[dp]=n
                padded=[1024]*4 if mode=='EXTEND' else [16]*4
                for rank in range(dp*16,(dp+1)*16):
                    samples=[]
                    for layer,point in keys:
                        is_logits=point=='lm_head.logits'
                        shape=[1 if is_logits else padded[dp],2]
                        mapping=([0],[n-1],1,0) if is_logits else probe.select_rows(
                            {'ATTN_DP'},padded,dp,rank%16,n,shape[0])
                        ph,lr,live,pad=mapping
                        values=torch.ones((len(lr),2),dtype=torch.bfloat16)
                        samples.append(dict(layer=layer,point=point,shape=shape,dtype=str(values.dtype),
                            axes=['ATTN_DP'],physical_rows=ph,rows=lr,live_rows=live,
                            padding_rows=pad,status='sampled',values=values))
                    payload=dict(schema=4,rid=item['rid'],tp=rank,dp=dp,mode=mode,topology=topo(rank),
                                 input_ids=prompt if mode=='EXTEND' else [forced],
                                 positions=list(range(852)) if mode=='EXTEND' else [852],
                                 num_layers=38,raw_counts=raw,padded_counts=padded,samples=samples)
                    torch.save(payload,dump/f"{item['rid']}.tp{rank}.{mode}.pt")
    mp=tmp/'manifest.json'; mp.write_text(json.dumps(manifest))
    model=tmp/'model'; model.mkdir(); (model/'config.json').write_text('{"num_layers":38}')
    out=tmp/'comparison.json'
    command=[sys.executable,str(root/'compare_fixed_decode_dp4.py'),'--manifest',str(mp),
             '--dump-dir',str(dump),'--model',str(model),'--out',str(out)]
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    data=json.loads(out.read_text())
    assert sum('summary' in r for r in data)==8
    cross=[r['cross_dp_final'] for r in data if 'cross_dp_final' in r]
    assert len(cross)==12 and {r['other_dp'] for r in cross}=={1,2,3}
    assert all(r['difference'] is None for r in cross)
    item=manifest['requests'][-1]
    path=dump/f"{item['rid']}.tp63.DECODE.pt"
    payload=torch.load(path,weights_only=True); payload['dp']=1; torch.save(payload,path)
    fails(lambda:load(dump,item,'DECODE',63,prompt,forced,38))
    assert difference(torch.tensor([1.,2.]),torch.tensor([1.,3.]))['first_flat_index']==1

good=dict(tp_size=64,attn_dp_size=4,ep_size=64,moe_a2a_backend='deepep',deepep_mode='auto',
          cuda_graph_backend_prefill='disabled',cuda_graph_backend_decode='disabled',
          speculative_algorithm=None,json_model_override_args='{}',chunked_prefill_size=1024)
validate(good,True)
fails(lambda:validate({**good,'ep_size':32}))
fails(lambda:validate({**good,'speculative_algorithm':'NEXTN'}))
fails(lambda:validate({**good,'json_model_override_args':'{"num_hidden_layers":2}'}))

for mode in ('eval','trace'):
    runtime=root/f'runtime_dp4_{mode}'
    r=json.loads((runtime/'receipt.json').read_text()); assert r['mode']==mode
    text=(runtime/'python/sglang/srt/managers/tp_worker.py').read_text()
    assert ('bind_fixed_decode' in text)==(mode=='trace')
    e=(runtime/'run_eval_to_dir.py').read_text()
    original=(runtime/'python/sglang/test/run_eval.py').read_text()
    for ext in ('html','json'):
        original=original.replace('f"/tmp/{file_stem}.'+ext+'"',
            'os.path.join(os.environ["LC_EVAL_OUTPUT_DIR"], f"{file_stem}.'+ext+'")')
    assert e==original
print('CPU_CHECK_PASS DP0-3 offsets/padding; 64-worker validation; 38-layer comparator; internal snapshots; clean/trace separation; eval scorer unchanged')
