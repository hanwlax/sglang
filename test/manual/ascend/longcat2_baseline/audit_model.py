import collections,json,re,struct,sys
from pathlib import Path
root=Path(sys.argv[1]); c=json.loads((root/'config.json').read_text())
idx=json.loads((root/'model.safetensors.index.json').read_text())
wm=idx['weight_map']; sizes=collections.Counter(); counts=collections.Counter(); layers=set(); examples={}
for name in sorted(set(wm.values())):
 with (root/name).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n))
 for key,t in header.items():
  if key=='__metadata__' or key not in wm or wm[key]!=name: continue
  m=re.match(r'model.layers.(\d+)\.',key)
  if m: layers.add(int(m[1]))
  cat=('mtp' if '.mtp.' in key else 'experts' if '.experts.' in key else 'attention' if '.self_attn.' in key else 'dense_ffn' if '.mlps.' in key else 'embedding' if 'embed' in key or 'oe_' in key else 'other')
  sizes[cat]+=t['data_offsets'][1]-t['data_offsets'][0]; counts[cat]+=1
  examples.setdefault(cat,{'name':key,'dtype':t['dtype'],'shape':t['shape']})
print(json.dumps({'model':str(root),'config':{k:c.get(k) for k in ['num_layers','num_hidden_layers','hidden_size','disable_quant_module']},'index_layers':sorted(layers),'bytes_by_category':sizes,'GiB_by_category':{k:round(v/2**30,4) for k,v in sizes.items()},'tensor_counts':counts,'examples':examples},indent=2))
