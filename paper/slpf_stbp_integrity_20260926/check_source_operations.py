import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
root=Path.cwd();sys.path.insert(0,str(root));torch.set_num_threads(1);torch.manual_seed(32)
def module(path,name):
 s=importlib.util.spec_from_file_location(name,root/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
src=module('references/STBP/src/model/model.py','original_stbp')
from model.comparison_model.STBP.stbp import STBP_Model
n=6;adj=torch.eye(n)+.2;pos=torch.randn(n,2)
a=SimpleNamespace(dropout=0.,y_len=12,base_node_size=n,year=2020,model={'in_channel':12,'hidden_channel':8,'out_channel':12},logger=SimpleNamespace(info=lambda *a:None))
adapt=STBP_Model(a,adj,pos).eval();original=src.STBP_Model(a).eval()
s={k:v for k,v in adapt.state_dict().items() if not k.startswith('pattern_encoder.')};s['pattern_bank']=adapt.pattern_encoder().detach().clone();original.load_state_dict(s,strict=True)
x=torch.randn(2*n,12);actual=adapt(SimpleNamespace(x=x),adj);expected=original(SimpleNamespace(x=x),adj)
assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
results={'scope':'CPU synthetic source-operation parity conditional on identical supplied embeddings; not complete-method or training parity','STBP_max_abs':float((actual-expected).abs().max().detach())}
src=module('references/SLPF/model/SLPF.py','original_slpf')
from model.comparison_model.SLPF.SLPF import SLPF
args=SimpleNamespace(num_locs=n,num_unsensed_locs=2,lag=12,horizon=12,input_dim=4,embed_dim=8,num_layer=1)
adapt=SLPF([0,1,2,3],[4,5],adj,args,pos).eval();original=src.SLPF([0,1,2,3],[4,5],adj,args).eval()
s={k:v for k,v in adapt.state_dict().items() if not k.startswith('full_locs_encoder.')};s['full_locs_emb']=adapt.full_locs_emb.detach().clone();original.load_state_dict(s,strict=True)
def features(nodes):
 x=torch.randn(2,12,nodes,4);x[...,1]=torch.arange(12)[None,:,None];x[...,2]=2;x[...,3]=torch.arange(nodes)[None,None,:];return x
for stage in ['adp','forecast','agg']:
 adapt.stage=original.stage=stage
 xs=[features(4 if stage=='adp' else n)]
 if stage=='agg':xs.append(features(4))
 actual=adapt(*[x.clone() for x in xs]);expected=original(*[x.clone() for x in xs]);assert torch.isfinite(actual).all();assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
 results['SLPF_'+stage]={'shape':list(actual.shape),'max_abs':float((actual-expected).abs().max().detach())}
results['training_started']=False;results['real_dataset_read']=False
print(json.dumps(results,indent=2))
Path(__file__).with_name('verification.json').write_text(json.dumps(results,indent=2)+'\n')
