import sys, io, contextlib, torch; sys.path.insert(0,'.')
from args import load_args_from_config
from data_provider.data_factory import data_provider
from model.regist_model import load_model_class
dev="cuda:0"
with contextlib.redirect_stdout(io.StringIO()):
    a = load_args_from_config("./configs/ALFNet/rf_l3s.yaml", overrides={"device": dev})
a.lcbt_manifest_root="./_smoke/manifests"; a.ples_manifest_root="./_smoke/manifests"
a.num_workers=0; a.lcbt_overwrite_manifest=True
with contextlib.redirect_stdout(io.StringIO()):
    tl = data_provider(a,"train"); m = load_model_class(a.model)(a).float().to(dev)
b=[t.to(dev) for t in next(iter(tl))]
adj,_=tl.dataset.get_adj_and_pos_mark(); adj=torch.as_tensor(adj,dtype=torch.float32,device=dev)
x=torch.nan_to_num(b[0].float(),nan=0.0); y=torch.nan_to_num(b[1].float()[:,-a.pred_len:],nan=0.0)
out=m(adj,x,b[2].float(),None,y,b[3].float(),0)
sel=(b[7][:,None,:,None]>0.5).expand_as(y)&torch.isfinite(y)
loss=torch.nn.functional.mse_loss(out["pred"][sel],y[sel])
from utils.model_output import sum_aux_losses
aux=sum_aux_losses(out.get("aux_losses"),device=y.device)
(loss+(aux if aux is not None else 0)).backward()
zero=[(n,tuple(p.shape)) for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().sum()==0]
none=[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None]
print(f"零梯度参数 {len(zero)} 个:")
for n,s in zero[:12]: print(f"   {n} {s}")
print(f"无梯度(None) {len(none)} 个: {none[:6]}")
# 该 batch 的角色分布
role=x[:,-1,:,1:5].argmax(-1)
mask=b[7]>0.5
import collections
print("\n本 batch 被监督节点的角色分布:", dict(collections.Counter(role[mask].tolist())), "(0=AAS 1=NAS 2=VS 3=FS)")
