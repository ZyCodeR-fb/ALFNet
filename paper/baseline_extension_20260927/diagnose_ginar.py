"""Bounded train/validation-only numerical diagnosis; not a formal accuracy run."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from experiments.v4_runner import build_v4_session,to_device,compute_losses,validate
from scripts.train_v4 import build_optimizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='PEMS03');p.add_argument('--lr',type=float,default=.006);p.add_argument('--batches',type=int,default=10000)
    a=p.parse_args();torch.set_num_threads(4)
    s=build_v4_session(ROOT/f'configs/paper/baseline_extension_20260927/{a.dataset}_GinAR.yaml',device='cuda:0',run_seed=32,num_workers=0)
    s.args.lr=a.lr;opt=build_optimizer(s);s.set_epoch(0);s.model.train()
    result={'dataset':a.dataset,'lr':a.lr,'test_accessed':False,'steps':[]};start=time.time()
    for i,raw in enumerate(s.train_loader):
        if i>=a.batches:break
        batch=to_device(raw,s.device);opt.zero_grad(set_to_none=True)
        pred=s.model.forward_v4(batch,s.train_context)['pred']
        legal=batch['loss_mask'].unsqueeze(-1)
        loss=compute_losses({'pred':pred},batch,s.args.loss,scaler=s.scaler)['optimize']
        if loss is None or not torch.isfinite(loss) or not torch.isfinite(pred[legal]).all():
            result.update(failed_at=i,reason='nonfinite_prediction_or_loss',finite_prediction_fraction=float(torch.isfinite(pred).float().mean()));break
        loss.backward()
        bad=[n for n,v in s.model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
        norm=torch.nn.utils.clip_grad_norm_(s.model.parameters(),5.)
        if bad or not torch.isfinite(norm):
            result.update(failed_at=i,reason='nonfinite_gradient' if bad else 'nonfinite_norm',bad_gradients=bad,gradient_norm=str(float(norm)),loss=float(loss.detach()));break
        opt.step()
        if i%25==0:
            row=dict(batch=i,loss=float(loss.detach()),gradient_norm=float(norm),pred_abs_max=float(pred.abs().max()))
            result['steps'].append(row);print(row,flush=True)
    else:
        result['epoch_complete']=True
    if result.get('epoch_complete'):
        vr,table=validate(s,return_cell_table=True);result.update(validation=vr,table=table)
    result['seconds']=time.time()-start
    path=ROOT/f'paper/baseline_extension_20260927/diagnosis_{a.dataset}_{a.lr}.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
