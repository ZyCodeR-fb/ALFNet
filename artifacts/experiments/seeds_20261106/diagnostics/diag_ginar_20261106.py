"""Diagnostic only (writes nothing to the repo): GinAR first 20 training batches per seed."""
import sys, json
from types import MethodType
from unittest.mock import patch
ROOT = '/media/xd/38162e9d-06bd-4953-8790-0bc340486a80/ZMY/ALFNet'
sys.path.insert(0, ROOT)
import os; os.chdir(ROOT)
import torch
torch.set_num_threads(4)
from scripts import run_seeds_20261106 as r
from scripts import train_v4 as training
p = r.check()
results = {}
for ds, seed in [(a, b) for a in sys.argv[1].split(',') for b in map(int, sys.argv[2].split(','))]:
    events = []
    orig = training.build_optimizer
    def build(session):
        opt = orig(session)
        step = opt.step
        names = {id(q): n for n, q in session.model.named_parameters()}
        def guarded(self, closure=None):
            bad = [names[id(q)] for g in opt.param_groups for q in g['params'] if q.grad is not None and not torch.isfinite(q.grad).all()]
            norm = float(torch.norm(torch.stack([q.grad.detach().norm() for g in opt.param_groups for q in g['params'] if q.grad is not None])))
            out = step(closure)
            pbad = [n for n, q in session.model.named_parameters() if not torch.isfinite(q).all()]
            events.append(dict(step=len(events), grad_nonfinite=bad[:5], n_bad=len(bad), clipped_norm=norm, params_nonfinite_after=len(pbad)))
            return out
        opt.step = MethodType(guarded, opt)
        return opt
    try:
        with patch.object(training, 'build_optimizer', build):
            training.train(ROOT + '/' + p['configs'][f'{ds}_GinAR_seed{seed}']['path'], None, device='cuda:0', seed=seed,
                           num_workers=0, plan=p, save=False, collect_development_table=False, max_epochs=1,
                           max_train_batches=20, max_validation_batches=4)
        status = 'ok'
    except Exception as e:
        status = repr(e)[:120]
    first = next((e for e in events if e['n_bad'] or e['params_nonfinite_after']), None)
    print(ds, seed, status, 'steps', len(events), 'first_bad', first, flush=True)
