"""Second recovery for 07_PEMS08_MetaDG (2026-11-06), frozen before any recovered result.

Recovery 4f1cea2f (allow the runner's skip of non-finite batches) showed genuine
divergence: from epoch-0 batch 138 every prediction was non-finite and the first
non-finite module was ``core.tod_embedding``, i.e. the parameters were already NaN.
A batch with a finite loss had produced a non-finite gradient; clipping a non-finite
total norm and the Adam step then wrote NaN into every parameter. The run was
stopped by the agent (STOPPED.json) because its own 1% rule could no longer pass.

Recovery (applies to every MetaDG dataset):
* before every optimiser step, all gradients are checked; if any is non-finite the
  step is skipped (no parameter or optimiser-state change, as an AMP grad-scaler
  skip) and the event is recorded with the offending parameter names;
* non-finite or unsupervised losses are skipped by the common runner as before and
  recorded;
* the run fails if the guarded events of any epoch exceed 1% of its batches;
* validation must skip nothing and its 33-cell table must be finite.
For METR-LA and PEMS04 (finished) and PEMS-BAY (running under the strict rule) the
guard is a no-op: any non-finite gradient there would have turned every later loss
non-finite and the strict rule would have aborted. They are not repeated; if the
PEMS-BAY run fails it is rerun under this rule. The released code has no such
guard; this is an adaptation, disclosed.
Same config file and hash as plan a8d285aa; fresh output root; no automatic retry.
A first freeze of this recovery (d74f0c3f) crashed while constructing the LR scheduler,
before any training step (the guarded step was not a bound method); it is superseded.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import MethodType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_extension2_20261105 as base  # noqa: E402
from scripts import recover2_extension2_20261105 as previous  # noqa: E402

NAME = 'extension2_recovery3_20261106'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
KEY = previous.KEY
MAX_EVENT_FRACTION = 0.01


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def freeze():
    assert not PLAN.exists()
    bp = base.check()
    pp = previous.check()
    proot = previous.output(pp)
    stopped = read(proot / 'STOPPED.json')
    events = read(proot / 'skip_events_at_stop.json')
    assert events and all(e['first_nonfinite_module'] == 'core.tod_embedding' for e in events)
    row = bp['configs'][KEY]
    files = [Path(__file__), base.PLAN, previous.PLAN, proot / 'STOPPED.json', proot / 'skip_events_at_stop.json',
             ROOT / row['path']]
    p = dict(kind=NAME, base_plan_sha256=bp['plan_sha256'], previous_recovery_sha256=pp['plan_sha256'],
             observed=dict(first_skipped_batch=events[0]['batch'], skipped_until_stop=stopped['skip_events'],
                           first_nonfinite_module='core.tod_embedding (parameters already NaN)'),
             change='skip optimiser steps whose gradients are not all finite (recorded); runner skips of '
                    f'non-finite losses recorded; fail if events exceed {MAX_EVENT_FRACTION:.0%} of an epoch',
             applies_to='every MetaDG dataset; no-op for runs that finished under the strict rule',
             configs={KEY: row}, anchors={row['dataset']: bp['anchors'][row['dataset']]}, seed=32,
             status='authorized_training_validation', max_concurrent_with_base=4,
             files={str(f.relative_to(ROOT)): sha(f) for f in files} | bp['files'])
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'])


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    return p


def guarded_train(p, root, log_path):
    import torch
    from experiments import v4_runner
    from scripts import train_v4 as training
    row = p['configs'][KEY]
    reference = p['anchors'][row['dataset']]['table']
    original_losses, original_train, original_validate = v4_runner.compute_losses, training.train_one_epoch, training.validate
    original_optimizer = training.build_optimizer
    state = {'epoch': None, 'batch': -1, 'events': [], 'names': None}

    def record(event):
        state['events'].append(event)
        write(log_path, state['events'])

    def build_optimizer(session):
        opt = original_optimizer(session)
        state['names'] = {id(q): n for n, q in session.model.named_parameters()}
        step = opt.step

        def guarded_step(self_opt, closure=None):  # bound below: LR schedulers wrap opt.step.__func__
            bad = [state['names'].get(id(q), '?') for g in opt.param_groups for q in g['params']
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad:
                record(dict(kind='nonfinite_gradient', epoch=state['epoch'], batch=state['batch'],
                            parameters=bad[:20], count=len(bad)))
                opt.zero_grad(set_to_none=True)
                return None
            return step(closure)
        opt.step = MethodType(guarded_step, opt)
        return opt

    def losses(output, batch, loss_name='mse', scaler=None):
        result = original_losses(output, batch, loss_name, scaler=scaler)
        if state['epoch'] is not None and torch.is_grad_enabled():
            state['batch'] += 1
            if result['optimize'] is None or not torch.isfinite(result['optimize']):
                record(dict(kind='nonfinite_or_unsupervised_loss', epoch=state['epoch'], batch=state['batch'],
                            supervised=bool(batch['loss_mask'].any())))
        return result

    def epoch(session, optimizer, epoch_index, *args, **kwargs):
        state.update(epoch=epoch_index, batch=-1)
        before = len(state['events'])
        r = original_train(session, optimizer, epoch_index, *args, **kwargs)
        events = state['events'][before:]
        losses_skipped = sum(e['kind'] == 'nonfinite_or_unsupervised_loss' for e in events)
        assert r['skipped'] == losses_skipped, ('runner/recorder skip counts differ', r['skipped'], losses_skipped)
        assert len(events) <= MAX_EVENT_FRACTION * (r['batches'] + r['skipped']), ('too many guarded events', len(events))
        r['recovery_events'] = events
        return r

    def validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        r = result[0] if isinstance(result, tuple) else result
        assert r['skipped'] == 0, 'validation skipped a batch'
        if isinstance(result, tuple):
            for g in ('main', 'phase'):
                for c, values in reference[g].items():
                    for field, v in values.items():
                        if field.endswith('_count'):
                            assert result[1][g][c][field] == v, (g, c, field)
        return result
    with patch.object(v4_runner, 'compute_losses', losses), patch.object(training, 'train_one_epoch', epoch), \
            patch.object(training, 'validate', validate), patch.object(training, 'build_optimizer', build_optimizer):
        return training.train(ROOT / row['path'], root, device='cuda:0', seed=32, num_workers=0, plan=p,
                              save=True, collect_development_table=True)


def run():
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    while True:
        n = previous.busy()
        write(root / 'waiting.json', dict(task_processes_running=n, updated=time.time()))
        if n < 4:
            break
        time.sleep(60)
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    with base.run_context():
        r = guarded_train(p, root / 'runs' / KEY, root / 'guard_events.json')
    write(root / f'{KEY}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'run'])
    a = parser.parse_args()
    print(check()['plan_sha256']) if a.mode == 'check' else globals()[a.mode]()
