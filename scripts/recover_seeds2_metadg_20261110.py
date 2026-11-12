"""Recovery for MetaDG runs of plan d6f66d20 (scripts/run_seeds2_20261110.py), frozen before any recovered result.

Failure (2026-11-10 13:39): PEMSBAY_MetaDG_seed42 stopped at the end of epoch 0 because the
non-finite-gradient step guard recorded 56 skipped optimiser steps, above the plan's limit of
1% of the epoch's batches (scripts/run_seeds_20261106.nonfinite_gradient_guard). The guard kept
every parameter finite (a skipped step changes neither parameters nor optimiser state); the
limit, inherited from recovery b136d9b6, was meant to separate a rare event from NaN
propagation, which the guard now prevents. The events are not in the released MetaDG code
path's control: one batch with a finite loss produces non-finite gradients in almost every
parameter (as in the single PEMS08 seed-32 event of b136d9b6).

Recovery (applies to every MetaDG run of plan d6f66d20, all datasets and seeds):
* targets: every MetaDG job of the base queue that failed with the guard limit
  ("AssertionError: ('nonfinite-gradient steps', n)" in its log), determined after the base
  queue has finished; a run that completed under the 1% limit is not rerun (the relaxed rule is
  a no-op for it); a MetaDG failure of any other kind is not covered (ATTENTION_REQUIRED);
* the same config file (same SHA256), seed, context (run_extension2_20261105.run_context) and
  guard; only the per-epoch limit changes from 1% to RECOVERY_EVENT_FRACTION (10%) of the epoch's
  batches, and every event is written with its epoch, batch index and parameter names;
* validation must be finite; the runner's skips are counted as in the base plan (>1% training
  skips flags divergence); fresh output root; no automatic retry; at most 5 concurrent processes
  (the recovery starts only after the base queue has no queued or running job).
Every run's guarded-step fraction is reported. Not in the released code; disclosed.
Modes: freeze | check | run | worker --key K | collect.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_seeds2_20261110 as base  # noqa: E402

NAME = 'seeds2_recovery_metadg_20261110'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
RECOVERY_EVENT_FRACTION = 0.10
MAX_CONCURRENT = base.MAX_CONCURRENT
GUARD_FAILURE = "AssertionError: ('nonfinite-gradient steps'"
FIRST_FAILURE = 'PEMSBAY_MetaDG_seed42'
AUTHORITY = ('2026-11-10 user (AskUserQuestion after the PEMSBAY_MetaDG_seed42 failure): '
             '"冻结 10% 恢复方案 (Recommended)"; the 10% limit was proposed after the 2.9% observation (disclosed).')


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def freeze():
    assert not PLAN.exists()
    bp = base.check()
    broot = base.output(bp)
    log = broot / f'{FIRST_FAILURE}.log'
    status = read(broot / 'queue_status.json')
    failed = next(j for j in status['jobs'] if j['key'] == FIRST_FAILURE)
    assert failed['status'] == 'failed' and GUARD_FAILURE in log.read_text()
    files = [Path(__file__), base.PLAN, log]
    p = dict(kind=NAME, authority=AUTHORITY, base_plan=str(base.PLAN.relative_to(ROOT)), base_plan_sha256=bp['plan_sha256'],
             observed=dict(key=FIRST_FAILURE, exit_code=failed.get('exit_code'), log=str(log.relative_to(ROOT)),
                           cause='56 guarded (skipped) optimiser steps in epoch 0 > 1% of its batches'),
             change=f'per-epoch guarded-step limit 1% -> {RECOVERY_EVENT_FRACTION:.0%}; every event recorded '
                    '(epoch, batch, parameters); validation must be finite; nothing else changes',
             targets='every MetaDG job of the base queue that failed with the guard limit (decided after the base '
                     'queue has finished); MetaDG runs completed under the 1% limit are not rerun',
             applies_to='every MetaDG run of the base plan, all datasets and seeds',
             status='authorized_training_then_one_test', max_concurrent=MAX_CONCURRENT,
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


def base_finished(bp):
    path = base.output(bp) / 'queue_status.json'
    return path.exists() and not any(j['status'] in ('queued', 'running') for j in read(path)['jobs'])


def targets(bp):
    """(guard-limit MetaDG failures to rerun, other MetaDG failures not covered)."""
    broot = base.output(bp)
    covered, other = [], []
    for j in read(broot / 'queue_status.json')['jobs']:
        if bp['configs'][j['key']]['model'] != 'MetaDG' or j['status'] == 'complete':
            continue
        log = broot / f"{j['key']}.log"
        (covered if j['status'] == 'failed' and log.exists() and GUARD_FAILURE in log.read_text()
         else other).append(j['key'])
    return covered, other


def relaxed_guard(training, events_path):
    """scripts/run_seeds_20261106.nonfinite_gradient_guard with the recovery limit and per-event batch records."""
    import torch
    original_optimizer, original_epoch = training.build_optimizer, training.train_one_epoch
    state = {'epoch': None, 'step': -1, 'events': []}

    def build_optimizer(session, *args, **kwargs):
        opt = original_optimizer(session, *args, **kwargs)
        names = {id(q): n for n, q in session.model.named_parameters()}
        step = opt.step

        def guarded_step(self_opt, closure=None):  # bound: LR schedulers wrap opt.step.__func__
            state['step'] += 1
            bad = [names.get(id(q), '?') for g in opt.param_groups for q in g['params']
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad:
                state['events'].append(dict(epoch=state['epoch'], optimiser_call=state['step'],
                                            parameters=bad[:20], count=len(bad)))
                write(events_path, state['events'])
                opt.zero_grad(set_to_none=True)
                return None
            return step(closure)
        opt.step = MethodType(guarded_step, opt)
        return opt

    def train_one_epoch(session, optimizer, epoch, *args, **kwargs):
        state.update(epoch=epoch, step=-1)
        before = len(state['events'])
        r = original_epoch(session, optimizer, epoch, *args, **kwargs)
        events = state['events'][before:]
        assert len(events) <= RECOVERY_EVENT_FRACTION * len(session.train_loader), \
            ('nonfinite-gradient steps', len(events))
        r['nonfinite_gradient_steps'] = events
        r['nonfinite_gradient_fraction'] = len(events) / len(session.train_loader)
        return r
    stack = ExitStack()
    stack.enter_context(patch.object(training, 'build_optimizer', build_optimizer))
    stack.enter_context(patch.object(training, 'train_one_epoch', train_one_epoch))
    return stack


def worker(key):
    import math
    import torch
    from scripts import train_v4 as training
    torch.set_num_threads(4)
    p = check()
    bp = base.check()
    root = output(p)
    assert key in targets(bp)[0] and not (root / f'{key}_complete.json').exists()
    row = bp['configs'][key]
    with base.context(row['model'], row['study_context']), relaxed_guard(training, root / f'{key}_events.json'):
        r = training.train(ROOT / row['path'], root / 'runs' / key, device='cuda:0', seed=row['seed'],
                           num_workers=0, plan=bp, save=True, collect_development_table=True)
    table = r['development_validation_table']
    assert all(math.isfinite(v) for g in ('main', 'phase') for c in table[g].values()
               for f, v in c.items() if f in ('MAE', 'RMSE', 'MAPE'))
    write(root / f'{key}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


def run():
    p = check()
    bp = base.check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not base_finished(bp):
        write(root / 'waiting.json', dict(updated=time.time(), reason='base queue still has queued/running jobs'))
        time.sleep(120)
    covered, other = targets(bp)
    write(root / 'targets.json', dict(covered=covered, not_covered=other, decided=time.time()))
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    jobs = [dict(key=k, status='queued') for k in covered]
    active, last = {}, 0.
    while any(j['status'] in ('queued', 'running') for j in jobs):
        for key, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                j = next(j for j in jobs if j['key'] == key)
                j.update(status='complete' if code == 0 and (root / f'{key}_complete.json').exists() else 'failed',
                         exit_code=code, finished=time.time())
                log.close()
                del active[key]
        pending = next((j for j in jobs if j['status'] == 'queued'), None)
        if pending and len(active) < MAX_CONCURRENT and time.monotonic() - last > 60 and \
                base.extension.gpu_free_gib() > 30:
            key = pending['key']
            log = (root / f'{key}.log').open('x')
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            pending.update(status='running', pid=proc.pid, started=time.time(), log=str((root / f'{key}.log').relative_to(ROOT)),
                           command=f'{sys.executable} -u scripts/{Path(__file__).name} worker --key {key}')
            active[key] = proc, log
            last = time.monotonic()
        write(root / 'queue_status.json', dict(plan_sha256=p['plan_sha256'], queue_pid=os.getpid(), jobs=jobs,
                                               updated=time.time(), max_concurrent=MAX_CONCURRENT))
        time.sleep(20)
    collect()


def collect():
    p = check()
    bp = base.check()
    root = output(p)
    t = read(root / 'targets.json')
    rows, missing, flags = {}, [], []
    for key in t['covered']:
        e = bp['configs'][key]
        done = root / f'{key}_complete.json'
        if not done.exists():
            missing.append(key)
            continue
        report_path = Path(read(done)['report_path'])
        r = read(report_path)
        assert r['plan_sha256'] == bp['plan_sha256'] and r['config_sha256'] == e['sha256'] and r['seed'] == e['seed']
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history'])[1] == r['best_epoch']
        train_skipped = sum(h['train']['skipped'] for h in r['history'])
        train_batches = sum(h['train']['batches'] + h['train']['skipped'] for h in r['history'])
        if train_skipped > 0.01 * train_batches:
            flags.append(dict(key=key, train_skipped=train_skipped, train_batches=train_batches))
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'], train_skipped=train_skipped,
                         nonfinite_gradient_steps=[g for h in r['history'] for g in h['train'].get('nonfinite_gradient_steps', [])],
                         nonfinite_gradient_fraction_per_epoch=[h['train'].get('nonfinite_gradient_fraction') for h in r['history']],
                         validation_skipped=sorted({h['validation']['skipped'] for h in r['history']}),
                         recovery_plan_sha256=p['plan_sha256'])
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], base_plan_sha256=bp['plan_sha256'],
                                                 rows=rows, missing=missing, flags=flags,
                                                 not_covered=t['not_covered'], test_accessed=False))
    write(root / 'COMPLETE.json', dict(passed=not missing and not flags and not t['not_covered'], runs=len(rows),
                                       missing=missing, flags=flags, not_covered=t['not_covered'], test_accessed=False))
    print('recovery collected', len(rows), 'missing', missing, 'flags', flags, 'not covered', t['not_covered'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'run', 'worker', 'collect'])
    parser.add_argument('--key')
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.key)
    elif a.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[a.mode]()
