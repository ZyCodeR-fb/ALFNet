"""Recovery for 07_PEMS08_MetaDG of plan a8d285aa (2026-11-05), frozen before any recovered result.

Failure: during epoch 0 the common runner skipped at least one training batch
(non-finite loss; the PEMS08 training design has no unsupervised batch of 16), and
the worker's count-check wrapper, which allows no skipped batch, aborted. The
20-batch preflight had passed; METR-LA (same released block), PEMS-BAY and PEMS04
MetaDG ran without any skip under the strict rule.

Recovery (applies identically to every MetaDG dataset; a no-op for runs without a
skip, so only PEMS08 is rerun):
* a training batch may be skipped -- the common runner's own behaviour: no update
  -- only if its loss is non-finite or it has no supervision; every skip is
  recorded with epoch, batch index and the first submodule whose output became
  non-finite (diagnosed by re-running that batch's forward with hooks);
* the run fails if skips in any epoch exceed 1% of that epoch's batches;
* validation must skip nothing and the 33-cell table must be finite.
Same config file and hash as plan a8d285aa; fresh output root; no automatic retry.
Waits until this task has fewer than 4 running processes (base queue + MCPST
recovery + this run).
"""
import argparse
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_extension2_20261105 as base  # noqa: E402
from scripts import recover_extension2_20261105 as first_recovery  # noqa: E402

NAME = 'extension2_recovery2_20261105'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
KEY = '07_PEMS08_MetaDG'
MAX_SKIP_FRACTION = 0.01


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def freeze():
    assert not PLAN.exists()
    bp = base.check()
    broot = base.output(bp)
    row = bp['configs'][KEY]
    log = broot / f'{KEY}.log'
    failed = next(j for j in read(broot / 'queue_status.json')['jobs'] if j['key'] == KEY)
    assert failed['status'] == 'failed' and 'training skipped an invalid batch' in log.read_text()
    files = [Path(__file__), base.PLAN, log, ROOT / row['path'], first_recovery.PLAN]
    p = dict(kind=NAME, base_plan=str(base.PLAN.relative_to(ROOT)), base_plan_sha256=bp['plan_sha256'],
             failed_run=dict(key=KEY, exit_code=failed.get('exit_code'), log=str(log.relative_to(ROOT)),
                             cause='non-finite training loss in epoch 0; strict wrapper allows no skipped batch'),
             change='common-runner skip of non-finite/unsupervised training batches allowed and recorded; '
                    f'fail if any epoch skips more than {MAX_SKIP_FRACTION:.0%} of its batches; validation strict',
             applies_to='every MetaDG dataset; no-op for runs without a skip, so only PEMS08 is rerun',
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


def first_nonfinite_module(session, batch):
    import torch
    found = []

    def hook(name):
        def fn(module, args, out):
            if found:
                return
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            if any(torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all() for t in tensors):
                found.append(name)
        return fn
    import numpy as np
    # The re-run must not perturb training: restore every RNG and the curriculum counter.
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             np.random.get_state(), session.model.batches_seen.clone())
    session.model.batches_seen -= 1
    handles = [m.register_forward_hook(hook(n)) for n, m in session.model.named_modules() if n]
    try:
        with torch.no_grad():
            session.model.forward_v4(batch, session.train_context)
    finally:
        for h in handles:
            h.remove()
        torch.set_rng_state(saved[0])
        if saved[1] is not None:
            torch.cuda.set_rng_state_all(saved[1])
        np.random.set_state(saved[2])
        session.model.batches_seen.copy_(saved[3])
    return found[0] if found else 'not reproduced (dropout/curriculum draw differs on re-run)'


def tolerant_train(p, root, log_path):
    import torch
    from experiments import v4_runner
    from scripts import train_v4 as training
    row = p['configs'][KEY]
    reference = p['anchors'][row['dataset']]['table']
    original_losses, original_train, original_validate = v4_runner.compute_losses, training.train_one_epoch, training.validate
    state = {'epoch': None, 'batch': -1, 'session': None, 'events': []}

    def losses(output, batch, loss_name='mse', scaler=None):
        result = original_losses(output, batch, loss_name, scaler=scaler)
        if state['session'] is not None and state['session'].model.training:
            state['batch'] += 1
            bad = result['optimize'] is None or not torch.isfinite(result['optimize'])
            if bad:
                event = dict(epoch=state['epoch'], batch=state['batch'],
                             supervised=bool(batch['loss_mask'].any()),
                             nonfinite_prediction_fraction=float((~torch.isfinite(result['prediction'])).float().mean()))
                event['first_nonfinite_module'] = first_nonfinite_module(state['session'], batch)
                state['session'].model.train()
                state['events'].append(event)
                write(log_path, state['events'])
        return result

    def epoch(session, optimizer, epoch_index, *args, **kwargs):
        state.update(epoch=epoch_index, batch=-1, session=session)
        before = len(state['events'])
        r = original_train(session, optimizer, epoch_index, *args, **kwargs)
        skipped = len(state['events']) - before
        assert r['skipped'] == skipped, ('runner/recorder skip counts differ', r['skipped'], skipped)
        assert skipped <= MAX_SKIP_FRACTION * (r['batches'] + r['skipped']), ('too many skipped batches', skipped)
        r['recovery_skipped_events'] = state['events'][before:]
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
            patch.object(training, 'validate', validate):
        return training.train(ROOT / row['path'], root, device='cuda:0', seed=32, num_workers=0, plan=p,
                              save=True, collect_development_table=True)


def busy():
    bp = base.check()
    running = sum(j['status'] == 'running' for j in read(base.output(bp) / 'queue_status.json')['jobs'])
    rp = first_recovery.check()
    rroot = first_recovery.output(rp)
    recovery_running = (rroot / 'started.json').exists() and not (rroot / f'{first_recovery.KEY}_complete.json').exists()
    if recovery_running:
        pid = read(rroot / 'started.json')['pid']
        recovery_running = Path(f'/proc/{pid}').exists()
    return running + int(recovery_running)


def run():
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    while True:
        n = busy()
        write(root / 'waiting.json', dict(task_processes_running=n, updated=time.time()))
        if n < 4:
            break
        time.sleep(60)
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    with base.run_context():
        r = tolerant_train(p, root / 'runs' / KEY, root / 'skip_events.json')
    write(root / f'{KEY}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'run'])
    a = parser.parse_args()
    print(check()['plan_sha256']) if a.mode == 'check' else globals()[a.mode]()
