"""Recovery for 00_METRLA_MCPST of plan a8d285aa (2026-11-05), frozen before any recovered result.

Failure: the worker's count-check wrapper asserts that validation skips no batch.
With MCPST's batch 8, the METR-LA validation design contains two batches (indices
745 and 832) in which none of the eight samples has a supervised position, so the
common runner skips them (nothing to score) and the wrapper aborted at epoch 0.
The other three MCPST datasets have no such batch. The model, config, data,
optimiser, selection and 33-cell table are unaffected by these batches.

Recovery (applies identically to every MCPST dataset; it is a no-op where no
empty batch exists, so the other three runs are not repeated):
* a skipped validation batch is accepted only if it is one of the batches whose
  loss mask is empty, and their number must equal the precomputed count;
* a skipped training batch is accepted only if its loss mask is empty; any
  non-finite loss with supervision still fails.
Same config file and hash as plan a8d285aa; fresh output root; no automatic retry.
Modes: freeze | check | run (waits until plan a8d285aa has no queued job and fewer
than 4 running, so this task never exceeds 4 concurrent processes).
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

NAME = 'extension2_recovery_20261105'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
KEY = '00_METRLA_MCPST'


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def empty_validation_batches(config):
    from experiments import v4_runner
    with base.run_context():
        s = v4_runner.build_v4_session(config, device='cpu', run_seed=32, num_workers=0)
    return [i for i, b in enumerate(s.validation_loader) if not bool(b['loss_mask'].any())]


def freeze():
    assert not PLAN.exists()
    bp = base.check()
    broot = base.output(bp)
    row = bp['configs'][KEY]
    log = broot / f'{KEY}.log'
    status = read(broot / 'queue_status.json')
    failed = next(j for j in status['jobs'] if j['key'] == KEY)
    assert failed['status'] == 'failed' and 'validation skipped an invalid batch' in log.read_text()
    empty = empty_validation_batches(ROOT / row['path'])
    assert empty == [745, 832], empty
    files = [Path(__file__), base.PLAN, log, ROOT / row['path']]
    p = dict(kind=NAME, base_plan=str(base.PLAN.relative_to(ROOT)), base_plan_sha256=bp['plan_sha256'],
             failed_run=dict(key=KEY, exit_code=failed.get('exit_code'), log=str(log.relative_to(ROOT)),
                             cause='count-check wrapper asserted zero skipped validation batches'),
             diagnosis=dict(empty_validation_batches=empty, batch_size=8,
                            other_mcpst_datasets='0 empty validation batches (PEMSBAY, PEMS04, PEMS08)'),
             change='skipped batches accepted only when their loss mask is empty (validation: exactly the '
                    'precomputed batches); non-finite supervised loss still fails',
             applies_to='every MCPST dataset; no-op where no empty batch exists, so only the failed run is rerun',
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


def tolerant_train(p, root):
    from experiments import v4_runner
    from scripts import train_v4 as training
    row = p['configs'][KEY]
    allowed = list(p['diagnosis']['empty_validation_batches'])
    reference = p['anchors'][row['dataset']]['table']
    original_losses, original_train, original_validate = v4_runner.compute_losses, training.train_one_epoch, training.validate
    seen = {'train_empty_skips': 0}

    def losses(output, batch, loss_name='mse', scaler=None):
        result = original_losses(output, batch, loss_name, scaler=scaler)
        if result['optimize'] is None:
            assert not bool(batch['loss_mask'].any()), 'skipped batch has supervision'
            seen['train_empty_skips'] += 1
        else:
            import torch
            assert torch.isfinite(result['optimize']), 'non-finite supervised loss'
        return result

    def epoch(*args, **kwargs):
        before = seen['train_empty_skips']
        r = original_train(*args, **kwargs)
        assert r['skipped'] == seen['train_empty_skips'] - before, 'training skipped a supervised batch'
        return r

    def validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        r = result[0] if isinstance(result, tuple) else result
        assert r['skipped'] == len(allowed), ('validation skips differ from the empty batches', r['skipped'])
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


def run():
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    bp = base.check()
    status_path = base.output(bp) / 'queue_status.json'
    while True:
        jobs = read(status_path)['jobs']
        running = sum(j['status'] == 'running' for j in jobs)
        queued = sum(j['status'] == 'queued' for j in jobs)
        write(root / 'waiting.json', dict(running_in_base=running, queued_in_base=queued, updated=time.time()))
        if queued == 0 and running < 4:
            break
        time.sleep(60)
    write(root / 'started.json', dict(pid=os.getpid(), time=time.time()))
    with base.run_context():
        r = tolerant_train(p, root / 'runs' / KEY)
    write(root / f'{KEY}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'run'])
    a = parser.parse_args()
    print(check()['plan_sha256']) if a.mode == 'check' else globals()[a.mode]()
