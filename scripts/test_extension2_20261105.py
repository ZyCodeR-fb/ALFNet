"""One authorized test pass per final checkpoint of the 2026-11-05 extension study (MCPST, MetaDG).

Modes: freeze | check | queue | worker --key K. Tables: scripts/report_extension_20261103.py.
Frozen only after every training run is complete and its validation-selected
checkpoint is collected (scripts/run_extension2_20261105.py collect). Each worker
first replays validation from the stored best weights and must reproduce the
training-time 33-cell table; only then is the test context built, once. Test
numbers are never used to select checkpoints, tune, change methods or rerun.
Authority: user, 2026-11-05 (answer: test each final checkpoint once), for exactly these runs.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_extension2_20261105 as training  # noqa: E402

NAME = 'extension2_test_20261105'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
MAX_CONCURRENT = 4
AUTHORITY = ('2026-11-05 user: test each final validation-selected checkpoint of the MCPST/MetaDG study '
             'once; no selection, tuning, method change or rerun from test.')


def output(p):
    return ROOT / 'artifacts/evaluation' / NAME / p['plan_sha256']


RECOVERIES = ('scripts.recover_extension2_20261105', 'scripts.recover3_extension2_20261105')


def collected_rows(tp):
    """Every plan a8d285aa run: from its own queue, else from its frozen recovery."""
    import importlib
    root = training.output(tp)
    sources = {}
    for name in RECOVERIES:
        module = importlib.import_module(name)
        rp = module.check()
        done = module.output(rp) / f'{module.KEY}_complete.json'
        if done.exists():
            sources[module.KEY] = (rp['plan_sha256'], read(done), str(module.PLAN.relative_to(ROOT)))
    rows, files = {}, []
    for key, e in tp['configs'].items():
        if (root / f'{key}_complete.json').exists():
            plan_sha, done, plan_path = tp['plan_sha256'], read(root / f'{key}_complete.json'), str(training.PLAN.relative_to(ROOT))
        elif key in sources:
            plan_sha, done, plan_path = sources[key]
        else:
            raise RuntimeError(f'{key} has neither a completed run nor a completed recovery')
        report_path = Path(done['report_path'])
        r = read(report_path)
        assert done['passed'] and r['plan_sha256'] == plan_sha and r['config_sha256'] == e['sha256']
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history'])[1] == r['best_epoch']
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'], training_plan=plan_path,
                         training_plan_sha256=plan_sha,
                         train_skipped=sum(h['train']['skipped'] for h in r['history']),
                         validation_skipped_per_epoch=sorted({h['validation']['skipped'] for h in r['history']}))
        files += [report_path, best]
    return rows, files


def freeze():
    import torch
    assert not PLAN.exists()
    tp = training.check()
    rows, files = collected_rows(tp)
    files += [training.PLAN, Path(__file__)]
    entries = {}
    for key, row in rows.items():
        state = torch.load(ROOT / row['checkpoint'], map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == row['training_plan_sha256'] and state['epoch'] == row['best_epoch']
        assert all(torch.isfinite(v).all() for v in state['model_state'].values() if v.is_floating_point())
        entries[key] = dict(row, epoch=int(state['epoch']))
        files.append(ROOT / row['path'])
        del state
    p = dict(kind=NAME, test_authorized=True, authority=AUTHORITY, training_plan=str(training.PLAN.relative_to(ROOT)),
             training_plan_sha256=tp['plan_sha256'], entries=entries, reuse={}, max_concurrent=MAX_CONCURRENT,
             gate='validation replay from best.pt must agree with the training-time table (abs 1e-5 + rel 1e-5) before test',
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(entries))


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256'] and p['test_authorized']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    training.check()
    return p


def worker(key):
    import torch
    from torch.utils.data import DataLoader
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from experiments import v4_runner
    from experiments.v4_validation_collector import ValidationCellCollector, cell_count, collect_validation_table
    from lib.runtime_context import build_runtime_context
    from lib.v4_seeds import seed_process
    from scripts.eval_release import validation_agrees
    from utils.model_output import unpack_model_output
    torch.set_num_threads(4)
    p = check()
    r = p['entries'][key]
    root = output(p)
    dest = root / f'{key}.json'
    assert not dest.exists() and not (root / 'opened' / f'{key}.json').exists(), 'test already opened for this run'
    cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    with training.run_context():
        s = v4_runner.build_v4_session(ROOT / r['path'], device='cuda:0', run_seed=32, num_workers=0)
        assert cp['config_sha256'] == r['sha256'] and cp['scaler'] == s.scaler and cp['seed_contract'] == s.seed_contract
        assert cp['static_asset_hash'] == s.train_context.static_asset_hash()
        assert cp['train_exposure_hash'] == s.train_context.exposure_hash()
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
        s.model.load_state_dict(cp['model_state'], strict=True)
        if r['model'] == 'Merlin':
            assert bool(s.model.teacher_ready)
        s.model.eval()
        seed = s.seed_contract['derived']['training_runtime']
        seed_process(seed)
        replay = collect_validation_table(s, epoch=r['epoch'])
        ok, delta = validation_agrees(replay, r['table'])
        write(root / 'validation' / f'{key}.json', dict(passed=ok, max_abs_difference=delta, table=replay))
        assert ok, (key, delta)
        write(root / 'opened' / f'{key}.json', dict(plan_sha256=p['plan_sha256'], authority=p['authority'],
                                                    time=time.time(), pid=os.getpid()))
        seed_process(seed)
        ctx = build_runtime_context(r['dataset'], 'test', seed=32)
        assert ctx.static_asset_hash() == cp['static_asset_hash']
        dataset = V4LifecycleDataset(ctx, ROOT / s.args.data_path, input_len=s.args.input_len,
                                     pred_len=s.args.pred_len, slide_step=s.args.slide_step, freq=s.args.freq,
                                     seed=s.seed_contract['derived']['train_schedule'], epoch=0,
                                     scaler=dict(cp['scaler']), validation_seed=s.args.lcbt_validation_seed)
        loader = DataLoader(dataset, batch_size=s.train_loader.batch_size, shuffle=False, num_workers=0)
        collector = ValidationCellCollector(ctx, cp['scaler'], allow_phase='test')
        with torch.no_grad():
            for i, b in enumerate(loader):
                b = v4_runner.to_device(b, s.device)
                pred, _ = unpack_model_output(s.model.forward_v4(b, ctx, epoch=r['epoch'], teacher_forcing_mask=None))
                assert torch.isfinite(pred).all()
                collector.update(b, pred)
                if i % 100 == 0:
                    print('TEST_BATCH', key, i, len(loader), flush=True)
        table = collector.table()
        assert cell_count(table) == 33
        table.pop('development_validation_only', None)
        table.update(split='test')
    write(dest, dict(passed=True, plan_sha256=p['plan_sha256'], checkpoint_sha256=r['checkpoint_sha256'],
                     dataset=r['dataset'], model=r['model'], kind=r['kind'], ablation=r.get('ablation'),
                     seed=32, split='test', table=table, scaler=cp['scaler'], test_exposure_hash=ctx.exposure_hash(),
                     test_samples=len(dataset), evaluation_runtime_seed=seed, validation_replay_max_abs_difference=delta))


def queue():
    p = check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'queue_status.json').exists()
    jobs = [dict(key=k, status='queued') for k in sorted(p['entries'])]
    active = {}
    base_env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    while any(j['status'] in ('queued', 'running') for j in jobs):
        for k, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                j = next(j for j in jobs if j['key'] == k)
                j.update(status='complete' if code == 0 and (root / f'{k}.json').exists() else 'failed',
                         exit_code=code, finished=time.time())
                log.close()
                del active[k]
        pending = next((j for j in jobs if j['status'] == 'queued'), None)
        if pending and len(active) < MAX_CONCURRENT and training.first.gpu_free_gib() > 12:
            k = pending['key']
            env = dict(base_env)
            log = (root / f'{k}.log').open('x')
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', k],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            pending.update(status='running', pid=proc.pid, started=time.time())
            active[k] = proc, log
        write(root / 'queue_status.json', dict(plan_sha256=p['plan_sha256'], queue_pid=os.getpid(), jobs=jobs,
                                               updated=time.time()))
        time.sleep(15)
    if not all(j['status'] == 'complete' for j in jobs):
        write(root / 'ATTENTION_REQUIRED.json', dict(failed=[j for j in jobs if j['status'] != 'complete']))
        raise RuntimeError('some test evaluations failed')
    write(root / 'COMPLETE.json', dict(passed=True, evaluations=len(jobs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'queue', 'worker'])
    parser.add_argument('--key')
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.key)
    elif a.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[a.mode]()
