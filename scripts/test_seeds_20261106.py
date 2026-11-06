"""One authorized test pass per new final checkpoint of the 2026-11-06 three-seed study.

Modes: freeze | check | queue | worker --key K. Summary: scripts/report_seeds_20261106.py.
Frozen only after every training run of scripts/run_seeds_20261106.py is complete
and collected (COMPLETE.json passed). Each worker first replays validation from
the stored best weights and must reproduce the training-time 33-cell table; only
then is the test context (same run seed, hence the same node split) built, once.
Reused seed-32 runs are not re-tested: their earlier test tables are carried in
the plan. Test numbers are never used to select checkpoints, tune, change
methods or rerun. Authority: user, 2026-11-06 (answer to AskUserQuestion).
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
from scripts import run_seeds_20261106 as training  # noqa: E402

NAME = 'seeds_test_20261106'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
MAX_CONCURRENT = 4
AUTHORITY = ('2026-11-06 user: after training, test each final validation-selected checkpoint of the three-seed '
             'study once; no selection, tuning, method change or rerun from test.')


def output(p):
    return ROOT / 'artifacts/evaluation' / NAME / p['plan_sha256']


def freeze():
    import torch
    assert not PLAN.exists()
    tp = training.check()
    root = training.output(tp)
    assert read(root / 'COMPLETE.json')['passed']
    collected = read(root / 'validation_results.json')
    assert collected['plan_sha256'] == tp['plan_sha256'] and not collected['missing'] and not collected['flags']
    entries, files = {}, [training.PLAN, root / 'validation_results.json', root / 'PREFLIGHT_COMPLETE.json',
                          Path(__file__)]
    for key, row in collected['rows'].items():
        cp = ROOT / row['checkpoint']
        assert sha(cp) == row['checkpoint_sha256']
        state = torch.load(cp, map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == tp['plan_sha256'] and state['epoch'] == row['best_epoch']
        assert state['seed'] == row['seed'] and state['config_sha256'] == row['sha256']
        assert all(torch.isfinite(v).all() for v in state['model_state'].values() if v.is_floating_point())
        entries[key] = dict(row, epoch=int(state['epoch']))
        files += [cp, ROOT / row['report'], ROOT / row['path']]
        del state
    retrained = {f"{e['dataset']}_{e['model']}" for e in entries.values() if e['seed'] == 32}
    reuse = {k: v for k, v in tp['reuse'].items() if f"{v['dataset']}_{v['model']}" not in retrained}
    p = dict(kind=NAME, test_authorized=True, authority=AUTHORITY, training_plan=str(training.PLAN.relative_to(ROOT)),
             training_plan_sha256=tp['plan_sha256'], entries=entries, reuse=reuse, max_concurrent=MAX_CONCURRENT,
             gate='validation replay from best.pt must agree with the training-time table (abs 1e-5 + rel 1e-5) before test',
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(entries), 'tests;', len(reuse), 'reused seed-32 tables')


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
    with training.context(r['model']):
        s = v4_runner.build_v4_session(ROOT / r['path'], device='cuda:0', run_seed=r['seed'], num_workers=0)
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
        ctx = build_runtime_context(r['dataset'], 'test', seed=r['seed'])
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
                     dataset=r['dataset'], model=r['model'], seed=r['seed'], split='test', table=table,
                     scaler=cp['scaler'], test_exposure_hash=ctx.exposure_hash(), test_samples=len(dataset),
                     evaluation_runtime_seed=seed, validation_replay_max_abs_difference=delta))


def queue():
    p = check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'queue_status.json').exists()
    jobs = [dict(key=k, status='queued') for k in p['entries']]
    active, last = {}, 0.
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
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
        if pending and len(active) < MAX_CONCURRENT and time.monotonic() - last > 30 \
                and training.extension.gpu_free_gib() > 12:
            k = pending['key']
            log = (root / f'{k}.log').open('x')
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', k],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                           log=str((root / f'{k}.log').relative_to(ROOT)))
            active[k] = proc, log
            last = time.monotonic()
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
