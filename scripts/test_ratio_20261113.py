"""One authorized test pass per checkpoint of the 2026-11-12 unseen-node ratio scan (METR-LA, seed 32).

Modes: freeze | check | queue | worker --key K | collect. Same procedure as
scripts/test_seeds2_20261110.py: frozen only after every training run of
scripts/run_ratio_20261112.py is complete and collected (COMPLETE.json passed, no flags);
each worker runs inside the pool's manifest context (run_ratio_20261112.manifest_context),
first replays validation from the stored best weights and must reproduce the training-time
33-cell table, then builds the test context once (same run seed and pool, hence the T half of
that pool). Pool 50 is not re-tested: the seed-32 test tables of the two three-seed plans are
carried in the plan. Test numbers are never used to select checkpoints, tune, change methods
or rerun. Authority: user, 2026-11-13 ("开始测试，然后分析结果").
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
from scripts import run_ratio_20261112 as training  # noqa: E402
from scripts import run_seeds_20261106 as seeds1  # noqa: E402
from scripts import run_seeds2_20261110 as seeds2  # noqa: E402

NAME = 'ratio_test_20261113'
PLAN = ROOT / 'configs/paper' / NAME / 'plan.json'
MAX_CONCURRENT = 5
AUTHORITY = ('2026-11-13 user: test the checkpoints of the ratio scan, then analyse; no further training. Each '
             'validation-selected checkpoint is tested once; test results are never used for selection, tuning, '
             'method change or reruns.')


def output(p):
    return ROOT / 'artifacts/evaluation' / NAME / p['plan_sha256']


def freeze():
    import torch
    assert not PLAN.exists()
    tp = training.check()
    root = training.output(tp)
    complete, collected = read(root / 'COMPLETE.json'), read(root / 'validation_results.json')
    assert complete['passed'] and collected['plan_sha256'] == tp['plan_sha256']
    assert not collected['flags'] and not collected['missing'] and not collected['test_accessed']
    assert sorted(collected['rows']) == sorted(tp['configs'])
    entries, files = {}, [training.PLAN, root / 'validation_results.json', root / 'COMPLETE.json', Path(__file__)]
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
    reuse = {}
    for model in training.MODELS:  # protocol pool: test tables of the three-seed plans
        src = (seeds2 if model == 'ALFNet' else seeds1).check()['reuse'][f'{training.DATASET}_{model}_seed32']
        mine = tp['reuse'][f'{training.DATASET}_{model}_u{training.BASE}']
        assert src['checkpoint_sha256'] == mine['checkpoint_sha256'] and src['validation_table'] == mine['validation_table']
        reuse[f'{training.DATASET}_{model}_u{training.BASE}'] = dict(mine, test_table=src['test_table'])
    files += [seeds1.PLAN, seeds2.PLAN]
    p = dict(kind=NAME, test_authorized=True, authority=AUTHORITY, training_plan=str(training.PLAN.relative_to(ROOT)),
             training_plan_sha256=tp['plan_sha256'], entries=entries, reuse=reuse, pools=tp['pools'],
             max_concurrent=MAX_CONCURRENT,
             gate='validation replay from best.pt must agree with the training-time table (abs 1e-5 + rel 1e-5) before test',
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(entries), 'tests;', len(reuse), 'reused pool-50 tables')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256'] and p['test_authorized']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    assert training.check()['plan_sha256'] == p['training_plan_sha256']
    return p


def model_context(r):
    return seeds2.context(r['model'], r['study_context']) if r['model'] == 'ALFNet' else seeds1.context(r['model'])


def worker(key):
    import numpy as np
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
    pool = p['pools'][str(r['unseen_count'])]
    root = output(p)
    dest = root / f'{key}.json'
    assert not dest.exists() and not (root / 'opened' / f'{key}.json').exists(), 'test already opened for this run'
    cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    with training.manifest_context(r['unseen_count']), model_context(r):
        s = v4_runner.build_v4_session(ROOT / r['path'], device='cuda:0', run_seed=r['seed'], num_workers=0)
        assert cp['config_sha256'] == r['sha256'] and cp['scaler'] == s.scaler and cp['seed_contract'] == s.seed_contract
        assert cp['static_asset_hash'] == s.train_context.static_asset_hash()
        assert cp['train_exposure_hash'] == s.train_context.exposure_hash() == r['context']['train_exposure_hash']
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash() == r['context']['validation_exposure_hash']
        assert s.train_context.known_nodes.size == pool['known']
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
        assert np.array_equal(ctx.known_nodes, s.train_context.known_nodes)
        assert ctx.signal_scope.size == pool['known'] + pool['test_unseen_count']
        assert not set(ctx.signal_scope.tolist()) & set(pool['validation_unseen'])
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
                     dataset=r['dataset'], model=r['model'], seed=r['seed'], unseen_count=r['unseen_count'],
                     split='test', table=table, scaler=cp['scaler'], test_exposure_hash=ctx.exposure_hash(),
                     test_samples=len(dataset), evaluation_runtime_seed=seed,
                     validation_replay_max_abs_difference=delta))


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
                and training.extension.gpu_free_gib() > 16:
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
    collect()


def collect():
    p = check()
    root = output(p)
    assert read(root / 'COMPLETE.json')['passed']
    tables = {}
    for key, e in p['entries'].items():
        r = read(root / f'{key}.json')
        assert r['passed'] and r['plan_sha256'] == p['plan_sha256'] and r['checkpoint_sha256'] == e['checkpoint_sha256']
        tables.setdefault(str(e['unseen_count']), {})[e['model']] = r['table']
    for key, e in p['reuse'].items():
        tables.setdefault(str(e['unseen_count']), {})[e['model']] = e['test_table']
    for n, by in tables.items():  # every model of one pool must be scored on identical cells
        ts = list(by.values())
        for t in ts[1:]:
            assert all(t[g][c][f] == ts[0][g][c][f] for g in ('main', 'phase') for c in t[g]
                       for f in t[g][c] if f.endswith('_count')), n
    write(root / 'test_results.json', dict(plan_sha256=p['plan_sha256'], split='test', seed=training.SEED,
                                           tables=tables, reused_pools=[str(training.BASE)]))
    for n in sorted(tables, key=int):
        print(n, {m: round(t['main']['LifeAvg']['MAE'], 3) for m, t in tables[n].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'queue', 'worker', 'collect'])
    parser.add_argument('--key')
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.key)
    elif a.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[a.mode]()
