"""2026-11-12 unseen-node ratio scan on METR-LA (training and validation only; no test access).

Question: does the comparison between ALFNet and the representative baselines depend on the
single unseen-node ratio of the protocol (50 of 207 nodes, 24.15%)?

Design. The unseen pool is the only thing that changes. Nested pools of 30 / 50 / 70 / 90 nodes
(14.5% / 24.2% / 33.8% / 43.5%): a fixed order (POOL_SEED) puts a permutation of the frozen
50-node pool first and a permutation of the other 157 nodes after it; pool n is its first n
entries, so 30 < 50 < 70 < 90 by inclusion and pool 50 is exactly data_v4/METR-LA/unknown_nodes.npy.
Each pool has its own run manifest (a copy of the canonical one with only the `unseen` block
changed) under configs/paper/ratio_20261112/u<n>/; data_v4/ and the canonical manifests are not
touched. A run points the three MANIFEST_DIR globals (lib.runtime_context, experiments.v4_runner,
scripts.train_v4) at that directory, so K, V, T, the scaler (fitted on K, training period) and
the scenario pools all follow the pool; everything else is the production runner.
Models: ALFNet and MoGERNN, Merlin, STBP, STGCN (best validation LifeAvg MAE of their group at the
protocol ratio), each with the very config of its final METR-LA seed-32 run (same SHA256, no
retuning), seed 32 only, same training path as the three-seed studies (scripts/run_seeds_20261106.py,
scripts/run_seeds2_20261110.py) including the non-finite-gradient step guard (1% per epoch).
Pool 50 is not retrained: the seed-32 validation tables of the two three-seed plans are reused.
Selection: validation 33-cell main/LifeAvg/MAE, strict, earlier epoch on ties.
Limits (disclosed): one seed; no retuning per ratio; V and T node identities differ between
pools (the seed splits each pool in halves), so across-ratio differences mix ratio and node
identity, while methods are compared on identical cells within a ratio.

Modes: prepare | freeze | check | preflight --shard i/n | queue | worker --key K | collect.
Authority: user, 2026-11-12 ("只做T3，只在METR-LA数据集上即可，请你开始训练，同样可以同时运行5个").
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_extension_20261103 as extension  # noqa: E402
from scripts import run_seeds_20261106 as seeds1  # noqa: E402
from scripts import run_seeds2_20261110 as seeds2  # noqa: E402

NAME = 'ratio_20261112'
DIR = ROOT / 'configs/paper' / NAME
PLAN = DIR / 'plan.json'
DATASET = 'METRLA'
CANONICAL = ROOT / 'configs/paper/v4_data_manifests'
BASE_POOL = ROOT / 'data_v4/METR-LA/unknown_nodes.npy'
NUM_NODES = 207
BASE = 50
POOLS = (30, 70, 90)
POOL_SEED = 20261112
SEED = 32
MODELS = ('MoGERNN', 'ALFNet', 'STBP', 'STGCN', 'Merlin')  # queue order: longest first
MAX_CONCURRENT = 5
AUTHORITY = ('2026-11-12 user: only T3 (unseen-node ratio scan), METR-LA only, start training, at most 5 '
             'concurrent runs. No test access is authorized by this plan.')


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def manifest_dir(n):
    return CANONICAL if n == BASE else DIR / f'u{n}'


def pool_order():
    base = np.sort(np.load(BASE_POOL).astype(np.int64))
    rest = np.setdiff1d(np.arange(NUM_NODES, dtype=np.int64), base)
    rng = np.random.default_rng(POOL_SEED)
    return np.concatenate([base[rng.permutation(base.size)], rest[rng.permutation(rest.size)]])


def prepare():
    """Write the pools and their manifests (once; nothing existing is overwritten)."""
    order = pool_order()
    assert order.size == NUM_NODES == np.unique(order).size
    base = read(CANONICAL / f'{DATASET}_run_manifest.json')
    assert base['unseen']['count'] == BASE and sha(BASE_POOL) == base['unseen']['sha256']
    assert np.array_equal(np.sort(order[:BASE]), np.sort(np.load(BASE_POOL)))
    for n in POOLS:
        folder = DIR / f'u{n}'
        folder.mkdir(parents=True, exist_ok=False)
        pool = np.sort(order[:n])
        np.save(folder / 'unknown_nodes.npy', pool)
        m = dict(base)
        m['unseen'] = dict(path=(folder / 'unknown_nodes.npy').relative_to(ROOT).as_posix(),
                           sha256=sha(folder / 'unknown_nodes.npy'), count=int(n),
                           ratio=round(float(n / NUM_NODES), 4))
        m['ratio_study'] = dict(study=NAME, pool_seed=POOL_SEED, derived_from=f'configs/paper/v4_data_manifests/'
                                f'{DATASET}_run_manifest.json', changed='unseen block only')
        write(folder / f'{DATASET}_run_manifest.json', m)
        print(n, m['unseen'])


def manifest_context(n):
    """Point every reader of the run manifest at pool n's directory (no-op for the protocol pool)."""
    from unittest.mock import patch
    from experiments import v4_runner
    from lib import runtime_context
    from scripts import train_v4
    stack = ExitStack()
    if n != BASE:
        for module in (runtime_context, v4_runner, train_v4):
            assert module.MANIFEST_DIR == CANONICAL
            stack.enter_context(patch.object(module, 'MANIFEST_DIR', manifest_dir(n)))
    return stack


def split(n):
    """K / V / T of pool n at SEED, from the manifest alone (no payload, no test data)."""
    from lib.runtime_context import split_unseen_pool
    from lib.v4_seeds import derive_seed
    pool = np.load(ROOT / read(manifest_dir(n) / f'{DATASET}_run_manifest.json')['unseen']['path'])
    v, t = split_unseen_pool(pool, seed=derive_seed(SEED, 'node_split'))
    return np.setdiff1d(np.arange(NUM_NODES), pool), v, t


def freeze():
    import torch
    assert not PLAN.exists(), 'plan already frozen'
    p1, p2 = seeds1.check(), seeds2.check()
    order = pool_order()
    configs, reuse, pools = {}, {}, {}
    files = [Path(__file__), seeds1.PLAN, seeds2.PLAN, BASE_POOL, ROOT / 'data_v4/METR-LA/raw_data.pkl',
             CANONICAL / f'{DATASET}_run_manifest.json', ROOT / 'args.py']
    for n in sorted(POOLS + (BASE,)):
        m = read(manifest_dir(n) / f'{DATASET}_run_manifest.json')
        pool = np.load(ROOT / m['unseen']['path'])
        assert np.array_equal(np.sort(pool), np.sort(order[:n])) and m['unseen']['sha256'] == sha(ROOT / m['unseen']['path'])
        k, v, t = split(n)
        assert k.size == NUM_NODES - n and v.size == n // 2 and t.size == n - n // 2
        assert not (set(v) & set(t)) and not (set(k) & set(pool))
        pools[str(n)] = dict(count=n, ratio=n / NUM_NODES, known=int(k.size), validation_unseen=v.tolist(),
                             test_unseen_count=int(t.size), unseen_sha256=m['unseen']['sha256'],
                             manifest=str((manifest_dir(n) / f'{DATASET}_run_manifest.json').relative_to(ROOT)),
                             trained=n != BASE)
        files += [manifest_dir(n) / f'{DATASET}_run_manifest.json', ROOT / m['unseen']['path']]
    for model in MODELS:
        src = (p2 if model == 'ALFNet' else p1)['reuse'][f'{DATASET}_{model}_seed32']
        assert src['seed'] == SEED and sha(ROOT / src['config']) == src['config_sha256']
        reuse[f'{DATASET}_{model}_u{BASE}'] = dict(
            dataset=DATASET, model=model, seed=SEED, unseen_count=BASE, config=src['config'],
            config_sha256=src['config_sha256'], checkpoint=src['checkpoint'],
            checkpoint_sha256=src['checkpoint_sha256'], epoch=src['epoch'], validation_table=src['validation_table'],
            source_plan=(p2 if model == 'ALFNet' else p1)['plan_sha256'])
        files.append(ROOT / src['config'])
        if model == 'ALFNet':
            files += seeds2.nested_configs(src['config'])
        for n in POOLS:
            configs[f'{DATASET}_{model}_u{n}'] = dict(
                path=src['config'], sha256=src['config_sha256'], dataset=DATASET, model=model, seed=SEED,
                unseen_count=n, kind='ALFNet' if model == 'ALFNet' else 'baseline', label=model,
                study_context='bare' if model == 'ALFNet' else None)
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider'):
        files.extend(f for f in (ROOT / folder).rglob('*.py') if '__pycache__' not in f.parts)
    import importlib
    for name in ('scripts.train_v4', 'experiments.v4_runner', 'experiments.v4_validation_collector',
                 'experiments.extension_baselines_20261103'):
        importlib.import_module(name)
    files.extend(Path(m.__file__) for m in list(sys.modules.values())
                 if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == ROOT / 'scripts')
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    p = dict(kind=NAME, status='authorized_training_validation_only', test_authorized=False, authority=AUTHORITY,
             dataset=DATASET, seed=SEED, models=list(MODELS), pools=pools, pool_seed=POOL_SEED,
             pool_rule='first n of [permutation(frozen 50-node pool), permutation(other 157 nodes)], default_rng(pool_seed)',
             max_concurrent=MAX_CONCURRENT, configs=configs, reuse=reuse,
             selection='validation 33-cell main/LifeAvg/MAE, strict, earlier epoch on ties',
             training='production runner; same path as the three-seed studies (Merlin in the 2026-11-03 study '
                      'context, ALFNet bare); only MANIFEST_DIR differs',
             nonfinite_gradient_guard=f'skip and record; fail if one epoch exceeds {seeds1.MAX_EVENT_FRACTION:.0%} of its batches',
             limits=['one seed', 'no retuning per ratio', 'V/T node identities differ between pools'],
             git_head=git, worktree_dirty=bool(dirty.strip()), torch=torch.__version__,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(Path(f).resolve() for f in files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(configs), 'runs')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256'] and not p['test_authorized']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    return p


def verify_context(p, n):
    """The contexts a run of pool n will see (train and validation only)."""
    from lib.runtime_context import build_runtime_context
    info = p['pools'][str(n)]
    with manifest_context(n):
        train = build_runtime_context(DATASET, 'train', seed=SEED)
        val = build_runtime_context(DATASET, 'validation', seed=SEED)
    assert train.known_nodes.size == info['known'] == NUM_NODES - n
    assert val.validation_unseen.tolist() == info['validation_unseen']
    assert np.array_equal(train.signal_scope, train.known_nodes)
    assert np.array_equal(val.signal_scope, np.union1d(train.known_nodes, val.validation_unseen))
    return dict(known=int(train.known_nodes.size), validation_unseen=int(val.validation_unseen.size),
                train_exposure_hash=train.exposure_hash(), validation_exposure_hash=val.exposure_hash())


def train_one(p, key, root, smoke=False):
    row = p['configs'][key]
    with manifest_context(row['unseen_count']):
        fn = seeds2.train_seed if row['model'] == 'ALFNet' else seeds1.train_seed
        return fn(p, key, root, smoke=smoke)


def preflight(shard='0/1'):
    """Per run: context check, then a smoke (20 train / 4 validation batches, nothing saved)."""
    import traceback
    import torch
    torch.set_num_threads(4)
    index, count = map(int, shard.split('/'))
    p = check()
    root = output(p)
    for i, key in enumerate(p['configs']):
        dest = root / 'preflight' / f'{key}.json'
        if i % count != index or dest.exists():
            continue
        print('PREFLIGHT_START', key, flush=True)
        info = dict(key=key, pid=os.getpid())
        try:
            info['context'] = verify_context(p, p['configs'][key]['unseen_count'])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            report = train_one(p, key, None, smoke=True)
            info.update(passed=True, seconds=time.time() - start,
                        smoke_validation_mae=report['history'][0]['validation']['mae'],
                        smoke_guard_events=len(report['history'][0]['train'].get('nonfinite_gradient_steps', [])),
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        except Exception:
            info.update(passed=False, error=traceback.format_exc()[-6000:])
        torch.cuda.empty_cache()
        write(dest, info)
        print('PREFLIGHT_DONE', key, info['passed'], flush=True)


def preflight_complete(p, root):
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')}
    missing = [k for k in p['configs'] if k not in reports]
    failed = [k for k in p['configs'] if k in reports and not reports[k]['passed']]
    write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=not missing and not failed, plan_sha256=p['plan_sha256'],
                                                missing=missing, failed=failed, reports=reports))


def jobs_for(p, root):
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    keys = sorted(p['configs'], key=lambda k: (MODELS.index(p['configs'][k]['model']), p['configs'][k]['unseen_count']))
    return keys, {k: pre['reports'][k]['peak_reserved_gib'] for k in keys}


def worker(key):
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert key in p['configs'] and read(root / 'PREFLIGHT_COMPLETE.json')['passed']
    assert not (root / f'{key}_complete.json').exists()
    context = verify_context(p, p['configs'][key]['unseen_count'])
    r = train_one(p, key, root / 'runs' / key)
    write(root / f'{key}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              context=context,
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


def launch(args, log_path, env):
    log = log_path.open('a')
    return subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *args], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT), log


def queue():
    p = check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    if not (root / 'PREFLIGHT_COMPLETE.json').exists():
        while extension.gpu_free_gib() < 48:
            time.sleep(60)
        procs = [launch(['preflight', '--shard', f'{i}/{MAX_CONCURRENT}'], root / f'preflight_{i}.log', env)
                 for i in range(MAX_CONCURRENT)]
        write(root / 'preflight_status.json', dict(queue_pid=os.getpid(), pids=[q.pid for q, _ in procs],
                                                  started=time.time()))
        for q, log in procs:
            q.wait()
            log.close()
        preflight_complete(p, root)
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    if not pre['passed']:  # nothing is trained unless every run passed its smoke
        write(root / 'ATTENTION_REQUIRED.json', dict(preflight_failed=pre['failed'], missing=pre['missing'],
                                                     test_accessed=False))
        raise SystemExit('preflight failed; no training queued')
    keys, peaks = jobs_for(p, root)
    status_path = root / 'queue_status.json'
    jobs = read(status_path)['jobs'] if status_path.exists() else [dict(key=k, status='queued') for k in keys]
    for j in jobs:  # a restarted queue re-queues nothing by itself; interrupted runs need a decision
        if j['status'] == 'running' and not (root / f"{j['key']}_complete.json").exists():
            j['status'] = 'interrupted'
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
        if pending and len(active) < MAX_CONCURRENT and time.monotonic() - last > 60:
            need = max(8., peaks[pending['key']] * 1.5 + 4.)
            free = extension.gpu_free_gib()
            if free > need:
                key = pending['key']
                proc, log = launch(['worker', '--key', key], root / f'{key}.log', env)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need,
                               log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_ratio_20261112.py worker --key {key}')
                active[key] = proc, log
                last = time.monotonic()
        write(status_path, dict(plan_sha256=p['plan_sha256'], queue_pid=os.getpid(), jobs=jobs,
                                updated=time.time(), max_concurrent=MAX_CONCURRENT))
        time.sleep(20)
    if not all(j['status'] == 'complete' for j in jobs):
        write(root / 'ATTENTION_REQUIRED.json', dict(failed=[j for j in jobs if j['status'] != 'complete'],
                                                     test_accessed=False))
    collect()


def collect():
    p = check()
    root = output(p)
    rows, missing, flags = {}, [], []
    for key, e in p['configs'].items():
        done = root / f'{key}_complete.json'
        if not done.exists():
            missing.append(key)
            continue
        report_path = Path(read(done)['report_path'])
        r = read(report_path)
        assert r['plan_sha256'] == p['plan_sha256'] and r['config_sha256'] == e['sha256'] and r['seed'] == e['seed']
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history'])[1] == r['best_epoch']
        train_skipped = sum(h['train']['skipped'] for h in r['history'])
        guard_steps = [g for h in r['history'] for g in h['train'].get('nonfinite_gradient_steps', [])]
        if r.get('teacher_report'):  # Merlin: the teacher stage is guarded too
            guard_steps += [dict(g, stage='teacher') for h in read(Path(r['teacher_report']))['history']
                            for g in h['train'].get('nonfinite_gradient_steps', [])]
        train_batches = sum(h['train']['batches'] + h['train']['skipped'] for h in r['history'])
        if train_skipped > 0.01 * train_batches:
            flags.append(dict(key=key, train_skipped=train_skipped, train_batches=train_batches))
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'], train_skipped=train_skipped,
                         nonfinite_gradient_steps=guard_steps, context=read(done)['context'],
                         validation_skipped=sorted({h['validation']['skipped'] for h in r['history']}),
                         teacher_report=r.get('teacher_report'))
    for n in POOLS:  # every model of one pool must be scored on identical cells
        tables = [r['table'] for r in rows.values() if r['unseen_count'] == n]
        for t in tables[1:]:
            assert all(t[g][c][f] == tables[0][g][c][f] for g in ('main', 'phase') for c in t[g]
                       for f in t[g][c] if f.endswith('_count')), n
    summary = {}
    for n in sorted(POOLS + (BASE,)):
        for m in MODELS:
            k = f'{DATASET}_{m}_u{n}'
            t = p['reuse'][k]['validation_table'] if n == BASE else rows.get(k, {}).get('table')
            if t:
                summary.setdefault(str(n), {})[m] = {c: t['main'][c]['MAE'] for c in t['main']}
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                                                 reuse=p['reuse'], pools=p['pools'], main_mae=summary,
                                                 missing=missing, flags=flags, test_accessed=False))
    write(root / 'COMPLETE.json', dict(passed=not missing and not flags, runs=len(rows), missing=missing,
                                       flags=flags, test_accessed=False))
    print('collected', len(rows), 'missing', missing, 'flags', flags)
    for n, by in summary.items():
        print(n, {m: round(v['LifeAvg'], 3) for m, v in by.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'freeze', 'check', 'preflight', 'queue', 'worker', 'collect'])
    parser.add_argument('--key')
    parser.add_argument('--shard', default='0/1')
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.key)
    elif a.mode == 'preflight':
        preflight(a.shard)
    elif a.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[a.mode]()
