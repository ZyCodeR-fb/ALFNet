"""2026-11-05 extension study: MCPST-FSL and MetaDG on the four main datasets.

Modes: freeze | check | preflight [--keys ...] | queue | worker --key K | collect.
Same protocol and rules as scripts/run_extension_20261103.py (whose leakage audit,
count-checked train() wrapper and GPU check are reused); at most 4 concurrent.
The single test pass per validation-selected checkpoint is a separately frozen
plan after training (scripts/test_extension2_20261105.py).
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, identity, read, sha, write  # noqa: E402
from scripts import run_extension_20261103 as first  # noqa: E402

NAME = 'extension2_20261105'
DIR = ROOT / 'configs/paper' / NAME
PLAN = Path(os.environ.get('ALFNET_EXT2_PLAN', DIR / 'plan.json'))
DATASETS = first.DATASETS
MODELS = ('MCPST', 'MetaDG')
MAX_CONCURRENT = 4
METADG_NODE_IDENTITY = os.environ.get('ALFNET_METADG_NODE_IDENTITY', 'structural')
AUTHORITY = ('2026-11-05 user: advisor asks for another 2026 comparison model; adapt references/MCPST and '
             'references/MetaDG under the same requirements as the 2026-11-03 study and train on the four '
             'datasets, at most 4 concurrent.')


def output(p):
    base = Path(os.environ['ALFNET_EXT2_OUTPUT']) if 'ALFNET_EXT2_OUTPUT' in os.environ else ROOT / 'artifacts/experiments' / NAME
    return base / p['plan_sha256']


def config(dataset, model):
    from experiments.extension2_baselines_20261105 import RECIPES
    folder, target = first.DATA[dataset]
    recipe = RECIPES[model]
    wd = recipe['weight_decay'][dataset] if isinstance(recipe['weight_decay'], dict) else recipe['weight_decay']
    c = first.baseline_config(dataset, 'STOP')  # common dataset/lifecycle sections
    c['runtime'].update(model=model, model_variant=f'{NAME}_{dataset}_{model}')
    c['training'].update(batch_size=recipe['batch_size'], loss='mse' if model == 'MCPST' else 'mae',
                         optimizer=recipe['optimizer'].lower(), lr=recipe['lr'], weight_decay=wd)
    family = {'model': {}, 'recipe': {'released_training_recipe': json.loads(json.dumps(recipe))}}
    if model == 'MetaDG':
        family['model']['node_identity'] = METADG_NODE_IDENTITY
    c['family'] = family
    return c


def freeze():
    import torch
    assert not PLAN.exists(), 'plan already frozen; a new identity needs a new name'
    DIR.mkdir(parents=True, exist_ok=True)
    anchors = read(first.ANCHOR_MANIFEST)['entries']
    files = [first.ANCHOR_MANIFEST, Path(__file__)]
    configs, anchor_rows = {}, {}
    for ds in DATASETS:
        a = anchors[ds]
        report_path = ROOT / Path(a['checkpoint']).with_name('run_report.json')
        assert sha(ROOT / a['checkpoint']) == a['checkpoint_sha256']
        anchor_rows[ds] = dict(key=a['key'], config=a['config'], checkpoint=a['checkpoint'],
                               checkpoint_sha256=a['checkpoint_sha256'], report=str(report_path.relative_to(ROOT)),
                               table=read(report_path)['development_validation_table'])
        folder, _ = first.DATA[ds]
        files += [ROOT / a['config'], report_path, ROOT / f'data_v4/{folder}/raw_data.pkl',
                  ROOT / f'data_v4/{folder}/unknown_nodes.npy', ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json']
    index = 0
    for model in MODELS:
        for ds in DATASETS:
            key = f'{index:02d}_{ds}_{model}'
            path = DIR / f'{key}.yaml'
            path.write_text(yaml.safe_dump(config(ds, model), sort_keys=False))
            configs[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), dataset=ds, model=model,
                                kind='baseline', seed=32, label=model)
            files.append(path)
            index += 1
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider', 'scripts', 'tests'):
        files.extend(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts
                     and not p.name.startswith(('test_extension', 'report_extension')))
    files.append(ROOT / 'args.py')
    for name in MODELS:
        manifest = ROOT / f'model/comparison_model/{name}/SOURCE_MANIFEST.json'
        files.append(manifest)
        files.extend(ROOT / e['reference'] for e in read(manifest)['source_files'])
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = dict(kind=NAME, status='authorized_training_validation', authority=AUTHORITY, seed=32,
             max_concurrent=MAX_CONCURRENT, metadg_node_identity=METADG_NODE_IDENTITY,
             test_policy='one test per final validation-selected checkpoint under a separately frozen test plan',
             selection='validation 33-cell main/LifeAvg/MAE, strict improvement, earlier epoch on ties',
             budget='max_epoch 100, patience 10',
             recovery_policy='no automatic retry; failures recorded; any recovery is a new plan applied to every dataset of that method',
             configs=configs, anchors=anchor_rows, git_head=git, torch=torch.__version__,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(configs), 'runs')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    return p


@contextmanager
def run_context():
    from experiments import v4_runner
    from experiments.extension2_baselines_20261105 import extension2_context
    from scripts import train_v4
    with ExitStack() as stack:
        stack.enter_context(extension2_context())
        build = v4_runner.build_v4_session

        def wrapped(*args, **kwargs):
            s = build(*args, **kwargs)
            if hasattr(s.model, 'set_scaler'):
                s.model.set_scaler(s.scaler)
            return s
        for obj in (v4_runner, train_v4):
            stack.enter_context(patch.object(obj, 'build_v4_session', wrapped))
        yield


def train_run(p, key, root, device='cuda:0', smoke=False, max_train_batches=None, max_validation_batches=None):
    row = p['configs'][key]
    return first.safe_train(ROOT / row['path'], root, p, p['anchors'][row['dataset']]['table'], device=device,
                            smoke=smoke, max_train_batches=max_train_batches,
                            max_validation_batches=max_validation_batches)


def preflight(keys=None):
    import torch
    from experiments import v4_runner
    from experiments.extension2_baselines_20261105 import RECIPES
    from scripts import train_v4
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    done = {f.stem for f in (root / 'preflight').glob('*.json')} if (root / 'preflight').exists() else set()
    with run_context():
        for key, row in sorted(p['configs'].items()):
            if (keys and key not in keys) or key in done:
                continue
            print('PREFLIGHT_START', key, flush=True)
            info = dict(key=key)
            s = v4_runner.build_v4_session(ROOT / row['path'], device='cpu', run_seed=32, num_workers=0)
            anchor = torch.load(ROOT / p['anchors'][row['dataset']]['checkpoint'], map_location='cpu', weights_only=False)
            assert all(anchor[k] == v for k, v in identity(s).items()), key
            info['identity'] = 'scaler, seed contract, exposures and static assets equal the ALFNet anchor'
            assert (s.args.max_epoch, s.args.patience, s.args.early_stopping_start_epoch) == (100, 10, 0)
            info['leakage'] = first.leakage_audit(s)
            opt, recipe = train_v4.build_optimizer(s), RECIPES[row['model']]
            sched = train_v4.build_scheduler(s, opt)
            assert type(opt).__name__ == 'Clipped' + recipe['optimizer'] and type(sched).__name__ == recipe['scheduler'][0]
            info['optimizer'] = dict(cls=type(opt).__name__, scheduler=type(sched).__name__,
                                     groups=[{k: v for k, v in g.items() if k in ('lr', 'weight_decay', 'eps')}
                                             for g in opt.param_groups])
            info['parameter_count'] = sum(t.numel() for t in s.model.parameters())
            info['trainable_parameter_count'] = sum(t.numel() for t in s.model.parameters() if t.requires_grad)
            info['node_identity'] = getattr(s.model, 'node_identity', None)
            del s, anchor
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            report = train_run(p, key, None, smoke=True, max_train_batches=20, max_validation_batches=4)
            info.update(passed=True, smoke_epochs=report['epochs_run'],
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3, seconds=time.time() - start)
            torch.cuda.empty_cache()
            write(root / 'preflight' / f'{key}.json', info)
            print('PREFLIGHT_OK', key, f"{info['peak_reserved_gib']:.2f}GiB {info['seconds']:.0f}s", flush=True)
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')}
    if all(reports.get(k, {}).get('passed') for k in p['configs']):
        write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=True, plan_sha256=p['plan_sha256'],
                                                    reports=[reports[k] for k in sorted(p['configs'])]))


def worker(key):
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert read(root / 'PREFLIGHT_COMPLETE.json')['plan_sha256'] == p['plan_sha256']
    assert not (root / f'{key}_complete.json').exists()
    with run_context():
        r = train_run(p, key, root / 'runs' / key)
    write(root / f'{key}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


def queue():
    p = check()
    root = output(p)
    assert read(root / 'PREFLIGHT_COMPLETE.json')['passed']
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'queue_status.json'
    assert not status_path.exists(), 'queue already ran; inspect queue_status.json'
    peaks = {r['key']: r['peak_reserved_gib'] for r in read(root / 'PREFLIGHT_COMPLETE.json')['reports']}
    jobs = [dict(key=k, status='queued') for k in sorted(p['configs'])]
    active, last = {}, 0.
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
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
            free = first.gpu_free_gib()
            if free > need:
                key = pending['key']
                log = (root / f'{key}.log').open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need,
                               log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_extension2_20261105.py worker --key {key}')
                active[key] = proc, log
                last = time.monotonic()
        write(status_path, dict(plan_sha256=p['plan_sha256'], queue_pid=os.getpid(), jobs=jobs,
                                updated=time.time(), max_concurrent=MAX_CONCURRENT))
        time.sleep(20)
    if all(j['status'] == 'complete' for j in jobs):
        collect()
    else:
        write(root / 'ATTENTION_REQUIRED.json', dict(failed=[j for j in jobs if j['status'] != 'complete'],
                                                     test_accessed=False))
        raise RuntimeError('some runs failed; evidence kept, no automatic retry')


def collect():
    p = check()
    root = output(p)
    rows = {}
    for key, e in p['configs'].items():
        done = read(root / f'{key}_complete.json')
        report_path = Path(done['report_path'])
        r = read(report_path)
        assert done['passed'] and r['plan_sha256'] == p['plan_sha256'] and r['config_sha256'] == e['sha256']
        assert all(h[s]['skipped'] == 0 for h in r['history'] for s in ('train', 'validation'))
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history'])[1] == r['best_epoch']
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'])
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                                                 anchors=p['anchors'], reuse={}, test_accessed=False))
    write(root / 'COMPLETE.json', dict(passed=True, runs=len(rows), test_accessed=False))
    print('collected', len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'preflight', 'queue', 'worker', 'collect'])
    parser.add_argument('--key')
    parser.add_argument('--keys', nargs='*')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.key)
    elif args.mode == 'check':
        print(check()['plan_sha256'])
    elif args.mode == 'preflight':
        preflight(args.keys)
    else:
        globals()[args.mode]()
