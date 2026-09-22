"""PEMS-BAY Focal sensitivity: five fixed, validation-only training runs."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = ROOT / 'configs/paper/pemsbay_hparams_20260917'
PLAN = CONFIGS / 'plan.json'
BASE = ROOT / 'artifacts/models/main/PEMSBAY/ALFNet/seed32'
SPECS = {'patch12': (12, 1.), 'patch14': (14, 1.), 'patch16': (16, 1.),
         'lr_half': (11, .5), 'lr_double': (11, 2.)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'plan_sha256'},
                                    sort_keys=True).encode()).hexdigest()


def candidate_config(base, patch, key):
    capacity, multiplier = SPECS[key]
    outer, nested = copy.deepcopy(base), copy.deepcopy(patch)
    outer['runtime']['model_variant'] = 'pemsbay_hparam_' + key
    nested['family']['model']['spa_patchsize'] = capacity
    outer['family']['experts']['patch']['config_path'] = str((CONFIGS / (key + '_patch.yaml')).relative_to(ROOT))
    outer['training']['lr'] *= multiplier
    for group in ('graph_expert_lr', 'patch_expert_lr', 'gate_lr'):
        outer['family']['recipe'][group] *= multiplier
    return outer, nested


def freeze():
    assert not PLAN.exists(), 'immutable plan already exists'
    base = yaml.safe_load((BASE / 'config.yaml').read_text())
    patch = yaml.safe_load((ROOT / base['family']['experts']['patch']['config_path']).read_text())
    CONFIGS.mkdir(parents=True, exist_ok=True)
    entries = {}
    for key, (capacity, multiplier) in SPECS.items():
        outer, nested = candidate_config(base, patch, key)
        for name, data in ((key, outer), (key + '_patch', nested)):
            path = CONFIGS / (name + '.yaml')
            assert not path.exists(), path
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        path = CONFIGS / (key + '.yaml')
        entries[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            capacity=capacity, lr_multiplier=multiplier, seed=32, dataset='PEMSBAY')
    files = [ROOT / 'args.py', BASE / 'config.yaml', BASE / 'config.original.yaml',
             BASE / 'best.pt', BASE / 'development_validation_table.json',
             ROOT / 'configs/paper/v4_data_manifests/PEMSBAY_run_manifest.json',
             ROOT / 'data_v4/PEMS-BAY/raw_data.pkl', ROOT / 'data_v4/PEMS-BAY/unknown_nodes.npy',
             ROOT / 'paper/hparams_pemsbay_20260917/PLAN.md']
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider', 'scripts'):
        files.extend((ROOT / folder).rglob('*.py'))
    files.extend((ROOT / 'configs/paper/final_seed32').glob('*.yaml'))
    files.extend(CONFIGS.glob('*.yaml'))
    p = dict(kind='pemsbay_focal_hparams_round1', status='authorized_validation_only',
             authorization='2026-09-17 user requests PEMSBAY hyperparameter experiments, fixed architecture, 4-5 parallel training runs.',
             configs=entries, max_concurrent=5, test_authorized=False,
             seed=32, selection='validation 33-cell main/LifeAvg/MAE; earlier epoch on ties',
             post_test_exploratory=True, baseline_checkpoint=str((BASE / 'best.pt').relative_to(ROOT)),
             max_epochs=100, early_stopping_start_epoch=40, patience=10,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))},
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'])


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    assert p['test_authorized'] is False and p['max_concurrent'] == 5
    assert set(p['configs']) == set(SPECS)
    for name, expected in p['files'].items():
        assert sha(ROOT / name) == expected, name
    base = yaml.safe_load((BASE / 'config.yaml').read_text())
    patch = yaml.safe_load((ROOT / base['family']['experts']['patch']['config_path']).read_text())
    for key, row in p['configs'].items():
        outer, nested = candidate_config(base, patch, key)
        assert yaml.safe_load((ROOT / row['path']).read_text()) == outer
        assert yaml.safe_load((CONFIGS / (key + '_patch.yaml')).read_text()) == nested
    return p


def output(p):
    return ROOT / 'artifacts/experiments/pemsbay_hparams_20260917' / p['plan_sha256']


def identity(session):
    return dict(scaler=session.scaler, seed_contract=session.seed_contract,
                train_exposure_hash=session.train_context.exposure_hash(),
                validation_exposure_hash=session.validation_context.exposure_hash(),
                static_asset_hash=session.train_context.static_asset_hash())


def parameter_identity(model):
    h = hashlib.sha256()
    shapes = {}
    for name, value in model.named_parameters():
        shapes[name] = list(value.shape)
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dict(shapes=shapes, initial_parameters_sha256=h.hexdigest(),
                parameter_count=sum(p.numel() for p in model.parameters()))


def preflight():
    import torch
    from experiments.v4_runner import build_v4_session
    from scripts.train_v4 import build_optimizer, train
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert not (root / 'PREFLIGHT_COMPLETE.json').exists()
    baseline = build_v4_session(BASE / 'config.yaml', device='cpu', run_seed=32, num_workers=0)
    base_identity = identity(baseline)
    cp = torch.load(BASE / 'best.pt', map_location='cpu', weights_only=False)
    for key, value in base_identity.items():
        assert cp[key] == value, key
    initial = parameter_identity(baseline.model)
    baseline.model.load_state_dict(cp['model_state'], strict=True)
    del cp, baseline
    reports = []
    for key, row in p['configs'].items():
        s = build_v4_session(ROOT / row['path'], device='cpu', run_seed=32, num_workers=0)
        assert identity(s) == base_identity
        assert parameter_identity(s.model) == initial, 'hyperparameter experiment changed parameter shapes or initial weights'
        optimizer = build_optimizer(s)
        lrs = [group['lr'] for group in optimizer.param_groups]
        assert lrs == [rate * row['lr_multiplier'] for rate in (5e-4, 1e-4, 5e-5)]
        capacity = s.model.patch_expert.cfg['model']['spa_patchsize']
        assert capacity == row['capacity'] and len(s.train_context.layout_parts) == 32
        assert len(s.model.patch_expert.reo_all_idx) == capacity * 32
        del optimizer, s
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        report = train(ROOT / row['path'], output_root=None, device='cuda:0', seed=32,
                       max_epochs=1, max_train_batches=2, max_validation_batches=2,
                       batch_size=64, num_workers=0, save=False, plan=p, collect_development_table=False)
        assert report['history'][0]['train']['skipped'] == 0
        assert report['history'][0]['validation']['skipped'] == 0
        summary = dict(key=key, passed=True, capacity=capacity, actual_optimizer_lrs=lrs,
                       parameter_count=initial['parameter_count'], initial_parameters_sha256=initial['initial_parameters_sha256'],
                       parameter_shapes_identical=True, exposure_scaler_seed_identical=True,
                       seconds=time.time()-started, peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                       scope='2 train +2 validation batches; batch64; no saved checkpoint; no test')
        reports.append(summary)
        write(root / 'preflight' / (key + '.json'), summary)
        print('PREFLIGHT_OK', key, flush=True)
        torch.cuda.empty_cache()
    write(root / 'PREFLIGHT_COMPLETE.json', dict(plan_sha256=p['plan_sha256'], passed=True, reports=reports))


def worker(key):
    import torch
    from scripts.train_v4 import train
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    gate = read(root / 'PREFLIGHT_COMPLETE.json')
    assert gate['passed'] and gate['plan_sha256'] == p['plan_sha256']
    started = time.time()
    report = train(ROOT / p['configs'][key]['path'], output_root=root / 'runs' / key,
                   device='cuda:0', seed=32, num_workers=0, plan=p, collect_development_table=True)
    assert all(e[phase]['skipped'] == 0 for e in report['history'] for phase in ('train', 'validation'))
    write(root / (key + '_complete.json'), dict(plan_sha256=p['plan_sha256'], passed=True,
          report_path=str(Path(report['run_directory']) / 'run_report.json'),
          seconds=time.time()-started, peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))


def collect():
    import csv
    p = check()
    root = output(p)
    baseline = read(BASE / 'development_validation_table.json')
    tables = {'baseline_reused': baseline}
    provenance = {'baseline_reused': dict(checkpoint=p['baseline_checkpoint'], checkpoint_sha256=sha(BASE / 'best.pt'))}
    for key in SPECS:
        record = read(root / (key + '_complete.json'))
        assert record['passed'] and record['plan_sha256'] == p['plan_sha256']
        report = read(record['report_path'])
        tables[key] = report['development_validation_table']
        provenance[key] = dict(report_path=record['report_path'], best_epoch=report['best_epoch'],
                               epochs_run=report['epochs_run'], seconds=record['seconds'])
    decisions = {}
    for key, table in tables.items():
        for section in ('main', 'phase'):
            for condition, values in baseline[section].items():
                for field, value in values.items():
                    if field.endswith('_count'):
                        assert table[section][condition][field] == value
        deltas = {m: table['main']['LifeAvg'][m]/baseline['main']['LifeAvg'][m]-1
                  for m in ('MAE', 'RMSE', 'MAPE')}
        early = sum(table['phase'][c]['MAE'] for c in ('Cold', 'Warm', 'Ready')) / sum(baseline['phase'][c]['MAE'] for c in ('Cold', 'Warm', 'Ready'))-1
        unseen = table['main']['Unseen']['MAE']/baseline['main']['Unseen']['MAE']-1
        decisions[key] = dict(lifeavg_relative_changes=deltas, early_mae_relative_change=early,
                             unseen_mae_relative_change=unseen,
                             passes_screen=deltas['MAE'] <= -.005 and deltas['RMSE'] <= .005
                             and deltas['MAPE'] <= .005 and unseen <= .01 and early <= 0.)
    out = ROOT / 'paper/hparams_pemsbay_20260917'
    write(out / 'results.json', dict(plan_sha256=p['plan_sha256'], split='validation',
          post_test_exploratory=True, tables=tables, provenance=provenance, screening=decisions,
          automatic_final_model_replacement=False))
    with (out / 'sensitivity_cells.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'table', 'condition', 'metric', 'value', 'baseline_relative_change'])
        for key, table in tables.items():
            for section in ('main', 'phase'):
                for condition, values in table[section].items():
                    for metric in ('MAE', 'RMSE', 'MAPE'):
                        value = values[metric]
                        writer.writerow([key, section, condition, metric, value, value/baseline[section][condition][metric]-1])
    write(root / 'COMPLETE.json', dict(plan_sha256=p['plan_sha256'], runs=5, test_accessed=False))


def queue():
    p = check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'queue_status.json').exists(), 'inspect existing queue, never duplicate'
    gate = read(root / 'PREFLIGHT_COMPLETE.json')
    assert gate['passed'] and gate['plan_sha256'] == p['plan_sha256']
    required_gib = max(12., max(r['peak_reserved_gib'] for r in gate['reports'])*1.3+4.)
    jobs = [dict(key=k, status='queued') for k in SPECS]
    active = {}
    last_launch = 0.
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', MPLCONFIGDIR='/tmp/mogernn-mpl')
    while any(j['status'] in ('queued', 'running') for j in jobs):
        for key, (proc, log) in list(active.items()):
            job = next(j for j in jobs if j['key'] == key)
            code = proc.poll()
            logpath = root / (key + '.log')
            job['log_age_seconds'] = time.time()-logpath.stat().st_mtime
            job['elapsed_seconds'] = time.time()-job['started']
            job['stall_advisory'] = job['log_age_seconds'] > 3600
            job['timeout_advisory'] = job['elapsed_seconds'] > 48*3600
            if code is not None:
                success = code == 0 and (root / (key + '_complete.json')).exists()
                job.update(status='complete' if success else 'failed', exit_code=code, finished=time.time())
                log.close()
                del active[key]
        if any(j['status'] == 'failed' for j in jobs):
            for j in jobs:
                if j['status'] == 'queued':
                    j['status'] = 'blocked'
        pending = next((j for j in jobs if j['status'] == 'queued'), None)
        if pending and len(active) < p['max_concurrent'] and time.monotonic()-last_launch > 45:
            free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'], text=True).strip())/1024
            if free > required_gib:
                key = pending['key']
                log = (root / (key + '.log')).open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time())
                active[key] = proc, log
                last_launch = time.monotonic()
        write(root / 'queue_status.json', dict(plan_sha256=p['plan_sha256'], jobs=jobs, updated=time.time(), required_free_gib=required_gib))
        time.sleep(15)
    assert all(j['status'] == 'complete' for j in jobs), 'failed training; no automatic retry'
    collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'check', 'preflight', 'worker', 'queue', 'collect'))
    parser.add_argument('--key', choices=tuple(SPECS))
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.key)
    elif args.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[args.mode]()
