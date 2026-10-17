"""Frozen seed32 flow transfer: fifteen ALFNet runs and three verified MoGERNN reuses."""
import argparse
import copy
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, write, sha, digest, identity, parameter_identity

CONFIGS = ROOT / 'configs/paper/flow_extension_20260926'
PLAN = CONFIGS / 'plan.json'
CAPACITIES = {'PEMS03': (12, 16, 20), 'PEMS04': (10, 14, 18), 'PEMS08': (6, 8, 10)}
HISTORY = {'PEMS03': '466a1401610941d3', 'PEMS04': '20a68a5dc326ca3d', 'PEMS08': '47265c5bc059ddce'}


def output(p):
    return ROOT / 'artifacts/experiments/flow_extension_20260926' / p['plan_sha256']


def freeze():
    assert not PLAN.exists(), 'immutable plan already exists'
    CONFIGS.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'artifacts/models/main/METRLA/ALFNet/seed32/config.yaml'
    base = yaml.safe_load(source.read_text())
    configs, reuse, sources = {}, {}, [source, ROOT / 'paper/flow_extension_20260926/PLAN.md']
    for ds, (c0, c1, c2) in CAPACITIES.items():
        dataset = yaml.safe_load((ROOT / f'configs/ALFNet/{ds}.yaml').read_text())['dataset']
        graph = yaml.safe_load((ROOT / base['family']['experts']['graph']['config_path']).read_text())
        patch = yaml.safe_load((ROOT / base['family']['experts']['patch']['config_path']).read_text())
        graph['dataset'] = copy.deepcopy(dataset)
        graph_path = CONFIGS / f'{ds}_graph.yaml'
        graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
        for name, capacity, multiplier in [('default', c0, 1.), ('capacity_mid', c1, 1.),
                                           ('capacity_high', c2, 1.), ('lr_half', c0, .5), ('lr_double', c0, 2.)]:
            key = ds + '_' + name
            outer, nested = copy.deepcopy(base), copy.deepcopy(patch)
            outer['dataset'] = copy.deepcopy(dataset)
            nested['dataset'] = copy.deepcopy(dataset)
            outer['runtime']['model_variant'] = 'flow_focal_' + key
            outer['family']['experts']['graph']['config_path'] = str(graph_path.relative_to(ROOT))
            patch_path = CONFIGS / (key + '_patch.yaml')
            outer['family']['experts']['patch']['config_path'] = str(patch_path.relative_to(ROOT))
            nested['family']['model']['spa_patchsize'] = capacity
            outer['training']['lr'] *= multiplier
            for group in ('graph_expert_lr', 'patch_expert_lr', 'gate_lr'):
                outer['family']['recipe'][group] *= multiplier
            patch_path.write_text(yaml.safe_dump(nested, sort_keys=False))
            path = CONFIGS / (key + '.yaml')
            path.write_text(yaml.safe_dump(outer, sort_keys=False))
            configs[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), dataset=ds,
                                capacity=capacity, lr_multiplier=multiplier, seed=32)
        old = ROOT / f'artifacts/models/history/{ds}/MoGERNN/seed32/{HISTORY[ds]}'
        reuse[ds] = str(old.relative_to(ROOT))
        sources.extend(old / f for f in ('config.original.yaml', 'best.pt', 'run_report.json', 'development_validation_table.json'))
        sources.extend([ROOT / f'configs/ALFNet/{ds}.yaml', ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json',
                        ROOT / f'data_v4/{ds}/raw_data.pkl', ROOT / f'data_v4/{ds}/unknown_nodes.npy'])
    sources.extend(ROOT / base['family']['experts'][kind]['config_path'] for kind in ('graph', 'patch'))
    sources.append(ROOT / 'args.py')
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider', 'scripts'):
        sources.extend((ROOT / folder).rglob('*.py'))
    sources.extend(CONFIGS.glob('*.yaml'))
    p = dict(kind='flow_final_focal_extension_v1', status='authorized_validation_only', configs=configs,
             authorization='2026-09-26 user: PEMS03/04/08 ALFNet and MoGERNN, capacity/LR tuning, maximum five concurrent trainings.',
             reuse=reuse, seed=32, max_concurrent=5, test_authorized=False,
             selection='validation main/LifeAvg/MAE; earlier checkpoint and default configuration on exact ties',
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(sources))},
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'])


def check():
    p = read(PLAN)
    assert p['plan_sha256'] == digest(p)
    assert p['test_authorized'] is False and p['max_concurrent'] == 5 and len(p['configs']) == 15
    for name, expected in p['files'].items():
        assert sha(ROOT / name) == expected, name
    return p


def preflight():
    import torch
    from experiments.v4_runner import build_v4_session, validate
    from scripts.train_v4 import train, build_optimizer, model_method_contract
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert not (root / 'PREFLIGHT_COMPLETE.json').exists()
    identities, initial, reports = {}, {}, []
    for ds, directory in p['reuse'].items():
        old = ROOT / directory
        s = build_v4_session(old / 'config.original.yaml', device='cuda:0', run_seed=32, num_workers=0)
        cp = torch.load(old / 'best.pt', map_location='cpu', weights_only=False)
        identities[ds] = identity(s)
        for key, value in identities[ds].items():
            assert cp[key] == value, (ds, key)
        manifest = read(ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        assert cp['payload_sha256'] == manifest['payload']['sha256']
        assert cp['unseen_sha256'] == manifest['unseen']['sha256']
        assert cp['model_method_contract'] == model_method_contract(s)
        assert cp['teacher_forcing_mode'] == 'linear_decay' and cp['teacher_forcing_decay_epochs'] == 40
        assert s.args.loss == 'mse' and s.args.early_stopping_start_epoch == 20 and s.args.patience == 3
        old_report = read(old / 'run_report.json')
        assert old_report['early_stopping_metric'] == 'validation 33-cell main/LifeAvg/MAE'
        assert all(e[phase]['skipped'] == 0 for e in old_report['history'] for phase in ('train', 'validation'))
        s.model.load_state_dict(cp['model_state'], strict=True)
        report, table = validate(s, return_cell_table=True)
        reference = read(old / 'development_validation_table.json')
        errors = []
        for section in ('main', 'phase'):
            for condition, values in reference[section].items():
                for metric in ('MAE', 'RMSE', 'MAPE'):
                    a, b = table[section][condition][metric], values[metric]
                    assert abs(a-b) <= 2e-5+1e-6*abs(b), (ds, section, condition, metric, a, b)
                    errors.append(abs(a-b))
                for field, value in values.items():
                    if field.endswith('_count'):
                        assert table[section][condition][field] == value
        assert len(errors) == 33 and report['skipped'] == 0
        write(root / f'{ds}_reuse.json', dict(passed=True, plan_sha256=p['plan_sha256'],
              source=directory, max_validation_delta=max(errors), table=table, identity=identities[ds], test_accessed=False))
        print('REUSE_VERIFIED', ds, max(errors), flush=True)
        del s, cp
        torch.cuda.empty_cache()
    for key, row in p['configs'].items():
        print('PREFLIGHT_START', key, flush=True)
        ds = row['dataset']
        s = build_v4_session(ROOT / row['path'], device='cpu', run_seed=32, num_workers=0)
        assert identity(s) == identities[ds], (ds, 'exposure/scaler mismatch')
        params = parameter_identity(s.model)
        if ds not in initial:
            initial[ds] = params
            from layers.gnn import calculate_random_walk_matrix
            from layers.lifecycle_moge import build_lifecycle_supports, LEGACY_SUPPORT_NAMES
            adj = s.train_context.adjacency
            supports = build_lifecycle_supports(adj, calculate_random_walk_matrix(adj), calculate_random_walk_matrix(adj.T))
            distances = {a + '__' + b: float((supports[i]-supports[j]).abs().max())
                         for i, a in enumerate(LEGACY_SUPPORT_NAMES) for j, b in enumerate(LEGACY_SUPPORT_NAMES) if i < j}
            write(root / f'{ds}_support_diagnostics.json', dict(operator_max_absolute_distances=distances,
                  layout_algorithm=s.train_context.layout_algorithm, parts=32, target=s.train_context.target,
                  scaler=s.scaler, support_names=LEGACY_SUPPORT_NAMES))
        assert params == initial[ds]
        assert params['parameter_count'] == 1037529
        lrs = [g['lr'] for g in build_optimizer(s).param_groups]
        assert lrs == [r*row['lr_multiplier'] for r in (5e-4, 1e-4, 5e-5)]
        assert len(s.train_context.layout_parts) == 32
        assert max(map(len, s.train_context.layout_parts)) == CAPACITIES[ds][0]
        assert s.model.patch_expert.cfg['model']['spa_patchsize'] == row['capacity']
        assert len(s.model.patch_expert.reo_all_idx) == row['capacity']*32
        del s
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        report = train(ROOT / row['path'], output_root=None, device='cuda:0', seed=32, max_epochs=1,
                       max_train_batches=2, max_validation_batches=2, batch_size=64, num_workers=0,
                       save=False, plan=p, collect_development_table=False)
        assert all(report['history'][0][phase]['skipped'] == 0 for phase in ('train', 'validation'))
        record = dict(key=key, passed=True, actual_optimizer_lrs=lrs, capacity=row['capacity'],
                      parameter_count=params['parameter_count'], initialization_sha=params['initial_parameters_sha256'],
                      seconds=time.time()-started, peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        reports.append(record)
        write(root / 'preflight' / (key + '.json'), record)
        print('PREFLIGHT_OK', key, flush=True)
        torch.cuda.empty_cache()
    write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=True, plan_sha256=p['plan_sha256'], reports=reports))


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
    write(root / (key + '_complete.json'), dict(passed=True, plan_sha256=p['plan_sha256'],
          seconds=time.time()-started, report_path=str(Path(report['run_directory']) / 'run_report.json')))


def collect():
    p = check()
    root = output(p)
    tables, counts, decisions = {}, {}, {}
    for ds in CAPACITIES:
        baseline = read(root / f'{ds}_reuse.json')['table']
        tables[ds + '_MoGERNN'] = baseline
        keys = [k for k, v in p['configs'].items() if v['dataset'] == ds]
        for key in keys:
            done = read(root / (key + '_complete.json'))
            assert done['passed'] and done['plan_sha256'] == p['plan_sha256']
            table = read(done['report_path'])['development_validation_table']
            tables[key] = table
            wins = {}
            for section in ('main', 'phase'):
                for c, values in baseline[section].items():
                    for field, value in values.items():
                        if field.endswith('_count'):
                            assert table[section][c][field] == value
                wins[section] = sum(table[section][c][m] < values[m]-1e-6
                                    for c, values in baseline[section].items() for m in ('MAE', 'RMSE', 'MAPE'))
            wins['total'] = wins['main'] + wins['phase']
            wins['meets_80pct_in_both_tables'] = wins['main'] >= 12 and wins['phase'] >= 15
            counts[key] = wins
        best = min(keys, key=lambda k: tables[k]['main']['LifeAvg']['MAE'])
        default = ds + '_default'
        b, t = tables[default], tables[best]
        changes = {m: t['main']['LifeAvg'][m]/b['main']['LifeAvg'][m]-1 for m in ('MAE', 'RMSE', 'MAPE')}
        early = sum(t['phase'][c]['MAE'] for c in ('Cold', 'Warm', 'Ready'))/sum(b['phase'][c]['MAE'] for c in ('Cold', 'Warm', 'Ready'))-1
        unseen = t['main']['Unseen']['MAE']/b['main']['Unseen']['MAE']-1
        passed = changes['MAE'] <= -.005 and changes['RMSE'] <= .005 and changes['MAPE'] <= .005 and early <= 0 and unseen <= .01
        decisions[ds] = dict(minimum_mae_candidate=best, passes_screen=passed,
                             retained=best if passed else default, lifeavg_relative_changes=changes,
                             early_mae_relative_change=early, unseen_mae_relative_change=unseen)
    write(ROOT / 'paper/flow_extension_20260926/results.json', dict(plan_sha256=p['plan_sha256'],
          split='validation', tables=tables, wins_vs_mogernn=counts, decisions=decisions, test_accessed=False))
    path = ROOT / 'paper/flow_extension_20260926/cells.csv'
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'section', 'condition', 'metric', 'value'])
        for k, table in tables.items():
            for section in ('main', 'phase'):
                for c, values in table[section].items():
                    for m in ('MAE', 'RMSE', 'MAPE'):
                        w.writerow([k, section, c, m, values[m]])
    write(root / 'COMPLETE.json', dict(plan_sha256=p['plan_sha256'], passed=True, test_accessed=False))


def queue():
    p = check()
    root = output(p)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'queue_status.json').exists(), 'inspect existing queue before restarting'
    if not (root / 'PREFLIGHT_COMPLETE.json').exists():
        preflight()
    gate = read(root / 'PREFLIGHT_COMPLETE.json')
    assert gate['passed'] and gate['plan_sha256'] == p['plan_sha256']
    required = max(12., max(r['peak_reserved_gib'] for r in gate['reports'])*1.3+4.)
    # Dispatch default configs across all three datasets first.
    order = sorted(p['configs'], key=lambda k: (0 if k.endswith('_default') else 1, k))
    jobs, active, last = [dict(key=k, status='queued') for k in order], {}, 0.
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', MPLCONFIGDIR='/tmp/mogernn-mpl')
    while any(j['status'] in ('queued', 'running') for j in jobs):
        for key, (proc, log) in list(active.items()):
            job = next(j for j in jobs if j['key'] == key)
            code = proc.poll()
            job['log_age_seconds'] = time.time()-(root / (key + '.log')).stat().st_mtime
            if code is not None:
                ok = code == 0 and (root / (key + '_complete.json')).exists()
                job.update(status='complete' if ok else 'failed', exit_code=code, finished=time.time())
                log.close()
                del active[key]
        if any(j['status'] == 'failed' for j in jobs):
            for j in jobs:
                if j['status'] == 'queued':
                    j['status'] = 'blocked'
        pending = next((j for j in jobs if j['status'] == 'queued'), None)
        if pending and len(active) < 5 and time.monotonic()-last > 45:
            free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'], text=True).strip())/1024
            if free > required:
                key = pending['key']
                log = (root / (key + '.log')).open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time())
                active[key] = proc, log
                last = time.monotonic()
        write(root / 'queue_status.json', dict(plan_sha256=p['plan_sha256'], jobs=jobs, updated=time.time(), required_free_gib=required))
        time.sleep(15)
    assert all(j['status'] == 'complete' for j in jobs), 'failure; no automatic retry'
    collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'check', 'preflight', 'worker', 'queue', 'collect'))
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.key)
    elif args.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[args.mode]()
