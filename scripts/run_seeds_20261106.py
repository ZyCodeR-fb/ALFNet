"""2026-11-06 three-seed study of the comparison methods (ALFNet's seeds are a later study).

Methods: UIGNN, GinAR, MoGERNN, SLPF, STGCN, STID, PatchSTG, STBP (the eight paper
baselines), Merlin and STONE (2026-11-03 extension), on METR-LA, PEMS-BAY, PEMS04,
PEMS08. Seeds 32/42/52 (project convention; the run seed also draws the K/V/T node
split). Seed 32 reuses the final runs; seeds 42 and 52 are new trainings with the
very config file (same SHA256) of the seed-32 run, so only the seed differs.
No earlier seed-42/52 run qualifies for reuse: all used the older micro
LifeAvg selector and differ in config (audited at freeze, recorded in the plan).

Training uses the production runner exactly as the seed-32 runs did (skipped batches
are the runner's own behaviour, counted and reported, not a failure); Merlin uses its
two released stages; STONE its released objective (2026-11-03 study context).
Non-finite-gradient step guard (every method, every run of this plan): before each
optimiser step all gradients are checked; if any is non-finite the step is skipped
(no parameter or optimiser-state change) and recorded; a run fails if the events of
one epoch exceed 1% of that epoch's batches. For a run without such a gradient the
guard is a no-op; any reused seed-32 run had none, since one non-finite gradient makes
every parameter NaN (NaN clip norm / Adam moments) and the run could not have finished.
Reason: the first freeze of this study (839232c9, superseded before any training) found
METR-LA GinAR seed 42 turning all parameters NaN at optimiser step 13 of its preflight
smoke (58 parameters with non-finite gradients); same rule as the 2026-11-06 MetaDG
recovery b136d9b6. Not in the released code; disclosed.
GinAR learning rate (METR-LA, PEMS-BAY): the second freeze (5e70abfa, superseded before
any preflight) was checked in a diagnostic: under the guard METR-LA GinAR seed 42 still
had 4 guarded steps in 20 batches and an exploded (1e15) validation MAE, i.e. lr .006 is
unstable for that split. As for the flow datasets on 2026-09-27 (PEMS03 lr .006 gradient
overflow; PEMS04/PEMS08 GinAR lr .001 since), GinAR uses lr .001 on METR-LA and PEMS-BAY
too, so all four datasets share it: only the lr line (and the variant label) of the
seed-32 config changes, all three seeds are trained (the lr .006 seed-32 runs stay
untouched and are not part of the three-seed mean), and their preflight runs a full
first epoch with full validation. Evidence: artifacts/experiments/seeds_20261106/diagnostics/.
Preflight replays every seed-32 checkpoint with the current code; a pair whose
replay does not reproduce its recorded 33-cell validation table (abs 1e-5 + rel
1e-5) is not reused and gets a seed-32 retrain as well (rule fixed here).
Modes: freeze | check | preflight | queue | worker --key K | collect. At most 4 concurrent.
Test (one pass per final checkpoint, user-authorized 2026-11-06): scripts/test_seeds_20261106.py.
"""
import argparse
import copy
import fcntl
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, identity, read, sha, write  # noqa: E402
from scripts import run_extension_20261103 as extension  # noqa: E402

NAME = 'seeds_20261106'
DIR = ROOT / 'configs/paper' / NAME
PLAN = DIR / 'plan.json'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
EXISTING = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP')
EXTENSION = ('Merlin', 'STONE')
MODELS = EXISTING + EXTENSION
NEW_SEEDS = (42, 52)
MAX_CONCURRENT = 4
MAX_EVENT_FRACTION = 0.01
SUPERSEDED = DIR / 'plan.superseded_839232c9_preflight_ginar_nonfinite_gradient.json'
SUPERSEDED2 = DIR / 'plan.superseded_5e70abfa_ginar_guard_insufficient.json'
EVIDENCE = ROOT / 'artifacts/experiments/seeds_20261106/diagnostics/ginar_lr0.006_divergence_20261106.json'
GINAR_LR = {'METRLA': DIR / 'GinAR_lr0.001/METRLA_GinAR.yaml', 'PEMSBAY': DIR / 'GinAR_lr0.001/PEMSBAY_GinAR.yaml'}
SELECTED = ROOT / 'paper/pems08_wd_test_20261012/selected_tables.json'
RECORDS = ROOT / 'paper/hparams_combinations_20260930/results_and_provenance.json'
EXT_TEST = ROOT / 'configs/paper/extension_test_20261103/plan.json'
AUTHORITY = ('2026-11-06 user: after the current round, run 3 seeds of UIGNN, GinAR, MoGERNN, SLPF, STGCN, '
             'STID, PatchSTG, STBP, Merlin, STONE on the four datasets, max 4 concurrent, reuse earlier '
             'final-protocol seeds; test each final checkpoint once afterwards (answer to AskUserQuestion).')
ORDER = {'Merlin': 0, 'MoGERNN': 1, 'UIGNN': 2, 'STBP': 3, 'STONE': 4, 'GinAR': 5, 'PatchSTG': 6, 'SLPF': 7,
         'STGCN': 8, 'STID': 9}


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def context(model):
    """The 2026-11-03 study context for Merlin/STONE; the eight baselines ran under the bare runner."""
    return extension.run_context() if model in EXTENSION else nullcontext()


def canonical(path):
    c = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    c.get('runtime', {}).pop('model_variant', None)
    for k in ('seed', 'device', 'num_workers'):
        c.get('training', {}).pop(k, None)
    return c


def seed32_sources():
    """(dataset, model) -> config, checkpoint, validation/test tables of the reused seed-32 run."""
    import torch
    selected = read(SELECTED)
    records = read(RECORDS)['records']
    ext = read(EXT_TEST)
    ext_root = ROOT / 'artifacts/evaluation/extension_test_20261103' / ext['plan_sha256']
    out = {}
    for ds in DATASETS:
        for m in EXISTING:
            main = ROOT / f'artifacts/models/main/{ds}/{m}/seed32'
            if (main / 'provenance.json').exists():
                prov = read(main / 'provenance.json')
                # the trained bytes; configs/paper/final_seed32 differs only in header comments
                checkpoint, config = main / 'best.pt', main / 'config.original.yaml'
                assert sha(checkpoint) == prov['checkpoint_sha256'] and sha(config) == prov['original_config_sha256']
                assert canonical(config) == canonical(ROOT / prov['config'])
                report = main / 'run_report.json'
            else:
                e = records[ds][m]
                if 'source' in e:
                    src = ROOT / e['source']
                    checkpoint = src / 'best.pt'
                    config = src / ('config.yaml' if (src / 'config.yaml').exists() else 'config.original.yaml')
                    report = src / 'run_report.json'
                else:
                    checkpoint, config, report = ROOT / e['checkpoint'], ROOT / e['config'], ROOT / e['report']
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['seed'] == 32 and state['dataset'] == ds and state['model'] == m, (ds, m)
            assert sha(config) == state['config_sha256'], (ds, m, 'config bytes differ from the trained config')
            assert read(report)['development_validation_table'] == selected['validation'][ds][m], (ds, m)
            out[(ds, m)] = dict(config=str(config.relative_to(ROOT)), config_sha256=sha(config),
                                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                                epoch=int(state['epoch']), training_plan_sha256=state.get('plan_sha256'),
                                validation_table=selected['validation'][ds][m], test_table=selected['test'][ds][m],
                                source='paper/pems08_wd_test_20261012/selected_tables.json')
            del state
        for m in EXTENSION:
            key, e = next((k, e) for k, e in ext['entries'].items() if e['dataset'] == ds and e['model'] == m)
            t = read(ext_root / f'{key}.json')
            assert t['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256']
            out[(ds, m)] = dict(config=e['path'], config_sha256=e['sha256'], checkpoint=e['checkpoint'],
                                checkpoint_sha256=e['checkpoint_sha256'], epoch=e['epoch'],
                                training_plan_sha256=ext['training_plan_sha256'],
                                validation_table=e['table'], test_table=t['table'],
                                source=str((ext_root / f'{key}.json').relative_to(ROOT)))
    return out


def historical_audit(sources):
    """Earlier seed-42/52 runs of these methods: why none is reused."""
    rows = []
    for (ds, m), s in sources.items():
        ref = canonical(ROOT / s['config'])
        for seed in NEW_SEEDS:
            base = ROOT / f'artifacts/models/history/{ds}/{m}/seed{seed}'
            for d in sorted(base.glob('*')) if base.exists() else []:
                cfg, rep = d / 'config.original.yaml', d / 'run_report.json'
                if not (cfg.exists() and rep.exists()):
                    rows.append(dict(dataset=ds, model=m, seed=seed, run=d.name, reusable=False, reason='incomplete'))
                    continue
                r = read(rep)
                reasons = []
                if canonical(cfg) != ref:
                    reasons.append('config differs from the seed-32 final config')
                if r.get('early_stopping_metric') != 'validation 33-cell main/LifeAvg/MAE':
                    reasons.append(f"selector {r.get('early_stopping_metric')!r}")
                rows.append(dict(dataset=ds, model=m, seed=seed, run=str(d.relative_to(ROOT)),
                                 reusable=not reasons, reason='; '.join(reasons) or 'qualifies'))
    return rows


def freeze():
    import torch
    assert not PLAN.exists(), 'plan already frozen'
    old = read(SUPERSEDED)
    assert old['plan_sha256'].startswith('839232c9') and digest(old) == old['plan_sha256']
    old_root = ROOT / 'artifacts/experiments' / NAME / old['plan_sha256']
    assert not (old_root / 'queue_status.json').exists() and not any(old_root.glob('*_complete.json'))
    evidence = read(old_root / 'preflight/METRLA_GinAR.json')
    assert not evidence['passed'] and 'no valid loss position' in evidence['error']
    old2 = read(SUPERSEDED2)
    assert old2['plan_sha256'].startswith('5e70abfa') and digest(old2) == old2['plan_sha256']
    assert not (ROOT / 'artifacts/experiments' / NAME / old2['plan_sha256']).exists()  # never ran
    read(EVIDENCE)
    sources = seed32_sources()
    audit = historical_audit(sources)
    assert not any(r['reusable'] for r in audit), 'a historical run qualifies: add it to reuse before freezing'
    configs, reuse, files = {}, {}, [SELECTED, RECORDS, EXT_TEST, Path(__file__), SUPERSEDED, SUPERSEDED2,
                                     old_root / 'preflight/METRLA_GinAR.json', EVIDENCE,
                                     EVIDENCE.with_name('diag_ginar_20261106.py'), *GINAR_LR.values()]
    for (ds, m), s in sorted(sources.items(), key=lambda kv: (ORDER[kv[0][1]], kv[0][0])):
        reuse[f'{ds}_{m}_seed32'] = dict(s, dataset=ds, model=m, seed=32)
        files += [ROOT / s['config'], ROOT / s['checkpoint']]
        lr_recovery = m == 'GinAR' and ds in GINAR_LR
        if lr_recovery:
            src, new = (ROOT / s['config']).read_text(), GINAR_LR[ds].read_text()
            assert new == src.replace('  lr: 0.006\n', '  lr: 0.001\n').replace(
                'model_variant: ginar_source_inductive_v1\n', f'model_variant: seeds_20261106_{ds}_GinAR_lr0.001\n')
            assert new != src and yaml.safe_load(new)['training']['lr'] == 0.001
        path = str(GINAR_LR[ds].relative_to(ROOT)) if lr_recovery else s['config']
        for seed in NEW_SEEDS + (32,):
            key = f'{ds}_{m}_seed{seed}'
            conditional = seed == 32 and not lr_recovery
            configs[key] = dict(path=path, sha256=sha(ROOT / path), dataset=ds, model=m, seed=seed,
                                kind='baseline', label=m, conditional=conditional, lr_recovery=lr_recovery,
                                condition='only if the seed-32 replay does not reproduce' if conditional else None)
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider'):
        files.extend(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts)
    # scripts/: only what training, replay and test import (plotting or report edits must not break the queue)
    import importlib
    for name in ('scripts.train_v4', 'scripts.eval_release', 'experiments.v4_runner',
                 'experiments.v4_validation_collector', 'experiments.extension_baselines_20261103'):
        importlib.import_module(name)
    files.extend(Path(m.__file__) for m in list(sys.modules.values())
                 if getattr(m, '__file__', None) and Path(m.__file__).resolve().parent == ROOT / 'scripts')
    files.append(ROOT / 'args.py')
    for ds in DATASETS:
        folder, _ = extension.DATA[ds]
        files += [ROOT / f'data_v4/{folder}/raw_data.pkl', ROOT / f'data_v4/{folder}/unknown_nodes.npy',
                  ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json']
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = dict(kind=NAME, status='authorized_training_then_one_test', authority=AUTHORITY, seeds=[32, 42, 52],
             new_seeds=list(NEW_SEEDS), max_concurrent=MAX_CONCURRENT, configs=configs, reuse=reuse,
             historical_seed_runs=audit, selection='validation 33-cell main/LifeAvg/MAE, strict, earlier epoch on ties',
             replay_gate='seed-32 checkpoint replay with current code must agree within abs 1e-5 + rel 1e-5',
             training='production runner as for seed 32 (skips counted and reported); Merlin/STONE in the 2026-11-03 study context',
             nonfinite_gradient_guard=dict(rule='skip and record optimiser steps whose gradients are not all finite; '
                                                f'fail if one epoch exceeds {MAX_EVENT_FRACTION:.0%} of its batches',
                                           applies_to='every run of this plan (no-op without a non-finite gradient)'),
             ginar_lr_recovery=dict(datasets=sorted(GINAR_LR), lr=0.001, previous_lr=0.006,
                                    configs={d: str(f.relative_to(ROOT)) for d, f in GINAR_LR.items()},
                                    seeds_trained=[32, 42, 52], evidence=str(EVIDENCE.relative_to(ROOT)),
                                    precedent='paper/baseline_extension_20260927/RECOVERY.md',
                                    preflight='full first epoch + full validation, seed 42'),
             superseded2=dict(plan_sha256=old2['plan_sha256'], file=str(SUPERSEDED2.relative_to(ROOT)),
                              reason='guard alone insufficient for GinAR lr .006 (diagnostic); never ran'),
             superseded=dict(plan_sha256=old['plan_sha256'], file=str(SUPERSEDED.relative_to(ROOT)),
                             reason='preflight smoke: METR-LA GinAR seed 42 non-finite gradients at optimiser step 13 '
                                    'made all parameters NaN (validation then refused); no training was started',
                             evidence=str((old_root / 'preflight/METRLA_GinAR.json').relative_to(ROOT))),
             git_head=git, torch=torch.__version__,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(configs), 'config entries;', sum(not c['conditional'] for c in configs.values()), 'new runs')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    return p


def nonfinite_gradient_guard(training):
    """Skip (and record) optimiser steps whose gradients are not all finite; see the module docstring."""
    import torch
    from contextlib import ExitStack
    from types import MethodType
    from unittest.mock import patch
    original_optimizer, original_epoch = training.build_optimizer, training.train_one_epoch
    state = {'epoch': None, 'events': []}

    def build_optimizer(session, *args, **kwargs):
        opt = original_optimizer(session, *args, **kwargs)
        names = {id(q): n for n, q in session.model.named_parameters()}
        step = opt.step

        def guarded_step(self_opt, closure=None):  # bound: LR schedulers wrap opt.step.__func__
            bad = [names.get(id(q), '?') for g in opt.param_groups for q in g['params']
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad:
                state['events'].append(dict(epoch=state['epoch'], parameters=bad[:20], count=len(bad)))
                opt.zero_grad(set_to_none=True)
                return None
            return step(closure)
        opt.step = MethodType(guarded_step, opt)
        return opt

    def train_one_epoch(session, optimizer, epoch, *args, **kwargs):
        state['epoch'] = epoch
        before = len(state['events'])
        r = original_epoch(session, optimizer, epoch, *args, **kwargs)
        events = state['events'][before:]
        assert len(events) <= MAX_EVENT_FRACTION * len(session.train_loader), ('nonfinite-gradient steps', len(events))
        r['nonfinite_gradient_steps'] = events
        return r
    stack = ExitStack()
    stack.enter_context(patch.object(training, 'build_optimizer', build_optimizer))
    stack.enter_context(patch.object(training, 'train_one_epoch', train_one_epoch))
    return stack


def train_seed(p, key, root, smoke=False, full_epoch=False):
    from scripts import train_v4 as training
    row = p['configs'][key]
    short = smoke and not full_epoch
    kwargs = dict(device='cuda:0', smoke=smoke, max_train_batches=20 if short else None,
                  max_validation_batches=4 if short else None)

    def fn(config, output_root, device, plan, resume, smoke, max_train_batches, max_validation_batches):
        # Installed here, inside any Merlin stage patch, so every method's optimiser is guarded.
        with nonfinite_gradient_guard(training):
            return training.train(config, output_root, device=device, seed=row['seed'], num_workers=0, plan=plan,
                                  save=not smoke, resume=resume, collect_development_table=not smoke,
                                  max_epochs=1 if smoke else None, max_train_batches=max_train_batches,
                                  max_validation_batches=max_validation_batches)
    with context(row['model']):
        if row['model'] == 'Merlin':
            from experiments.extension_baselines_20261103 import train_merlin
            return train_merlin(fn, ROOT / row['path'], root, p, **kwargs)
        return fn(ROOT / row['path'], root, kwargs['device'], p, False, smoke, kwargs['max_train_batches'],
                  kwargs['max_validation_batches'])


def replay_seed32(p, ds, m):
    import torch
    from experiments import v4_runner
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.eval_release import validation_agrees
    s32 = p['reuse'][f'{ds}_{m}_seed32']
    cp = torch.load(ROOT / s32['checkpoint'], map_location='cpu', weights_only=False)
    with context(m):
        s = v4_runner.build_v4_session(ROOT / s32['config'], device='cuda:0', run_seed=32, num_workers=0)
        ids = identity(s)
        assert cp['scaler'] == ids['scaler'] and cp['train_exposure_hash'] == ids['train_exposure_hash']
        assert cp['validation_exposure_hash'] == ids['validation_exposure_hash']
        s.model.load_state_dict(cp['model_state'], strict=True)
        if m == 'Merlin':
            assert bool(s.model.teacher_ready)
        s.model.eval()
        seed_process(s.seed_contract['derived']['training_runtime'])
        table = collect_validation_table(s, epoch=s32['epoch'])
    ok, delta = validation_agrees(table, s32['validation_table'])
    return dict(passed=bool(ok), max_abs_difference=delta)


def pairs(p):
    return sorted({(c['dataset'], c['model']) for c in p['configs'].values()}, key=lambda x: (ORDER[x[1]], x[0]))


def preflight(shard='0/1'):
    """Per pair: seed-32 replay gate, then a seed-42 smoke (20 train / 4 validation batches)."""
    import traceback
    import torch
    torch.set_num_threads(4)
    index, count = map(int, shard.split('/'))
    p = check()
    root = output(p)
    for i, (ds, m) in enumerate(pairs(p)):
        dest = root / 'preflight' / f'{ds}_{m}.json'
        if i % count != index or dest.exists():
            continue
        print('PREFLIGHT_START', ds, m, flush=True)
        info = dict(dataset=ds, model=m, pid=os.getpid())
        try:
            info['replay_seed32'] = replay_seed32(p, ds, m)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            full = p['configs'][f'{ds}_{m}_seed42'].get('lr_recovery', False)
            report = train_seed(p, f'{ds}_{m}_seed42', None, smoke=True, full_epoch=full)
            info.update(passed=True, smoke_epochs=report['epochs_run'], seconds=time.time() - start,
                        full_first_epoch=full, smoke_validation_mae=report['history'][0]['validation']['mae'],
                        smoke_guard_events=len(report['history'][0]['train'].get('nonfinite_gradient_steps', [])),
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        except Exception:
            info.update(passed=False, error=traceback.format_exc()[-6000:])
        torch.cuda.empty_cache()
        write(dest, info)
        print('PREFLIGHT_DONE', ds, m, info['passed'], info.get('replay_seed32'), flush=True)


def preflight_complete(p, root):
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')}
    names = [f'{ds}_{m}' for ds, m in pairs(p)]
    missing = [k for k in names if k not in reports]
    failed = [k for k in names if k in reports and not reports[k]['passed']]
    retrain = [k for k in names if k in reports and reports[k]['passed'] and not reports[k]['replay_seed32']['passed']]
    write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=not missing and not failed, plan_sha256=p['plan_sha256'],
                                                missing=missing, failed=failed, retrain_seed32=retrain,
                                                reports=reports))


def jobs_for(p, root):
    """Pairs that passed preflight: seeds 42/52, plus seed 32 where the replay gate failed."""
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    retrain = set(pre['retrain_seed32'])
    keys = []
    for k, c in p['configs'].items():
        pair = f"{c['dataset']}_{c['model']}"
        if pre['reports'].get(pair, {}).get('passed') and (not c['conditional'] or pair in retrain):
            keys.append(k)
    return keys, {k: pre['reports'][f"{p['configs'][k]['dataset']}_{p['configs'][k]['model']}"]['peak_reserved_gib']
                  for k in keys}


def worker(key):
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert key in jobs_for(p, root)[0] and not (root / f'{key}_complete.json').exists()
    r = train_seed(p, key, root / 'runs' / key)
    write(root / f'{key}_complete.json', dict(passed=True, plan_sha256=p['plan_sha256'], pid=os.getpid(),
                                              report_path=str(Path(r['run_directory']) / 'run_report.json')))


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
        procs = []
        for i in range(MAX_CONCURRENT):
            log = (root / f'preflight_{i}.log').open('a')
            procs.append((subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'preflight',
                                            '--shard', f'{i}/{MAX_CONCURRENT}'], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT), log))
        write(root / 'preflight_status.json', dict(queue_pid=os.getpid(), pids=[q.pid for q, _ in procs],
                                                  started=time.time()))
        for q, log in procs:
            q.wait()
            log.close()
        preflight_complete(p, root)
    keys, peaks = jobs_for(p, root)
    status_path = root / 'queue_status.json'
    jobs = read(status_path)['jobs'] if status_path.exists() else [dict(key=k, status='queued') for k in keys]
    for j in jobs:  # a restarted queue re-queues only jobs that never produced a result
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
                log = (root / f'{key}.log').open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need,
                               log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_seeds_20261106.py worker --key {key}')
                active[key] = proc, log
                last = time.monotonic()
        write(status_path, dict(plan_sha256=p['plan_sha256'], queue_pid=os.getpid(), jobs=jobs,
                                updated=time.time(), max_concurrent=MAX_CONCURRENT))
        time.sleep(20)
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    if pre['failed'] or not all(j['status'] == 'complete' for j in jobs):
        write(root / 'ATTENTION_REQUIRED.json', dict(failed=[j for j in jobs if j['status'] != 'complete'],
                                                     preflight_failed=pre['failed'], test_accessed=False))
    collect()


def collect():
    p = check()
    root = output(p)
    keys, _ = jobs_for(p, root)
    rows, missing, flags = {}, [], []
    for key in keys:
        e = p['configs'][key]
        done = root / f'{key}_complete.json'
        if not done.exists():
            missing.append(key)
            continue
        report_path = Path(read(done)['report_path'])
        r = read(report_path)
        assert r['plan_sha256'] == p['plan_sha256'] and r['config_sha256'] == e['sha256'] and r['seed'] == e['seed']
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history'])[1] == r['best_epoch']
        train_skipped = sum(h['train']['skipped'] for h in r['history'])
        guard_steps = [e for h in r['history'] for e in h['train'].get('nonfinite_gradient_steps', [])]
        if r.get('teacher_report'):  # Merlin: the teacher stage is guarded too
            guard_steps += [dict(e, stage='teacher') for h in read(Path(r['teacher_report']))['history']
                            for e in h['train'].get('nonfinite_gradient_steps', [])]
        train_batches = sum(h['train']['batches'] + h['train']['skipped'] for h in r['history'])
        if train_skipped > 0.01 * train_batches:
            flags.append(dict(key=key, train_skipped=train_skipped, train_batches=train_batches))
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'], train_skipped=train_skipped,
                         nonfinite_gradient_steps=guard_steps,
                         validation_skipped=sorted({h['validation']['skipped'] for h in r['history']}),
                         teacher_report=r.get('teacher_report'))
    # every model of one (dataset, seed) must be scored on identical cells
    for ds in DATASETS:
        for seed in (32,) + NEW_SEEDS:
            tables = [r['table'] for r in rows.values() if r['dataset'] == ds and r['seed'] == seed]
            for t in tables[1:]:
                for g in ('main', 'phase'):
                    for c in t[g]:
                        for f, v in t[g][c].items():
                            if f.endswith('_count'):
                                assert v == tables[0][g][c][f], (ds, seed, g, c, f)
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                                                 reuse=p['reuse'], missing=missing, flags=flags, test_accessed=False))
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    # Missing runs, failed preflights or a divergence flag stop the chain before any test access.
    write(root / 'COMPLETE.json', dict(passed=not missing and not flags and not pre['failed'], runs=len(rows),
                                       missing=missing, flags=flags, preflight_failed=pre['failed'],
                                       test_accessed=False))
    print('collected', len(rows), 'missing', missing, 'flags', flags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'preflight', 'queue', 'worker', 'collect'])
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
