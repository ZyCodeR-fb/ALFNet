"""2026-11-10 three-seed study of ALFNet, MCPST and MetaDG (the 2026-11-06 study covered ten other methods).

Methods ALFNet, MCPST, MetaDG on METR-LA, PEMS-BAY, PEMS04, PEMS08; seeds 32/42/52 (the
run seed also draws the K/V/T node split). Seed 32 reuses the existing runs; seeds 42
and 52 are new trainings with the very config file (same SHA256) of the seed-32 run, so
only the seed differs.
* ALFNet: the seed-32 run of the configuration recorded in alfnet_selection.json (written
  by mode ``select`` from the user's decision after the 2026-11-10 configuration audit,
  paper/alfnet_config_audit_20261110/). Each run is trained in the study context of its
  seed-32 run (bare runner; flow_support_transfer adapter; support + unified adapters),
  and preflight first re-trains one full seed-32 epoch on that path and compares it with
  the original epoch-0 record before any training is queued: counts, method contract and
  parameter count exact; epoch-mean training losses within rel TRAIN_REL (+ abs REPRO_ABS);
  validation metrics after that epoch only within a sanity bound (rel VALIDATION_SANITY_REL).
  Why validation is not held to 1e-3 (first freeze 643ed7a4, superseded before any training):
  GPU kernels are not deterministic (no deterministic algorithms in the original runs);
  independent current-code reproductions agree bit for bit for the first 5-10 batches and
  then drift, so after one epoch their validation metrics differ by up to 2-4% on PEMS-BAY
  and PEMS08 (1e-4 on METR-LA/PEMS04), with the original record inside that spread, while
  their epoch-mean training losses differ by <= 3e-4 (original vs new <= 1.2e-4). A wrong
  path (PEMS08 without the flow_unified adapter) moves the training losses by +24%.
  Evidence: artifacts/experiments/seeds2_20261110/diagnostics/.
* MCPST, MetaDG: seed 32 from the 2026-11-05 study (plan a8d285aa, recoveries 034802bd
  and b136d9b6; test plan 60b911dd), trained in run_extension2_20261105.run_context().
  The frozen recovery rules carry over: the runner's skip of a batch without supervision
  is its own behaviour and is counted (a run whose training skips exceed 1% is flagged
  as divergent and stops the chain before test); every run has the non-finite-gradient
  step guard (scripts/run_seeds_20261106.nonfinite_gradient_guard: skip and record the
  step, fail when one epoch's events exceed 1% of its batches; a no-op without events).
The guard is installed for ALFNet too (no-op without a non-finite gradient; the epoch-0
reproduction runs with it). Preflight per pair: seed-32 replay with current code (abs
1e-5 + rel 1e-5; a pair that fails gets a seed-32 retrain, rule fixed here), ALFNet
epoch-0 reproduction, seed-42 smoke (20 train / 4 validation batches); preflight shards
run in parallel. At most 5 concurrent processes (preflight and test included).
Modes: select | freeze | check | preflight --shard i/n | queue | worker --key K | collect.
Test (one pass per new final checkpoint, user-authorized 2026-11-10): scripts/test_seeds2_20261110.py.
"""
import argparse
import copy
import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import digest, read, sha, write  # noqa: E402
from scripts import run_extension_20261103 as extension  # noqa: E402
from scripts import run_seeds_20261106 as seeds1  # noqa: E402

NAME = 'seeds2_20261110'
DIR = ROOT / 'configs/paper' / NAME
PLAN = DIR / 'plan.json'
SELECTION = DIR / 'alfnet_selection.json'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
MODELS = ('ALFNet', 'MCPST', 'MetaDG')
NEW_SEEDS = (42, 52)
MAX_CONCURRENT = 5
MAX_EVENT_FRACTION = seeds1.MAX_EVENT_FRACTION
TRAIN_REL, REPRO_ABS, VALIDATION_SANITY_REL = 1e-3, 1e-6, 0.1
SUPERSEDED = DIR / 'plan.superseded_643ed7a4_epoch0_validation_tolerance.json'
DIAGNOSTICS = ROOT / 'artifacts/experiments' / NAME / 'diagnostics'
FINAL = ROOT / 'paper/pems08_wd_test_20261012/selected_configuration_manifest.json'
SELECTED_TABLES = ROOT / 'paper/pems08_wd_test_20261012/selected_tables.json'
AUDIT = ROOT / 'paper/alfnet_config_audit_20261110/candidates.json'
EXT2_TEST = ROOT / 'configs/paper/extension2_test_20261105/plan.json'
SEEDS1_PLAN = ROOT / 'configs/paper/seeds_20261106/plan.json'
AUTHORITY = ('2026-11-10 user: after the ALFNet configuration audit, train ALFNet, MCPST and MetaDG on the four '
             'datasets with seeds 32/42/52 (reuse seed 32 after an exact replay), max 5 concurrent processes; '
             'test each new final checkpoint once; test results never used for selection, tuning or reruns.')
# Training path of each ALFNet study (scripts/run_*.py, verified 2026-11-10): which adapters wrapped train().
STUDY_CONTEXT = {
    'metrla_hparams_20260920': 'bare', 'pemsbay_hparams_20260917': 'bare', 'hparams_completion_20260928': 'bare',
    'hparams_combinations_20260930': 'bare', 'flow_extension_20260926': 'bare', 'main': 'bare',
    'flow_support_transfer_20261003': 'support', 'flow_unified_20261003': 'unified',
    'flow_ofat_20261006': 'support+unified', 'speed_support_20261004': 'support+unified',
    'flow_lr_cross_20261004': 'support+unified',
}
# Seed-32 wall-clock hours (queue records), only to start the longest jobs first.
HOURS = {('ALFNet', 'METRLA'): 6.5, ('ALFNet', 'PEMSBAY'): 11.6, ('ALFNet', 'PEMS04'): 2.7, ('ALFNet', 'PEMS08'): 4.3,
         ('MCPST', 'METRLA'): 2.0, ('MCPST', 'PEMSBAY'): 15.0, ('MCPST', 'PEMS04'): 3.0, ('MCPST', 'PEMS08'): 2.7,
         ('MetaDG', 'METRLA'): 3.3, ('MetaDG', 'PEMSBAY'): 8.9, ('MetaDG', 'PEMS04'): 5.1, ('MetaDG', 'PEMS08'): 4.1}


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def context(model, study_context=None):
    stack = ExitStack()
    if model != 'ALFNet':
        from scripts.run_extension2_20261105 import run_context
        stack.enter_context(run_context())
        return stack
    if study_context in ('support', 'support+unified'):
        from experiments.flow_support_transfer import adapter
        stack.enter_context(adapter())
    if study_context in ('unified', 'support+unified'):
        from experiments.flow_unified import adapter
        stack.enter_context(adapter())
    assert study_context in ('bare', 'support', 'unified', 'support+unified'), study_context
    return stack


def select(choices, authority):
    """Record the user's ALFNet configuration decision (dataset -> audit candidate key)."""
    assert not PLAN.exists() and not SELECTION.exists()
    rows = read(AUDIT)['rows']
    entries = {}
    for item in choices:
        ds, key = item.split('=', 1)
        row = rows[key]
        assert ds in DATASETS and row['dataset'] == ds and not row['comparable_problems'], (ds, key)
        assert row['replay'] and row['replay']['passed'], (key, 'needs an exact current-code replay first')
        entries[ds] = dict(candidate=key, config=row['config'], config_sha256=row['config_sha256'],
                           checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'])
    assert set(entries) == set(DATASETS)
    DIR.mkdir(parents=True, exist_ok=True)
    write(SELECTION, dict(authority=authority, audit=str(AUDIT.relative_to(ROOT)), audit_sha256=sha(AUDIT),
                          entries=entries))
    print(entries)


def nested_configs(config):
    c = yaml.safe_load((ROOT / config).read_text())
    return [ROOT / c['family']['experts'][e]['config_path'] for e in ('graph', 'patch')]


def seed32_sources():
    """(dataset, model) -> config, checkpoint, validation/test tables and training report of the reused seed-32 run."""
    import torch
    selection = read(SELECTION)['entries']
    rows, final, tables = read(AUDIT)['rows'], read(FINAL)['entries'], read(SELECTED_TABLES)
    out = {}
    for ds in DATASETS:
        row = rows[selection[ds]['candidate']]
        assert sha(ROOT / row['checkpoint']) == row['checkpoint_sha256'] == selection[ds]['checkpoint_sha256']
        assert sha(ROOT / row['config']) == row['config_sha256']
        state = torch.load(ROOT / row['checkpoint'], map_location='cpu', weights_only=False)
        assert state['seed'] == 32 and state['dataset'] == ds and state['model'] == 'ALFNet'
        assert state['config_sha256'] == row['config_sha256'] and state['epoch'] == row['selector']['best_epoch']
        report = read(ROOT / row['report'])
        assert report['development_validation_table'] == row['validation_table']
        if row['checkpoint_sha256'] == final[ds]['checkpoint_sha256']:
            assert tables['validation'][ds]['ALFNet'] == row['validation_table']
            test_table, source = tables['test'][ds]['ALFNet'], str(SELECTED_TABLES.relative_to(ROOT))
            # several records of one checkpoint may differ only in their free-text 'note'
            assert all(row['test_table'][g] == test_table[g] for g in ('main', 'phase'))
        else:
            assert row['test_tables_distinct'] == 1, (ds, row['key'], 'needs exactly one recorded test table')
            test_table, source = row['test_table'], row['test_records'][0]
        out[(ds, 'ALFNet')] = dict(config=row['config'], config_sha256=row['config_sha256'],
                                   checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'],
                                   epoch=int(state['epoch']), training_plan_sha256=row['plan_sha256'],
                                   report=row['report'], validation_table=row['validation_table'],
                                   test_table=test_table, source=source, candidate=row['key'],
                                   study_context=STUDY_CONTEXT[row['study']])
        del state
    ext = read(EXT2_TEST)
    xroot = ROOT / 'artifacts/evaluation/extension2_test_20261105' / ext['plan_sha256']
    for key, e in ext['entries'].items():
        t = read(xroot / f'{key}.json')
        assert t['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(ROOT / e['checkpoint'])
        assert sha(ROOT / e['path']) == e['sha256'] and read(ROOT / e['report'])['development_validation_table'] == e['table']
        out[(e['dataset'], e['model'])] = dict(config=e['path'], config_sha256=e['sha256'], checkpoint=e['checkpoint'],
                                               checkpoint_sha256=e['checkpoint_sha256'], epoch=e['epoch'],
                                               training_plan_sha256=e['training_plan_sha256'], report=e['report'],
                                               validation_table=e['table'], test_table=t['table'],
                                               source=str((xroot / f'{key}.json').relative_to(ROOT)),
                                               study_context='run_extension2_20261105.run_context')
    assert set(out) == {(d, m) for d in DATASETS for m in MODELS}
    return out


def canonical(path):
    c = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    c.get('runtime', {}).pop('model_variant', None)
    for k in ('seed', 'device', 'num_workers'):
        c.get('training', {}).pop(k, None)
    return c


def historical_audit(sources):
    """Every earlier seed-42/52 run of these methods on these datasets: why none is reused."""
    rows = []
    for f in sorted(ROOT.glob('artifacts/**/run_report.json')):
        r = read(f)
        ds, m, seed = r.get('dataset'), r.get('model'), r.get('seed')
        if m not in MODELS or ds not in DATASETS or seed not in NEW_SEEDS:
            continue
        s = sources[(ds, m)]
        run = f.parent
        cfg = ROOT / r['config'] if (ROOT / r['config']).exists() else run / 'config.original.yaml'
        reasons = []
        if r.get('config_sha256') != s['config_sha256']:
            reasons.append('config bytes differ from the seed-32 config'
                           + ('' if not cfg.exists() or canonical(cfg) != canonical(ROOT / s['config'])
                              else ' (canonical content equal)'))
        if r.get('early_stopping_metric') != 'validation 33-cell main/LifeAvg/MAE':
            reasons.append(f"selector {r.get('early_stopping_metric')!r}")
        rows.append(dict(dataset=ds, model=m, seed=seed, run=str(run.relative_to(ROOT)), reusable=not reasons,
                         reason='; '.join(reasons) or 'qualifies'))
    return rows


def freeze():
    import importlib
    import torch
    assert not PLAN.exists(), 'plan already frozen'
    old = read(SUPERSEDED)
    assert old['plan_sha256'].startswith('643ed7a4') and digest(old) == old['plan_sha256']
    old_root = ROOT / 'artifacts/experiments' / NAME / old['plan_sha256']
    assert not (old_root / 'queue_status.json').exists() and not (old_root / 'runs').exists()  # never trained
    old_pre = read(old_root / 'PREFLIGHT_COMPLETE.json')
    assert old_pre['failed'] == ['PEMSBAY_ALFNet', 'PEMS08_ALFNet'], old_pre['failed']
    evidence = sorted(DIAGNOSTICS.glob('*'))
    assert any(f.name == 'epoch0_PEMS08_wrong_path_support_only.json' for f in evidence)
    sources = seed32_sources()
    audit = historical_audit(sources)
    assert not any(r['reusable'] for r in audit), 'a historical run qualifies: add it to reuse before freezing'
    files = [Path(__file__), SELECTION, AUDIT, FINAL, SELECTED_TABLES, EXT2_TEST, SEEDS1_PLAN, SUPERSEDED,
             old_root / 'PREFLIGHT_COMPLETE.json', *evidence]
    configs, reuse = {}, {}
    for (ds, m), s in sorted(sources.items()):
        reuse[f'{ds}_{m}_seed32'] = dict(s, dataset=ds, model=m, seed=32)
        files += [ROOT / s['config'], ROOT / s['checkpoint'], ROOT / s['report']]
        if m == 'ALFNet':
            files += nested_configs(s['config'])
        for seed in NEW_SEEDS + (32,):
            configs[f'{ds}_{m}_seed{seed}'] = dict(
                path=s['config'], sha256=s['config_sha256'], dataset=ds, model=m, seed=seed, kind=m,
                study_context=s['study_context'], conditional=seed == 32,
                condition='only if the seed-32 replay does not reproduce' if seed == 32 else None)
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider'):
        files.extend(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts)
    # scripts/: only what training, replay and test import
    for name in ('scripts.train_v4', 'scripts.eval_release', 'experiments.v4_runner',
                 'experiments.v4_validation_collector', 'scripts.run_extension2_20261105',
                 'experiments.flow_support_transfer', 'experiments.flow_unified'):
        importlib.import_module(name)
    files.extend(Path(mod.__file__) for mod in list(sys.modules.values())
                 if getattr(mod, '__file__', None) and Path(mod.__file__).resolve().parent == ROOT / 'scripts')
    files.append(ROOT / 'args.py')
    for ds in DATASETS:
        folder, _ = extension.DATA[ds]
        files += [ROOT / f'data_v4/{folder}/raw_data.pkl', ROOT / f'data_v4/{folder}/unknown_nodes.npy',
                  ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json']
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = dict(kind=NAME, status='authorized_training_then_one_test', authority=AUTHORITY,
             alfnet_selection=read(SELECTION), seeds=[32, 42, 52], new_seeds=list(NEW_SEEDS),
             max_concurrent=MAX_CONCURRENT, configs=configs, reuse=reuse, historical_seed_runs=audit,
             selection='validation 33-cell main/LifeAvg/MAE, strict, earlier epoch on ties',
             replay_gate='seed-32 checkpoint replay with current code must agree within abs 1e-5 + rel 1e-5',
             alfnet_training_path=dict(rule='each ALFNet run in the study context of its seed-32 run; before queueing, '
                                            'one full seed-32 epoch must reproduce the original epoch-0 record: '
                                            'counts, method contract and parameter count exact; epoch-mean training '
                                            f'losses within rel {TRAIN_REL:g} + abs {REPRO_ABS:g}; validation metrics '
                                            f'within the sanity bound rel {VALIDATION_SANITY_REL:g} (GPU nondeterminism '
                                            'alone moves them by up to 2-4% after one epoch on PEMS-BAY/PEMS08)',
                                       contexts={d: sources[(d, 'ALFNet')]['study_context'] for d in DATASETS}),
             superseded=dict(plan_sha256=old['plan_sha256'], file=str(SUPERSEDED.relative_to(ROOT)),
                             reason='epoch-0 reproduction gate held one-epoch validation metrics to rel 1e-3; '
                                    'PEMS-BAY (2.1e-2) and PEMS08 (2.5e-2) failed. Diagnostics: independent '
                                    'current-code reproductions are bit-identical for 5-10 batches, then drift to '
                                    '1-4% validation differences, the original inside their spread; epoch-mean '
                                    'training losses within 1.2e-4 of the original; a wrong path (no flow_unified) '
                                    'moves them +24%. No training was started.',
                             preflight=str((old_root / 'PREFLIGHT_COMPLETE.json').relative_to(ROOT)),
                             evidence=str(DIAGNOSTICS.relative_to(ROOT))),
             training='production runner (skips counted and reported; >1% training skips flags divergence); '
                      'MCPST/MetaDG in run_extension2_20261105.run_context()',
             nonfinite_gradient_guard=dict(rule='skip and record optimiser steps whose gradients are not all finite; '
                                                f'fail if one epoch exceeds {MAX_EVENT_FRACTION:.0%} of its batches',
                                           applies_to='every run of this plan (no-op without a non-finite gradient)',
                                           implementation='scripts/run_seeds_20261106.py nonfinite_gradient_guard'),
             git_head=git, torch=torch.__version__,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(configs), 'config entries;', sum(not c['conditional'] for c in configs.values()),
          'new runs')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    return p


def train_seed(p, key, root, smoke=False, full_epoch=False):
    from scripts import train_v4 as training
    row = p['configs'][key]
    short = smoke and not full_epoch
    with context(row['model'], row['study_context']), seeds1.nonfinite_gradient_guard(training):
        return training.train(ROOT / row['path'], root, device='cuda:0', seed=row['seed'], num_workers=0, plan=p,
                              save=not smoke, collect_development_table=not smoke, max_epochs=1 if smoke else None,
                              max_train_batches=20 if short else None, max_validation_batches=4 if short else None)


def replay_seed32(p, ds, m):
    import torch
    from experiments import v4_runner
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.eval_release import validation_agrees
    from scripts.run_metrla_hparams import identity
    s32 = p['reuse'][f'{ds}_{m}_seed32']
    cp = torch.load(ROOT / s32['checkpoint'], map_location='cpu', weights_only=False)
    with context(m, s32['study_context']):
        s = v4_runner.build_v4_session(ROOT / s32['config'], device='cuda:0', run_seed=32, num_workers=0)
        ids = identity(s)
        assert cp['scaler'] == ids['scaler'] and cp['train_exposure_hash'] == ids['train_exposure_hash']
        assert cp['validation_exposure_hash'] == ids['validation_exposure_hash']
        s.model.load_state_dict(cp['model_state'], strict=True)
        s.model.eval()
        seed_process(s.seed_contract['derived']['training_runtime'])
        table = collect_validation_table(s, epoch=s32['epoch'])
    ok, delta = validation_agrees(table, s32['validation_table'])
    return dict(passed=bool(ok), max_abs_difference=delta)


def compare_epoch0(new, old):
    """Epoch-0 record of a fresh seed-32 epoch vs the original run (fields of the original record).

    Non-float fields (counts, policies, seeds) must be equal; float training losses within rel TRAIN_REL;
    float validation metrics within the sanity bound rel VALIDATION_SANITY_REL (see the module docstring).
    """
    problems = []
    worst = {part: dict(rel=0., field=None) for part in ('train', 'validation')}
    for part in ('train', 'validation'):
        bound = TRAIN_REL if part == 'train' else VALIDATION_SANITY_REL
        for k, v in old[part].items():
            n = new[part].get(k)
            if isinstance(v, float) and not isinstance(v, bool):
                d = abs(n - v)
                rel = d / max(abs(v), 1e-12)
                if rel > worst[part]['rel']:
                    worst[part] = dict(rel=rel, field=k, new=n, old=v)
                if d > REPRO_ABS + bound * abs(v):
                    problems.append(f'{part}.{k}: {n!r} vs {v!r}')
            elif n != v:
                problems.append(f'{part}.{k}: {n!r} vs {v!r}')
    if new['train'].get('nonfinite_gradient_steps'):
        problems.append('non-finite gradient steps in the reproduction epoch')
    return problems, worst


def reproduce_epoch0(p, ds):
    """One full seed-32 epoch on the planned training path vs the original epoch-0 record."""
    key = f'{ds}_ALFNet_seed32'
    original = read(ROOT / p['reuse'][key]['report'])
    report = train_seed(p, key, None, smoke=True, full_epoch=True)
    problems, worst = compare_epoch0(report['history'][0], original['history'][0])
    if report['model_method_contract'] != original['model_method_contract']:
        problems.append('model_method_contract differs')
    if report['parameter_count'] != original['parameter_count']:
        problems.append(f"parameter_count {report['parameter_count']} vs {original['parameter_count']}")
    return dict(passed=not problems, problems=problems, worst_relative=worst,
                new=report['history'][0], original=original['history'][0])


def pairs(p):
    return sorted({(c['dataset'], c['model']) for c in p['configs'].values()},
                  key=lambda x: (MODELS.index(x[1]), DATASETS.index(x[0])))


def preflight(shard='0/1'):
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
            if m == 'ALFNet':
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                info['epoch0_reproduction'] = reproduce_epoch0(p, ds)
                info['epoch0_reproduction'].update(seconds=time.time() - start,
                                                   peak=torch.cuda.max_memory_reserved() / 1024 ** 3)
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            report = train_seed(p, f'{ds}_{m}_seed42', None, smoke=True)
            info.update(passed=info.get('epoch0_reproduction', {}).get('passed', True),
                        smoke_epochs=report['epochs_run'], seconds=time.time() - start,
                        smoke_validation_mae=report['history'][0]['validation']['mae'],
                        smoke_guard_events=len(report['history'][0]['train'].get('nonfinite_gradient_steps', [])),
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
            if m == 'ALFNet':  # the reproduction epoch is a full epoch: its memory is the relevant peak
                info['peak_reserved_gib'] = max(info['peak_reserved_gib'], info['epoch0_reproduction'].get('peak', 0))
        except Exception:
            info.update(passed=False, error=traceback.format_exc()[-6000:])
        torch.cuda.empty_cache()
        write(dest, info)
        print('PREFLIGHT_DONE', ds, m, info['passed'], info.get('replay_seed32'),
              info.get('epoch0_reproduction', {}).get('worst_relative'), flush=True)


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
    """Seeds 42/52 of every pair that passed preflight, plus seed 32 where the replay gate failed."""
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    retrain = set(pre['retrain_seed32'])
    keys = []
    for k, c in p['configs'].items():
        pair = f"{c['dataset']}_{c['model']}"
        if pre['reports'].get(pair, {}).get('passed') and (not c['conditional'] or pair in retrain):
            keys.append(k)
    keys.sort(key=lambda k: (-HOURS[(p['configs'][k]['model'], p['configs'][k]['dataset'])], p['configs'][k]['seed']))
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
        shards = min(MAX_CONCURRENT, len(pairs(p)))
        procs = [launch(['preflight', '--shard', f'{i}/{shards}'], root / f'preflight_{i}.log', env)
                 for i in range(shards)]
        write(root / 'preflight_status.json', dict(queue_pid=os.getpid(), pids=[q.pid for q, _ in procs],
                                                  started=time.time(), shards=shards))
        for q, log in procs:
            q.wait()
            log.close()
        preflight_complete(p, root)
    pre = read(root / 'PREFLIGHT_COMPLETE.json')
    if not pre['passed']:  # nothing is trained unless every pair (incl. the ALFNet epoch-0 reproduction) passed
        write(root / 'ATTENTION_REQUIRED.json', dict(preflight_failed=pre['failed'], missing=pre['missing'],
                                                     test_accessed=False))
        raise SystemExit('preflight failed; no training queued')
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
                proc, log = launch(['worker', '--key', key], root / f'{key}.log', env)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need,
                               log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_seeds2_20261110.py worker --key {key}')
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
        train_batches = sum(h['train']['batches'] + h['train']['skipped'] for h in r['history'])
        guard_steps = [g for h in r['history'] for g in h['train'].get('nonfinite_gradient_steps', [])]
        if train_skipped > 0.01 * train_batches:
            flags.append(dict(key=key, train_skipped=train_skipped, train_batches=train_batches))
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'], train_skipped=train_skipped,
                         nonfinite_gradient_steps=guard_steps,
                         validation_skipped=sorted({h['validation']['skipped'] for h in r['history']}))
    # every model of one (dataset, seed) is scored on identical cells, also against the 2026-11-06 study
    seeds1_rows = read(seeds1.output(read(SEEDS1_PLAN)) / 'validation_results.json')['rows']
    for ds in DATASETS:
        for seed in (32,) + NEW_SEEDS:
            tables = [r['table'] for r in rows.values() if r['dataset'] == ds and r['seed'] == seed]
            tables += [r['table'] for r in seeds1_rows.values() if r['dataset'] == ds and r['seed'] == seed][:1]
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
    parser.add_argument('mode', choices=['select', 'freeze', 'check', 'preflight', 'queue', 'worker', 'collect'])
    parser.add_argument('--key')
    parser.add_argument('--shard', default='0/1')
    parser.add_argument('--choice', nargs='*', help='DATASET=audit candidate key (mode select)')
    parser.add_argument('--authority', help="the user's decision, quoted (mode select)")
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.key)
    elif a.mode == 'preflight':
        preflight(a.shard)
    elif a.mode == 'select':
        select(a.choice, a.authority)
    elif a.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[a.mode]()
