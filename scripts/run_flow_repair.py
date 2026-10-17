"""Nine validation-only flow repairs, isolated from the frozen shared model."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hparam_completion as previous
from scripts.run_metrla_hparams import read, write, sha, digest
from scripts.collect_hparams_final import params
from scripts.reselect_eight_baselines import SELECTED, EXCLUDED, audit
from experiments.flow_repair_20261002 import NAME as LOSS, training_adapter

NAME = 'flow_repair_20261002'
DIR = ROOT/'configs/paper'/NAME
PLAN = DIR/'plan.json'
BASE_TABLE = ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json'
VARIANTS = ('unit_focal', 'unit_no_role', 'unit_no_refiner')
DATASETS = ('PEMS03', 'PEMS04', 'PEMS08')


def output(p):
    return ROOT/'artifacts/experiments'/NAME/p['plan_sha256']


def freeze():
    assert not PLAN.exists()
    from scripts import test_all_final_20261001 as old
    old.check()
    prior_path = ROOT/'paper/hparams_combinations_20260930/results.json'
    prior = read(prior_path)['rows']
    DIR.mkdir(parents=True, exist_ok=True)
    files = [prior_path, BASE_TABLE, ROOT/'paper'/NAME/'PLAN.md']
    entries, anchors, reused = {}, {}, {}
    for ds in DATASETS:
        key = SELECTED[ds]; r = prior[key]
        anchors[ds] = dict(config=r['config'], checkpoint=r['checkpoint'], table=r['table'])
        reused[key] = r
        base = yaml.safe_load((ROOT/r['config']).read_text())
        hp = params(r)
        files.extend([ROOT/r['config'], ROOT/r['checkpoint'], ROOT/r['report']])
        files.extend(ROOT/base['family']['experts'][x]['config_path'] for x in ('graph', 'patch'))
        files.extend(ROOT/base['dataset'][x] for x in ('data_path', 'unknown_nodes_path'))
        files.append(ROOT/f'configs/paper/v4_data_manifests/{ds}_run_manifest.json')
        for i, variant in enumerate(VARIANTS):
            key = f'{i*3+DATASETS.index(ds):02d}_{ds}_{variant}'
            cfg = copy.deepcopy(base)
            cfg['runtime']['model_variant'] = NAME+'_'+key
            cfg['training']['loss'] = LOSS
            if variant == 'unit_no_role':
                cfg['family']['recipe']['forecast_role_balanced_beta'] = 0.
            elif variant == 'unit_no_refiner':
                cfg['family']['recipe']['graph_diffusion_scale'] = 0.
            path = DIR/(key+'.yaml')
            assert not path.exists()
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            entries[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                dataset=ds, model='ALFNet', seed=32, axis='method_repair', value=variant,
                capacity=hp['capacity'], dropout=hp['dropout'], weight_decay=hp['weight_decay'],
                lrs=[hp[k] for k in ('graph_lr', 'temporal_lr', 'gate_lr')])
            files.append(path)
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider', 'scripts'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend([ROOT/'args.py', ROOT/'tests/test_flow_repair.py'])
    p = dict(kind=NAME, status='authorized_validation_only', configs=entries, anchors=anchors,
        reused=reused, max_concurrent=5, test_authorized=False, seed=32,
        authority='2026-10-02 user: try multiple flow repairs, max5, no comparison algorithm retraining',
        selection='checkpoint full validation33 LifeAvg MAE; report coverage and LifeAvg tradeoffs without hard regression gates',
        test_history='Old test observed for diagnosis/anchors; new candidates validation-only, no independent-test claim',
        files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'])


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    assert len(p['configs']) == 9 and p['max_concurrent'] == 5 and not p['test_authorized']
    for f, h in p['files'].items():
        assert sha(ROOT/f) == h, f
    for r in p['configs'].values():
        assert r['model'] == 'ALFNet' and r['dataset'] in DATASETS
        cfg = yaml.safe_load((ROOT/r['path']).read_text())
        assert cfg['training']['loss'] == LOSS
    return p


def context():
    stack = ExitStack()
    for name, value in [('check', check), ('output', output), ('NAME', NAME), ('BASE_TABLE', BASE_TABLE)]:
        stack.enter_context(patch.object(previous, name, value))
    stack.enter_context(training_adapter())
    return stack


def preflight():
    with context():
        previous.preflight()


def worker(key):
    with context():
        previous.worker(key)


def collect():
    p = check(); root = output(p); path = ROOT/'paper'/NAME/'results.json'
    rows = copy.deepcopy(p['reused'])
    for key, entry in p['configs'].items():
        done = read(root/(key+'_complete.json'))
        assert done['passed']
        rp = Path(done['report_path']); report = read(rp)
        assert report['plan_sha256'] == p['plan_sha256']
        assert report['config_sha256'] == entry['sha256']
        assert report['model_method_contract']['flow_repair_20261002']['loss'] == LOSS
        assert all(h[split]['skipped'] == 0 for h in report['history'] for split in ('train', 'validation'))
        assert min((h['validation']['checkpoint_life_avg_mae'], h['epoch'])
                   for h in report['history'])[1] == report['best_epoch']
        rows[key] = dict(dataset=entry['dataset'], config=entry['path'],
            table=report['development_validation_table'], report=str(rp.relative_to(ROOT)),
            checkpoint=str(rp.with_name('best.pt').relative_to(ROOT)),
            checkpoint_sha256=sha(rp.with_name('best.pt')), variant=entry['value'])
    data = dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                test_accessed=False, automatic_final_replacement=False)
    bases = read(BASE_TABLE)['records']; ranking = {}
    for ds in DATASETS:
        baseline = {n: r for n, r in bases[ds].items() if n not in EXCLUDED}
        keys = [k for k, r in data['rows'].items() if r['dataset'] == ds]
        ranking[ds] = audit(data['rows'], keys, baseline)
        for k, r in ranking[ds].items():
            data['rows'][k]['first_cells'] = r['first_cells']
    data['ranking_eight_baselines'] = ranking
    data['summary'] = {ds: dict(most_first_cells=next(iter(r)),
                      min_lifeavg_mae=min(r, key=lambda k: r[k]['lifeavg']['MAE'])) for ds, r in ranking.items()}
    write(path, data)
    lines = ['# Flow修复validation结果', '', '旧对照3项＋新修复9项；未打开test，不自动替换论文表。', '',
             '| run | 第一/33 | LifeAvg MAE | RMSE | MAPE |', '|---|---:|---:|---:|---:|']
    for ds, records in ranking.items():
        for k, r in records.items():
            lines.append(f'| {k} | {r["first_cells"]} | '+
                         ' | '.join(f'{r["lifeavg"][m]:.5f}' for m in ('MAE', 'RMSE', 'MAPE'))+' |')
    (path.parent/'results_summary.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json', dict(passed=True, new_runs=9, reused_controls=3, test_accessed=False))


def queue():
    p = check()
    if not (output(p)/'PREFLIGHT_COMPLETE.json').exists():
        preflight()
    from scripts import recover_baseline_extension as runner
    with ExitStack() as stack:
        for name, value in [('check', check), ('output', output), ('collect', collect),
                            ('__file__', __file__), ('isolate_failed_families', lambda jobs, configs: None)]:
            stack.enter_context(patch.object(runner, name, value))
        runner.queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'check', 'preflight', 'worker', 'queue', 'collect'])
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.key)
    elif args.mode == 'check':
        print(check()['plan_sha256'])
    else:
        globals()[args.mode]()
