"""2026-11-10 audit of every trained seed-32 ALFNet setting candidate, before the ALFNet three-seed study.

Read-only over saved artefacts (mode ``inventory``, CPU): no training, no inference,
no test access. Optional mode ``replay --keys ...`` (GPU) rebuilds a candidate in the
study context it was trained in and replays its validation table with the current
code (abs 1e-5 + rel 1e-5, scripts/eval_release.validation_agrees); still no test.

Candidates: every ALFNet seed-32 run_report.json under artifacts/ on METR-LA, PEMS-BAY,
PEMS04, PEMS08 that is a hyperparameter/input-handling setting of the model, i.e. not a
component ablation (studies listed in ABLATION_STUDIES are excluded and counted).
Comparability under the current protocol:
  * selector: validation 33-cell main/LifeAvg/MAE, strict, earlier epoch on ties, and the
    recorded best epoch is the argmin of the saved history;
  * data/split: payload, unseen-node file, manifest, static assets, train/validation
    exposure hashes, scaler and seed contract equal those of the current final checkpoint;
  * replay: the saved best.pt reproduces the recorded validation table with current code
    (only when ``replay`` has been run for that key; otherwise "not run").
Ranking (validation): strict first cells among 33 against the reported comparison set
(8 paper baselines + Merlin + STONE + MCPST + MetaDG, seed 32; GinAR on METR-LA/PEMS-BAY at
lr .001 as in the three-seed study, lr .006 paper run as a sensitivity), plus LifeAvg
MAE/RMSE/MAPE. Existing test records of a candidate are listed as facts only.
Output: paper/alfnet_config_audit_20261110/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, sha, write  # noqa: E402

OUT = ROOT / 'paper/alfnet_config_audit_20261110'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
FINAL = ROOT / 'paper/pems08_wd_test_20261012/selected_configuration_manifest.json'
SECTIONS = {'main': ('Unseen', 'Steady', 'Join', 'Fault', 'LifeAvg'),
            'phase': ('Cold', 'Warm', 'Ready', 'Onset', 'Decay', 'Outage')}
METRICS = ('MAE', 'RMSE', 'MAPE')
CELLS = [(g, c, m) for g, cs in SECTIONS.items() for c in cs for m in METRICS]
EXISTING = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP')
ADDED = ('Merlin', 'STONE', 'MCPST', 'MetaDG')
ABLATION_STUDIES = ('final_ablations_20261006', 'ablation_recovery_20261008', 'ablation_repeat_20261012',
                    'mechanism_ablations_20261008', 'metrla_final_ablations_20260920', 'pems08_wd_ablations_20261012',
                    'reconstruction_gradient_20261009', 'extension_20261103')
HISTORY_ABLATIONS = ('mask_only', 'local_temporal', 'static_graph')
IDENTITY = ('payload_sha256', 'unseen_sha256', 'manifest_sha256', 'static_asset_hash', 'train_exposure_hash',
            'validation_exposure_hash', 'scaler', 'seed_contract')
SELECTOR = 'validation 33-cell main/LifeAvg/MAE'
# Study -> how its ALFNet sessions were built at training time (verified from scripts/run_*.py).
STUDY_CONTEXT = {
    'metrla_hparams_20260920': 'bare', 'pemsbay_hparams_20260917': 'bare', 'hparams_completion_20260928': 'bare',
    'hparams_combinations_20260930': 'bare', 'flow_extension_20260926': 'bare', 'main': 'bare',
    'flow_support_transfer_20261003': 'support', 'flow_unified_20261003': 'unified',
    'flow_ofat_20261006': 'support+unified', 'speed_support_20261004': 'support+unified',
    'flow_lr_cross_20261004': 'support+unified', 'flow_repair_20261002': 'repair',
    'flow_last_round_20261002': 'last_round',
}


def value(t, cell):
    g, c, m = cell
    return t[g][c][m]


def candidates():
    """All non-ablation ALFNet seed-32 runs of the four datasets."""
    out, excluded = {}, []
    for f in sorted(ROOT.glob('artifacts/**/run_report.json')):
        r = read(f)
        if r.get('model') != 'ALFNet' or r.get('dataset') not in DATASETS:
            continue
        rel = f.relative_to(ROOT)
        study = rel.parts[2]  # artifacts/{experiments,models}/<study>/...
        if r.get('seed') != 32:
            excluded.append(dict(run=str(rel.parent), reason=f"seed {r.get('seed')}"))
            continue
        if study in ABLATION_STUDIES or (study == 'history' and any(a in str(r.get('config')) for a in HISTORY_ABLATIONS)):
            excluded.append(dict(run=str(rel.parent), reason='component ablation, not a setting candidate'))
            continue
        key = f"{study}/{Path(r['config']).stem}" if study not in ('main', 'history') else f"{study}/{r['dataset']}"
        assert key not in out, key
        out[key] = dict(key=key, study=study, dataset=r['dataset'], run=str(rel.parent), report=str(rel))
    return out, excluded


def config_files(c):
    """The trained config bytes: the plan's YAML, or for artifacts/models the stored original copy."""
    run = ROOT / c['run']
    r = read(ROOT / c['report'])
    if (ROOT / r['config']).exists():
        cfg_path = ROOT / r['config']
        cfg = yaml.safe_load(cfg_path.read_text())
        patch_path = ROOT / cfg['family']['experts']['patch']['config_path']
        graph_path = ROOT / cfg['family']['experts']['graph']['config_path']
    else:
        cfg_path = run / 'config.original.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        patch_path, graph_path = run / 'expert.patch.original.yaml', run / 'expert.graph.original.yaml'
    graph_scale = yaml.safe_load(graph_path.read_text())['family']['recipe'].get('decoder_residual_scale', 0.3)
    return cfg_path, cfg, yaml.safe_load(patch_path.read_text())['family']['model'], graph_scale


def settings(c, report):
    cfg_path, cfg, patch, graph_scale = config_files(c)
    recipe = cfg['family']['recipe']
    lrs = [recipe[k] for k in ('graph_expert_lr', 'patch_expert_lr', 'gate_lr')]
    factors = [a / b for a, b in zip(lrs, (5e-4, 1e-4, 5e-5))]
    mc = report['model_method_contract']
    st, fu = mc.get('support_transfer'), mc.get('flow_unified')
    bridge = bool((st or {}).get('bridge') or (fu or {}).get('bridge'))
    if st:
        decoder, center = st['decoder_scale'], 1.0 if st['center'] else 0.0
    elif fu:
        decoder, center = fu['decoder_scale'], fu['center_strength']
    else:
        decoder, center = graph_scale, 0.0
    extra = []
    if fu and 'spatial' in fu['variant']:
        extra.append('spatial unseen center')
    if fu and 'qdelta' in fu['variant']:
        extra.append('visible-fraction decoder scale')
    for name in ('flow_last_round', 'flow_repair_20261002'):
        if name in mc:
            extra.append(f"{name}: " + json.dumps({k: v for k, v in mc[name].items() if v not in (None, 0, 0.0, False)},
                                                  sort_keys=True)[:160])
    if mc.get('paired_ablation'):
        extra.append(f"ablation {mc['paired_ablation']}")
    t = cfg['training']
    return dict(config=str(cfg_path.relative_to(ROOT)), config_sha256=sha(cfg_path),
                capacity=patch['spa_patchsize'], lr_multiplier=round(factors[0], 6),
                lr_groups_consistent=max(factors) - min(factors) < 1e-12, dropout=patch['crossformer_dropout'],
                weight_decay=t['weight_decay'], batch_size=t['batch_size'], max_epoch=t['max_epoch'],
                patience=t['patience'], early_stopping_start_epoch=t.get('early_stopping_start_epoch'),
                loss=t['loss'], interpolation=bridge, decoder_scale=decoder, center_alpha=center,
                support_variant=(st or {}).get('variant') or (fu or {}).get('variant'),
                other_switches=extra, accepted_parameterization=not extra and t['loss'] == 'mae_focal_raw02')


def selector_check(r):
    hist = [(h['validation']['checkpoint_life_avg_mae'], h['epoch']) for h in r['history']]
    start = r.get('early_stopping_start_epoch') or 0
    eligible = [x for x in hist if x[1] >= 0]
    argmin = min(eligible)[1]
    issues = []
    if r.get('early_stopping_metric') != SELECTOR:
        issues.append(f"selector {r.get('early_stopping_metric')!r}")
    if r.get('tie_rule') != 'keep_earlier_epoch':
        issues.append(f"tie rule {r.get('tie_rule')!r}")
    if argmin != r['best_epoch']:
        issues.append(f"best_epoch {r['best_epoch']} != argmin {argmin}")
    return dict(ok=not issues, issues=issues, early_stopping_start_epoch=start, epochs_run=r['epochs_run'],
                best_epoch=r['best_epoch'])


def comparison_tables():
    """{split: {dataset: {method: table}}} at seed 32 (GinAR lr .001 on speed) + GinAR lr .006 sensitivity."""
    from scripts import report_extension2_20261105 as comparison
    from scripts import report_seeds_20261106 as seeds
    _, tables, _ = comparison.load()
    _, found = seeds.tables()
    out, paper_ginar = {}, {}
    for split in ('validation', 'test'):
        out[split], paper_ginar[split] = {}, {}
        for d in DATASETS:
            out[split][d] = {m: tables[split][d][m] for m in EXISTING + ADDED}
            paper_ginar[split][d] = tables[split][d]['GinAR']
            out[split][d]['GinAR'] = found[(d, 'GinAR', 32)][split]
            if d in ('PEMS04', 'PEMS08'):
                assert out[split][d]['GinAR'] == paper_ginar[split][d]
    return out, paper_ginar


def first_cells(table, others):
    won, lost = [], {}
    for cell in CELLS:
        best = min((value(t, cell), m) for m, t in others.items())
        if value(table, cell) < best[0]:
            won.append(cell)
        else:
            lost.setdefault(best[1], []).append('/'.join(cell[1:]))
    return won, lost


def counts_ok(table, ref):
    return all(table[g][c][f] == ref[g][c][f] for g in SECTIONS for c in ref[g] for f in ref[g][c] if f.endswith('_count'))


def test_records(shas):
    """checkpoint sha256 -> [test records] from every saved evaluation JSON (facts only)."""
    found = {}

    def walk(x, path):
        if isinstance(x, dict):
            sh = x.get('checkpoint_sha256')
            t = x.get('table')
            if isinstance(sh, str) and sh in shas and isinstance(t, dict) and 'main' in t and \
                    (x.get('split') == 'test' or t.get('split') == 'test'):
                found.setdefault(sh, {})[str(path.relative_to(ROOT))] = t
            for v in x.values():
                walk(v, path)
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
    for f in sorted((ROOT / 'artifacts/evaluation').rglob('*.json')):
        try:
            walk(read(f), f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return found


def lifeavg(t):
    return {m: t['main']['LifeAvg'][m] for m in METRICS}


def inventory():
    import torch
    final = read(FINAL)['entries']
    anchors = {}
    for d in DATASETS:
        cp = torch.load(ROOT / final[d]['checkpoint'], map_location='cpu', weights_only=False)
        assert sha(ROOT / final[d]['checkpoint']) == final[d]['checkpoint_sha256']
        anchors[d] = {k: cp[k] for k in IDENTITY}
        del cp
    comp, paper_ginar = comparison_tables()
    cands, excluded = candidates()
    rows = {}
    for key, c in cands.items():
        r = read(ROOT / c['report'])
        best = ROOT / c['run'] / 'best.pt'
        row = dict(c, checkpoint=str(best.relative_to(ROOT)) if best.exists() else None)
        row.update(settings(c, r))
        row['plan_sha256'] = r.get('plan_sha256')
        row['selector'] = selector_check(r)
        table = r['development_validation_table']
        dvt = ROOT / c['run'] / 'development_validation_table.json'
        row['table_file_matches_report'] = read(dvt) == table if dvt.exists() else None
        problems = []
        if best.exists():
            row['checkpoint_sha256'] = sha(best)
            cp = torch.load(best, map_location='cpu', weights_only=False)
            diff = [k for k in IDENTITY if cp.get(k) != anchors[c['dataset']][k]]
            row['identity_differs'] = diff
            row['checkpoint_epoch'] = cp.get('epoch')
            row['git_commit'] = cp.get('git_commit')
            if cp.get('epoch') != r['best_epoch']:
                problems.append(f"checkpoint epoch {cp.get('epoch')} != report best {r['best_epoch']}")
            if cp.get('config_sha256') != row['config_sha256']:
                problems.append('checkpoint config_sha256 differs from the config bytes on disk')
            del cp
        else:
            row['checkpoint_sha256'], row['identity_differs'] = None, None
            problems.append('best.pt missing (deleted in the 2026-10-17 cleanup?)')
        if row['identity_differs']:
            problems.append('data/split identity differs: ' + ', '.join(row['identity_differs']))
        if not row['selector']['ok']:
            problems += row['selector']['issues']
        if not counts_ok(table, comp['validation'][c['dataset']]['UIGNN']):
            problems.append('validation cell counts differ from the comparison methods')
        row['comparable_problems'] = problems
        row['validation_lifeavg'] = lifeavg(table)
        others = comp['validation'][c['dataset']]
        won, lost = first_cells(table, others)
        row['validation_first'] = len(won)
        row['validation_lost_to'] = lost
        sens = dict(others, GinAR=paper_ginar['validation'][c['dataset']])
        row['validation_first_ginar_lr006'] = len(first_cells(table, sens)[0])
        row['validation_first_existing8'] = len(first_cells(table, {m: others[m] for m in EXISTING})[0])
        row['validation_table'] = table
        row['is_final'] = row['checkpoint_sha256'] == final[c['dataset']]['checkpoint_sha256']
        rows[key] = row
    tests = test_records({r['checkpoint_sha256'] for r in rows.values() if r['checkpoint_sha256']})
    for f in sorted((OUT / 'new_tests').glob('*.json')) if (OUT / 'new_tests').exists() else []:
        rec = read(f)  # opened by this audit (mode test), listed like any other test record
        tests.setdefault(rec['checkpoint_sha256'], {})[str(f.relative_to(ROOT))] = rec['table']
    for row in rows.values():
        recs = tests.get(row['checkpoint_sha256'], {})
        uniq = []
        for path, t in recs.items():
            if all(t != u[1] for u in uniq):
                uniq.append((path, t))
        row['test_records'] = sorted(recs)
        row['test_tables_distinct'] = len(uniq)
        if uniq:
            t = uniq[0][1]
            others = comp['test'][row['dataset']]
            won, lost = first_cells(t, others)
            row['test_first'] = len(won)
            row['test_first_existing8'] = len(first_cells(t, {m: others[m] for m in EXISTING})[0])
            row['test_lifeavg'] = lifeavg(t)
            row['test_table'] = t
    replays = {f.stem: read(f) for f in (OUT / 'replay').glob('*.json')} if (OUT / 'replay').exists() else {}
    for key, row in rows.items():
        rp = replays.get(key.replace('/', '__'))
        row['replay'] = None if rp is None else {k: rp[k] for k in ('passed', 'max_abs_difference', 'context', 'error')
                                                 if k in rp}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'candidates.json', dict(
        generated='2026-11-10', final_manifest=str(FINAL.relative_to(ROOT)), selector=SELECTOR,
        comparison_set=list(EXISTING + ADDED), ginar_speed='lr .001 seed 32 (three-seed study); lr .006 paper run as sensitivity',
        identity_fields=list(IDENTITY), excluded=excluded, rows=rows))
    summary(rows)


def summary(rows):
    lines = []
    for d in DATASETS:
        rs = sorted((r for r in rows.values() if r['dataset'] == d),
                    key=lambda r: (-r['validation_first'], r['validation_lifeavg']['MAE']))
        lines.append(f'\n## {d}\n')
        lines.append('| 候选 | b | s | α | cap | LR× | drop | WD | 其他 | 可比问题 | 重放 | 验证第一/33 | (GinAR .006) | 验证 LifeAvg MAE/RMSE/MAPE | 测试第一/33 | 测试 LifeAvg MAE/RMSE/MAPE |')
        lines.append('|' + '---|' * 16)
        for r in rs:
            v = r['validation_lifeavg']
            t = r.get('test_lifeavg')
            rp = r['replay']
            lines.append('| {}{} | {} | {} | {} | {} | {} | {} | {:g} | {} | {} | {} | {} | {} | {:.4f}/{:.4f}/{:.4f} | {} | {} |'.format(
                r['key'], ' **(final)**' if r['is_final'] else '', int(r['interpolation']), r['decoder_scale'],
                r['center_alpha'], r['capacity'], r['lr_multiplier'], r['dropout'], r['weight_decay'],
                '; '.join(r['other_switches']) or '-', '; '.join(r['comparable_problems']) or 'ok',
                'not run' if rp is None else ('ok' if rp['passed'] else f"FAIL {rp.get('max_abs_difference')}"),
                r['validation_first'], r['validation_first_ginar_lr006'], v['MAE'], v['RMSE'], v['MAPE'],
                r.get('test_first', '-'), '-' if t is None else f"{t['MAE']:.4f}/{t['RMSE']:.4f}/{t['MAPE']:.4f}"))
    (OUT / 'candidates_summary.md').write_text('# ALFNet seed-32 candidates (2026-11-10 audit)\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


def contexts(name):
    from contextlib import ExitStack
    stack = ExitStack()
    if name in ('support', 'support+unified'):
        from experiments.flow_support_transfer import adapter
        stack.enter_context(adapter())
    if name in ('unified', 'support+unified'):
        from experiments.flow_unified import adapter
        stack.enter_context(adapter())
    if name == 'repair':
        from experiments.flow_repair_20261002 import training_adapter
        stack.enter_context(training_adapter())
    if name == 'last_round':
        from experiments.flow_last_round import adapter
        stack.enter_context(adapter())
    return stack


def replay(keys):
    import traceback
    import torch
    from experiments import v4_runner
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.eval_release import validation_agrees
    torch.set_num_threads(4)
    rows = read(OUT / 'candidates.json')['rows']
    for key in keys:
        row = rows[key]
        dest = OUT / 'replay' / f"{key.replace('/', '__')}.json"
        if dest.exists():
            continue
        ctx = STUDY_CONTEXT[row['study']]
        info = dict(key=key, context=ctx, checkpoint_sha256=row['checkpoint_sha256'], pid=os.getpid())
        try:
            assert sha(ROOT / row['checkpoint']) == row['checkpoint_sha256']
            cp = torch.load(ROOT / row['checkpoint'], map_location='cpu', weights_only=False)
            config = ROOT / row['config']
            if row['study'] == 'main':
                # The trained YAML points at nested configs that were moved; the release copy
                # (provenance 'config') and its nested configs have identical parsed content.
                run = ROOT / row['run']
                config = ROOT / read(run / 'provenance.json')['config']
                moved = yaml.safe_load(config.read_text())
                assert moved == yaml.safe_load((run / 'config.yaml').read_text())
                for expert in ('graph', 'patch'):
                    assert yaml.safe_load((ROOT / moved['family']['experts'][expert]['config_path']).read_text()) == \
                        yaml.safe_load((run / f'expert.{expert}.original.yaml').read_text())
                info['replay_config'] = str(config.relative_to(ROOT)) + ' (content-equal; nested configs moved)'
            with contexts(ctx):
                s = v4_runner.build_v4_session(config, device='cuda:0', run_seed=32, num_workers=0)
                assert cp['scaler'] == s.scaler and cp['seed_contract'] == s.seed_contract
                assert cp['train_exposure_hash'] == s.train_context.exposure_hash()
                assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
                s.model.load_state_dict(cp['model_state'], strict=True)
                s.model.eval()
                seed_process(s.seed_contract['derived']['training_runtime'])
                table = collect_validation_table(s, epoch=row['selector']['best_epoch'])
            ok, delta = validation_agrees(table, row['validation_table'])
            info.update(passed=bool(ok), max_abs_difference=delta)
            del s, cp
        except Exception:
            info.update(passed=False, error=traceback.format_exc()[-4000:])
        torch.cuda.empty_cache()
        write(dest, info)
        print('REPLAY', key, info.get('passed'), info.get('max_abs_difference'), flush=True)


def test(key):
    """One authorized test pass (user, 2026-11-10: candidates without a test record whose validation rank
    is uncertain may be tested). Same procedure as scripts/test_all_final_20261001.py worker, inside the
    candidate's study context: validation replay must agree first; the test split is opened once."""
    import time
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
    row = read(OUT / 'candidates.json')['rows'][key]
    assert not row['test_records'], 'candidate already has a test record'
    name = key.replace('/', '__')
    dest, opened = OUT / 'new_tests' / f'{name}.json', OUT / 'new_tests' / 'opened' / f'{name}.json'
    assert not dest.exists() and not opened.exists(), 'test already opened for this candidate'
    assert sha(ROOT / row['checkpoint']) == row['checkpoint_sha256']
    cp = torch.load(ROOT / row['checkpoint'], map_location='cpu', weights_only=False)
    with contexts(STUDY_CONTEXT[row['study']]):
        from scripts.train_v4 import model_method_contract
        s = v4_runner.build_v4_session(ROOT / row['config'], device='cuda:0', run_seed=32, num_workers=0)
        assert model_method_contract(s) == cp['model_method_contract']
        assert cp['scaler'] == s.scaler and cp['seed_contract'] == s.seed_contract
        assert cp['static_asset_hash'] == s.train_context.static_asset_hash()
        assert cp['train_exposure_hash'] == s.train_context.exposure_hash()
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
        s.model.load_state_dict(cp['model_state'], strict=True)
        s.model.eval()
        seed = s.seed_contract['derived']['training_runtime']
        seed_process(seed)
        epoch = row['selector']['best_epoch']
        replay_table = collect_validation_table(s, epoch=epoch)
        ok, delta = validation_agrees(replay_table, row['validation_table'])
        write(OUT / 'new_tests' / 'validation' / f'{name}.json', dict(passed=ok, max_abs_difference=delta))
        assert ok, (key, delta)
        write(opened, dict(key=key, time=time.time(), pid=os.getpid(),
                           authority='2026-11-10 user: uncertain candidates may get a new test; '
                                     'final paper tables are test results (explicit exception to the test-use rule)'))
        seed_process(seed)
        ctx = build_runtime_context(row['dataset'], 'test', seed=32)
        assert ctx.static_asset_hash() == cp['static_asset_hash']
        dataset = V4LifecycleDataset(ctx, ROOT / s.args.data_path, input_len=s.args.input_len,
                                     pred_len=s.args.pred_len, slide_step=s.args.slide_step, freq=s.args.freq,
                                     seed=s.seed_contract['derived']['train_schedule'], epoch=0,
                                     scaler=dict(cp['scaler']), validation_seed=s.args.lcbt_validation_seed)
        loader = DataLoader(dataset, batch_size=s.train_loader.batch_size, shuffle=False, num_workers=0)
        collector = ValidationCellCollector(ctx, cp['scaler'], allow_phase='test')
        with torch.no_grad():
            for b in loader:
                b = v4_runner.to_device(b, s.device)
                pred, _ = unpack_model_output(s.model.forward_v4(b, ctx, epoch=epoch, teacher_forcing_mask=None))
                assert torch.isfinite(pred).all()
                collector.update(b, pred)
        table = collector.table()
        assert cell_count(table) == 33
        table.pop('development_validation_only', None)
        table.update(split='test', note='2026-11-10 audit: test of a validation-uncertain seed-32 candidate')
    write(dest, dict(passed=True, key=key, checkpoint_sha256=row['checkpoint_sha256'], dataset=row['dataset'],
                     model='ALFNet', seed=32, split='test', table=table, scaler=cp['scaler'],
                     test_exposure_hash=ctx.exposure_hash(), test_samples=len(dataset), evaluation_runtime_seed=seed,
                     validation_replay_max_abs_difference=delta, method_contract=cp['model_method_contract']))
    print('TEST', key, table['main']['LifeAvg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inventory', 'replay', 'test'])
    parser.add_argument('--key')
    parser.add_argument('--keys', nargs='*')
    parser.add_argument('--shard', default='0/1')
    a = parser.parse_args()
    if a.mode == 'inventory':
        inventory()
    elif a.mode == 'test':
        test(a.key)
    else:
        keys = a.keys or sorted(read(OUT / 'candidates.json')['rows'])
        i, n = map(int, a.shard.split('/'))
        replay([k for j, k in enumerate(keys) if j % n == i])
