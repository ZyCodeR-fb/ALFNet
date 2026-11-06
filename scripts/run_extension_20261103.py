"""2026-11-03 extension study: four comparison methods and two whole-mechanism ablations.

Modes: freeze | check | preflight | queue | worker --key K | collect.
Training/validation only; the one authorized test pass per final checkpoint is a
separate, later-frozen plan (scripts/test_extension_20261103.py).

Comparison runs: {Merlin, HDTTS, STONE, STOP} x {METRLA, PEMSBAY, PEMS04, PEMS08}.
Ablations at the current final anchors (PEMS08 = WD 1e-3): w/o OS on all four,
w/o SAHC on PEMS04/PEMS08; METRLA/PEMSBAY w/o SAHC reuse the 2026-10-06
no_imputer runs (identical definition there: no interpolation, no centering).
"""
import argparse
import copy
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
from scripts.run_metrla_hparams import digest, identity, parameter_identity, read, sha, write  # noqa: E402

NAME = 'extension_20261103'
DIR = ROOT / 'configs/paper' / NAME
PLAN = Path(os.environ.get('ALFNET_EXT_PLAN', DIR / 'plan.json'))
PYDEPS = ROOT / 'outputs_v4/extension_20261103/pydeps'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
BASELINES = ('Merlin', 'HDTTS', 'STONE', 'STOP')
ANCHOR_MANIFEST = ROOT / 'paper/pems08_wd_test_20261012/selected_configuration_manifest.json'
REUSE_PLAN = ROOT / 'configs/paper/final_ablations_20261006/plan.json'
REUSE_VALIDATION = ROOT / 'paper/ablations_complete_20261010/results_and_audit.json'
REUSE_TEST = ROOT / 'paper/all_ablations_test_20261010/results_and_audit.json'
LABELS = {'no_sahc': 'w/o SAHC', 'no_os': 'w/o OS'}
MAX_CONCURRENT = 3
DATA = {'METRLA': ('METR-LA', 'v'), 'PEMSBAY': ('PEMS-BAY', 'v'), 'PEMS04': ('PEMS04', 'q'),
        'PEMS08': ('PEMS08', 'q')}
AUTHORITY = ('2026-11-03 user (server prompt SERVER_EXPERIMENT_PROMPT_CN.md): adapt Merlin, HD-TTS, '
             'STONE, STOP and run w/o SAHC and w/o OS; max 3 concurrent processes; one test per final '
             'checkpoint after validation selection is frozen (separate test plan).')


def output(p):
    # ALFNET_EXT_OUTPUT is only for a draft (pre-freeze) dry run outside artifacts/.
    base = Path(os.environ['ALFNET_EXT_OUTPUT']) if 'ALFNET_EXT_OUTPUT' in os.environ else ROOT / 'artifacts/experiments' / NAME
    return base / p['plan_sha256']


def baseline_config(dataset, model):
    from experiments.extension_baselines_20261103 import RECIPES
    folder, target = DATA[dataset]
    recipe = RECIPES[model]
    return {
        'runtime': dict(model=model, model_variant=f'{NAME}_{dataset}_{model}',
                        model_recipe='released_source_v4_adapter', mode='train', t_mark=True),
        'dataset': dict(dloader_name=dataset, data_path=f'./data_v4/{folder}/raw_data.pkl',
                        unknown_nodes_path=f'./data_v4/{folder}/unknown_nodes.npy', input_len=12,
                        pred_len=12, look_back=0, slide_step=1, input_features=target,
                        target_features=target, freq='min'),
        'training': dict(device='cuda:0', seed=32, max_epoch=100, patience=10, stop_based='val_lifecycle',
                         batch_size=recipe['batch_size'], num_workers=0, loss='mae',
                         optimizer=recipe['optimizer'].lower(), scheduler='None', scheduler_patience=1,
                         lr=recipe['lr'], weight_decay=recipe['weight_decay'], d_model=64, e_layers=2,
                         d_layers=1, k_hop=1, early_stopping_start_epoch=0),
        'lifecycle': dict(train_for_async_lifecycle=False, lifecycle_role_features=False,
                          lifecycle_history_age_feature=False, lifecycle_observation_mask_feature=False,
                          lifecycle_age_mode='state_duration', lifecycle_training_strategy='lcbt_v4',
                          lcbt_validation_seed=12026, lcbt_train_end_ratio=0.6, lcbt_val_end_ratio=0.8,
                          lcbt_test_start_ratio=0.8, lcbt_scaler_end_ratio=0.6, lcbt_zero_history_ratio=0.25,
                          lcbt_steady_ratio=0.25, lcbt_join_ratio=0.25, lcbt_fault_ratio=0.25),
        # Documentation only: the study launcher installs exactly this recipe.
        'family': {'model': {}, 'recipe': {'released_training_recipe': json.loads(json.dumps(recipe))}},
    }


def canonical(c):
    c = copy.deepcopy(c)
    c['runtime'].pop('model_variant', None)
    for e in ('graph', 'patch'):
        c['family']['experts'][e]['config_path'] = yaml.safe_load(
            (ROOT / c['family']['experts'][e]['config_path']).read_text())
    return c


def ablation_config(dataset, kind, anchor_path, key):
    """Anchor + intervention key; nested graph config only for the static router."""
    c = yaml.safe_load((ROOT / anchor_path).read_text())
    expected = canonical(c)
    c['runtime']['model_variant'] = f'{NAME}_{key}'
    c['family']['recipe']['study_ablation'] = kind
    expected['family']['recipe']['study_ablation'] = kind
    if kind == 'no_os':
        nested = yaml.safe_load((ROOT / c['family']['experts']['graph']['config_path']).read_text())
        nested['family']['recipe']['moge_router_mode'] = 'static'
        path = DIR / f'{key}_graph.yaml'
        path.write_text(yaml.safe_dump(nested, sort_keys=False))
        c['family']['experts']['graph']['config_path'] = str(path.relative_to(ROOT))
        expected['family']['experts']['graph']['config_path']['family']['recipe']['moge_router_mode'] = 'static'
    if kind == 'no_sahc':
        recipe = c['family']['recipe']
        for c_ in (recipe, expected['family']['recipe']):
            c_.pop('unified_variant', None)
            if dataset in ('PEMS04', 'PEMS08'):
                # Keep only the anchor's decoder residual scale 1; no bridge, no centering.
                c_['support_transfer_variant'] = 'delta1'
            else:
                assert 'support_transfer_variant' not in c_
    path = DIR / f'{key}.yaml'
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    assert canonical(yaml.safe_load(path.read_text())) == expected, key
    return path


def freeze():
    import torch
    assert not PLAN.exists(), 'plan already frozen; a new identity needs a new name'
    DIR.mkdir(parents=True, exist_ok=True)
    anchors = read(ANCHOR_MANIFEST)['entries']
    files = [ANCHOR_MANIFEST, REUSE_PLAN, REUSE_VALIDATION, REUSE_TEST, Path(__file__)]
    configs, anchor_rows, reuse = {}, {}, {}
    for ds in DATASETS:
        a = anchors[ds]
        report_path = ROOT / Path(a['checkpoint']).with_name('run_report.json')
        report = read(report_path)
        assert sha(ROOT / a['checkpoint']) == a['checkpoint_sha256'] and sha(ROOT / a['config']) == a['config_sha256']
        assert report['config_sha256'] == a['config_sha256'] and report['best_epoch'] is not None
        anchor_rows[ds] = dict(key=a['key'], config=a['config'], checkpoint=a['checkpoint'],
                               checkpoint_sha256=a['checkpoint_sha256'], report=str(report_path.relative_to(ROOT)),
                               table=report['development_validation_table'])
        anchor_cfg = yaml.safe_load((ROOT / a['config']).read_text())
        files += [ROOT / a['config'], ROOT / a['checkpoint'], report_path]
        files += [ROOT / anchor_cfg['family']['experts'][e]['config_path'] for e in ('graph', 'patch')]
        folder, _ = DATA[ds]
        files += [ROOT / f'data_v4/{folder}/raw_data.pkl', ROOT / f'data_v4/{folder}/unknown_nodes.npy',
                  ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json']
    index = 0
    for model in BASELINES:
        for ds in DATASETS:
            key = f'{index:02d}_{ds}_{model}'
            path = DIR / f'{key}.yaml'
            path.write_text(yaml.safe_dump(baseline_config(ds, model), sort_keys=False))
            configs[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), dataset=ds, model=model,
                                kind='baseline', seed=32, label=model)
            files.append(path)
            index += 1
    for kind, scope in (('no_os', DATASETS), ('no_sahc', ('PEMS04', 'PEMS08'))):
        for ds in scope:
            key = f'{index:02d}_{ds}_{kind}'
            path = ablation_config(ds, kind, anchor_rows[ds]['config'], key)
            configs[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), dataset=ds, model='ALFNet',
                                kind='ablation', ablation=kind, seed=32, label=LABELS[kind])
            files.append(path)
            if kind == 'no_os':
                files.append(DIR / f'{key}_graph.yaml')
            index += 1
    # w/o SAHC on the speed datasets: the existing no_imputer run is the same intervention.
    validation_rows = read(REUSE_VALIDATION)['rows']
    test_rows = read(REUSE_TEST)['rows']
    replays = read(REUSE_TEST)['validation_replays']
    for ds, old in (('METRLA', '12_METRLA_no_imputer'), ('PEMSBAY', '13_PEMSBAY_no_imputer')):
        row = validation_rows[old]
        cfg = yaml.safe_load((ROOT / row['config']).read_text())
        anchor_cfg = yaml.safe_load((ROOT / anchor_rows[ds]['config']).read_text())
        expected = canonical(anchor_cfg)
        expected['family']['recipe']['study_ablation'] = 'no_imputer'
        assert canonical(cfg) == expected, old
        assert 'support_transfer_variant' not in cfg['family']['recipe'] and 'unified_variant' not in cfg['family']['recipe']
        report = read(ROOT / row['report'])
        assert report['development_validation_table'] == row['table']
        assert sha(ROOT / row['checkpoint']) == test_rows[old]['checkpoint_sha256']
        assert replays[old]['passed'] and replays[old]['max_abs_difference'] == 0.0
        reuse[f'{ds}_no_sahc'] = dict(
            dataset=ds, ablation='no_sahc', label=LABELS['no_sahc'], source_key=old, config=row['config'],
            report=row['report'], checkpoint=row['checkpoint'], checkpoint_sha256=test_rows[old]['checkpoint_sha256'],
            validation_table=row['table'], test_table=test_rows[old]['table'],
            basis=('final anchor has no window interpolation and no centering (no support_transfer/unified '
                   'variant, decoder scale .3), so w/o SAHC equals no_imputer: observed-only fill, S retained, '
                   'objective unchanged; config semantics equal current anchor + no_imputer; checkpoint hash '
                   'equals the 2026-10-10 test record; validation replay before that test exact (0.0); ALFNet, '
                   'GRIN, expert, data and evaluation code unchanged since its 2026-10-06 plan (v4_runner diff '
                   'only removes retired model registrations and the GWNET clip branch).'))
        files += [ROOT / row['config'], ROOT / row['report'], ROOT / row['checkpoint']]
    # The later test/report script is not part of training and gets its own frozen plan.
    later = {ROOT / 'scripts/test_extension_20261103.py'}
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider', 'scripts', 'tests'):
        files.extend(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts and p not in later)
    files.extend(p for p in (ROOT / 'model/comparison_model').rglob('SOURCE_MANIFEST.json'))
    files.append(ROOT / 'args.py')
    for name in ('Merlin', 'HDTTS', 'STONE', 'STOP'):
        for entry in read(ROOT / f'model/comparison_model/{name}/SOURCE_MANIFEST.json')['source_files']:
            files.append(ROOT / entry['reference'])
    deps = {str(p.relative_to(PYDEPS)): sha(p) for p in sorted(PYDEPS.glob('*.dist-info/RECORD'))}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = dict(kind=NAME, status='authorized_training_validation', authority=AUTHORITY, seed=32,
             max_concurrent=MAX_CONCURRENT, test_authorized_after_freeze=True,
             test_policy='one test per final validation-selected checkpoint under a separately frozen test plan; '
                         'test never used for selection, tuning, method change or rerun decisions',
             selection='validation 33-cell main/LifeAvg/MAE, strict improvement, earlier epoch on ties',
             budget='max_epoch 100, patience 10 (per stage for Merlin)',
             recovery_policy='no automatic retry; OOM or divergence is recorded first and any recovery is '
                             'written into a new plan before its results are seen, applied to every dataset '
                             'of that method',
             configs=configs, anchors=anchor_rows, reuse=reuse, pydeps=dict(path=str(PYDEPS.relative_to(ROOT)),
                                                                           records=deps),
             git_head=git, torch=torch.__version__,
             files={str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))})
    p['plan_sha256'] = digest(p)
    write(PLAN, p)
    print(p['plan_sha256'], len(configs), 'runs', len(reuse), 'reused')


def check():
    p = read(PLAN)
    assert digest(p) == p['plan_sha256']
    for f, h in p['files'].items():
        assert sha(ROOT / f) == h, f
    for f, h in p['pydeps']['records'].items():
        assert sha(PYDEPS / f) == h, f
    return p


@contextmanager
def run_context():
    """Extension baselines + flow input adapters + study interventions."""
    from experiments import v4_runner
    from experiments.alfnet_extension_ablation_adapter import apply_intervention
    from experiments.extension_baselines_20261103 import extension_context
    from experiments.flow_support_transfer import adapter as support
    from experiments.flow_unified import adapter as unified
    from scripts import train_v4
    with ExitStack() as stack:
        stack.enter_context(extension_context())
        stack.enter_context(support())
        stack.enter_context(unified())
        build, contract = v4_runner.build_v4_session, train_v4.model_method_contract

        def wrapped(*args, **kwargs):
            return apply_intervention(build(*args, **kwargs))

        def describe(s):
            r = contract(s)
            r['paired_ablation'] = getattr(s.model, 'study_ablation', 'reference')
            return r
        for obj, name, value in [(v4_runner, 'build_v4_session', wrapped), (train_v4, 'build_v4_session', wrapped),
                                 (train_v4, 'model_method_contract', describe)]:
            stack.enter_context(patch.object(obj, name, value))
        yield


def safe_train(cfg, root, plan, reference, device='cuda:0', smoke=False, resume=False,
               max_train_batches=None, max_validation_batches=None):
    """Common train() with no skipped batch and evaluation counts equal to the anchor's."""
    from scripts import train_v4 as training
    original_train, original_validate = training.train_one_epoch, training.validate

    def epoch(*args, **kwargs):
        r = original_train(*args, **kwargs)
        assert r['skipped'] == 0, 'training skipped an invalid batch'
        return r

    def validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        r = result[0] if isinstance(result, tuple) else result
        assert r['skipped'] == 0, 'validation skipped an invalid batch'
        if isinstance(result, tuple):
            for g in ('main', 'phase'):
                for c, values in reference[g].items():
                    for field, v in values.items():
                        if field.endswith('_count'):
                            assert result[1][g][c][field] == v, (g, c, field)
        return result
    with patch.object(training, 'train_one_epoch', epoch), patch.object(training, 'validate', validate):
        return training.train(cfg, root, device=device, seed=32, num_workers=0, plan=plan, save=not smoke,
                              resume=resume, collect_development_table=not smoke,
                              max_epochs=1 if smoke else None, max_train_batches=max_train_batches,
                              max_validation_batches=max_validation_batches)


def train_run(p, key, root, device='cuda:0', smoke=False, max_train_batches=None, max_validation_batches=None):
    row = p['configs'][key]
    reference = p['anchors'][row['dataset']]['table']
    cfg = ROOT / row['path']
    kwargs = dict(device=device, smoke=smoke, max_train_batches=max_train_batches,
                  max_validation_batches=max_validation_batches)
    if row['model'] == 'Merlin':
        from experiments.extension_baselines_20261103 import train_merlin

        def fn(config, output_root, device, plan, resume, smoke, max_train_batches, max_validation_batches):
            return safe_train(config, output_root, plan, reference, device=device, smoke=smoke, resume=resume,
                              max_train_batches=max_train_batches, max_validation_batches=max_validation_batches)
        return train_merlin(fn, cfg, root, p, **kwargs)
    return safe_train(cfg, root, p, reference, **kwargs)


# --- preflight -------------------------------------------------------------------


def leakage_audit(session, batches=3):
    """Held-out V/T dynamic signals and labels never reach a training batch."""
    import numpy as np
    import torch
    from data_provider.v4_lifecycle_loader import V4LifecycleDataset
    from lib.v4_payload import load_v4_payload
    from model.comparison_model.Merlin.adapter import teacher_history_for_batch
    ds, ctx = session.train_dataset, session.train_context
    known = np.asarray(ctx.known_nodes, dtype=np.int64)
    held = np.setdiff1d(np.arange(ctx.num_nodes), known)
    indices = list(range(0, len(ds), max(1, len(ds) // batches)))[:batches]
    before = [ds[i] for i in indices]
    saved = ds.normalised.copy(), ds.natural.copy()
    try:
        ds.normalised[:, held] = 1e6
        ds.natural[:, held] = True
        after = [ds[i] for i in indices]
        collated = torch.utils.data.default_collate(after)
        history_after = teacher_history_for_batch(ds, collated)
    finally:
        ds.normalised[:], ds.natural[:] = saved
    history_before = teacher_history_for_batch(ds, torch.utils.data.default_collate(before))
    fields = ('base_visible_input', 'input_available_mask', 'target', 'target_available_mask',
              'decoder_available_mask', 'loss_mask', 'encoder_time_marks')
    invariant = all(torch.equal(a[f], b[f]) for a, b in zip(before, after) for f in fields)
    zero_outside_k = all(
        float(b['base_visible_input'][:, held].abs().sum()) == 0 and float(b['target'][:, held].abs().sum()) == 0
        and not bool(b['loss_mask'][:, held].any()) and not bool(b['input_available_mask'][:, held].any())
        for b in before)
    payload = load_v4_payload(ROOT / session.args.data_path)
    raw = np.asarray(payload['data'], dtype=np.float32)[:, :, ctx.target_channel].astype(np.float64)
    natural = np.isfinite(raw) & (raw > 0)
    signal = np.where(natural, raw, np.nan)
    train_end = ds.boundaries['train'][1]
    poisoned = signal.copy()
    poisoned[:, held] = poisoned[:, held] * 1000. + 7.
    scaler_clean = V4LifecycleDataset._fit_scaler(signal, natural, train_end, ctx.known_nodes)
    scaler_poisoned = V4LifecycleDataset._fit_scaler(poisoned, natural, train_end, ctx.known_nodes)
    signal[train_end:] = 5e5
    scaler_future = V4LifecycleDataset._fit_scaler(signal, natural, train_end, ctx.known_nodes)
    result = dict(samples=indices, batch_fields_invariant_to_heldout_values=invariant,
                  heldout_zero_in_inputs_targets_and_masks=zero_outside_k,
                  merlin_teacher_history_invariant=bool(torch.equal(history_before, history_after)),
                  merlin_teacher_history_zero_outside_k=float(history_before[:, :, held].abs().sum()) == 0,
                  scaler_equals_session=dict(scaler_clean) == dict(session.scaler),
                  scaler_invariant_to_heldout_values=dict(scaler_clean) == dict(scaler_poisoned),
                  scaler_invariant_to_future_time=dict(scaler_clean) == dict(scaler_future),
                  known_nodes=int(known.size), heldout_nodes=int(held.size))
    assert all(v for k, v in result.items() if isinstance(v, bool)), result
    return result


def os_ablation_audit(session):
    """w/o OS: no learnable path sees role/progress/mask; S itself cannot move the output."""
    import torch
    from experiments.v4_runner import to_device
    model = session.model
    batch = next(iter(session.validation_loader))
    batch = to_device(batch, session.device)
    captured = {}
    raw = model.raw_enc_in
    bigril = model.shared_input_imputer.bigril
    hidden = bigril.fwd_rnn.hidden_size
    hooks = [
        bigril.fwd_rnn.spatial_decoder.register_forward_hook(
            lambda m, a, k, o: captured.setdefault('decoder_m', []).append(k['m']), with_kwargs=True),
        bigril.fwd_rnn.cells[0].register_forward_pre_hook(
            lambda m, a: captured.setdefault('cell_os', []).append(float(a[0][:, raw:].abs().sum()))),
        bigril.out.register_forward_pre_hook(
            lambda m, a: captured.setdefault('merge_mask', []).append(float(a[0][:, 4 * hidden:].abs().sum()))),
    ]
    original_patch = model.patch_expert.forward_v4

    def spy(x, marks):
        captured['patch_os'] = x[..., raw:].detach().abs().sum().item()
        return original_patch(x, marks)
    model.patch_expert.forward_v4 = spy
    original_graph = model.graph_expert.forward

    def graph_spy(adjacency, x_enc, *args, **kwargs):
        captured['graph_os'] = x_enc[..., raw:].detach().abs().sum().item()
        return original_graph(adjacency, x_enc, *args, **kwargs)
    model.graph_expert.forward = graph_spy
    model.eval()
    try:
        with torch.no_grad():
            reference = model.forward_v4(batch, session.validation_context)['pred']
            changed = dict(batch)
            context = batch['lifecycle_context'].clone()
            generator = torch.Generator().manual_seed(7)
            context[..., :5] = torch.rand(context[..., :5].shape, generator=generator).to(context)
            changed['lifecycle_context'] = context
            perturbed = model.forward_v4(changed, session.validation_context)['pred']
    finally:
        for h in hooks:
            h.remove()
        del model.patch_expert.forward_v4
        del model.graph_expert.forward
    result = dict(output_invariant_to_role_and_progress=bool(torch.equal(reference, perturbed)),
                  grin_decoder_mask_input_zero=all(float(m.abs().sum()) == 0 for m in captured['decoder_m']),
                  grin_cell_mask_and_state_inputs_zero=all(v == 0 for v in captured['cell_os']),
                  grin_merge_mask_input_zero=all(v == 0 for v in captured['merge_mask']),
                  temporal_os_input_zero=captured['patch_os'] == 0, graph_os_input_zero=captured['graph_os'] == 0,
                  router_mode=model.graph_expert.router_mode)
    assert all(v for k, v in result.items() if isinstance(v, bool)), result
    return result


def preflight(keys=None):
    import torch
    from experiments import v4_runner
    from scripts import train_v4  # patched attributes are read at call time
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    # One file per run, so preflight processes for disjoint key sets never race.
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')} if (root / 'preflight').exists() else {}
    with run_context():
        anchor_ids = {}
        for ds, a in p['anchors'].items():
            s = v4_runner.build_v4_session(ROOT / a['config'], device='cpu', run_seed=32, num_workers=0)
            anchor_ids[ds] = dict(identity=identity(s), parameters={n: t.detach().clone()
                                                                     for n, t in s.model.named_parameters()})
            cp = torch.load(ROOT / a['checkpoint'], map_location='cpu', weights_only=False)
            assert all(cp[k] == v for k, v in anchor_ids[ds]['identity'].items()), ds
            del s, cp
        for key, row in sorted(p['configs'].items()):
            if (keys and key not in keys) or reports.get(key, {}).get('passed'):
                continue
            print('PREFLIGHT_START', key, flush=True)
            info = dict(key=key)
            s = v4_runner.build_v4_session(ROOT / row['path'], device='cpu', run_seed=32, num_workers=0)
            ids = identity(s)
            assert ids == anchor_ids[row['dataset']]['identity'], key
            info['identity'] = 'equal to the dataset anchor (scaler, seed contract, exposures, static assets)'
            info['parameter_count'] = sum(t.numel() for t in s.model.parameters())
            info['trainable_parameter_count'] = sum(t.numel() for t in s.model.parameters() if t.requires_grad)
            if row['kind'] == 'ablation':
                initial = anchor_ids[row['dataset']]['parameters']
                for n, t in s.model.named_parameters():
                    if n in initial:
                        assert torch.equal(t, initial[n]), (key, n)
                    else:
                        assert row['ablation'] == 'no_os' and n.endswith('w_static') and torch.count_nonzero(t) == 0, n
                info['paired_initialization'] = 'all retained parameters equal the anchor initial values'
                assert (s.args.max_epoch, s.args.patience, s.args.early_stopping_start_epoch) == (100, 10, 40)
                opt = train_v4.build_optimizer(s)
                anchor_training = yaml.safe_load((ROOT / p['anchors'][row['dataset']]['config']).read_text())['training']
                assert all(g['weight_decay'] == float(anchor_training['weight_decay']) for g in opt.param_groups)
                info['optimizer'] = dict(cls=type(opt).__name__, learning_rates=[g['lr'] for g in opt.param_groups],
                                         weight_decay=float(anchor_training['weight_decay']))
                if row['ablation'] == 'no_os':
                    info['os_audit'] = os_ablation_audit(s)
                else:
                    from experiments.alfnet_ablation_adapter import ObservedOnly
                    assert isinstance(s.model.shared_input_imputer, ObservedOnly)
                    info['sahc'] = dict(imputer='ObservedOnly',
                                        support_transfer_variant=getattr(s.model, 'support_transfer_variant', None),
                                        decoder_scale=float(s.model.graph_expert.decoder.scale))
            else:
                assert (s.args.max_epoch, s.args.patience, s.args.early_stopping_start_epoch) == (100, 10, 0)
                info['leakage'] = leakage_audit(s)
                if row['model'] != 'Merlin':
                    from experiments.extension_baselines_20261103 import RECIPES
                    opt = train_v4.build_optimizer(s)
                    sched = train_v4.build_scheduler(s, opt)
                    recipe = RECIPES[row['model']]
                    assert type(opt).__name__ == 'Clipped' + recipe['optimizer']
                    assert type(sched).__name__ == recipe['scheduler'][0]
                    info['optimizer'] = dict(cls=type(opt).__name__, scheduler=type(sched).__name__, groups=[
                        {k: v for k, v in g.items() if k in ('lr', 'weight_decay', 'eps', 'betas')}
                        for g in opt.param_groups])
            del s
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            report = train_run(p, key, None, device='cuda:0', smoke=True, max_train_batches=20,
                               max_validation_batches=4)
            if row['model'] == 'Merlin':
                from experiments.extension_baselines_20261103 import merlin_student_smoke
                s = v4_runner.build_v4_session(ROOT / row['path'], device='cuda:0', run_seed=32, num_workers=0)
                info['merlin_student_smoke'] = merlin_student_smoke(s)
                del s
            info.update(passed=True, smoke_epochs=report['epochs_run'],
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                        seconds=time.time() - start)
            torch.cuda.empty_cache()
            reports[key] = info
            write(root / 'preflight' / f'{key}.json', info)
            print('PREFLIGHT_OK', key, f"{info['peak_reserved_gib']:.2f}GiB {info['seconds']:.0f}s", flush=True)
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')}
    if all(reports.get(k, {}).get('passed') for k in p['configs']):
        write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=True, plan_sha256=p['plan_sha256'],
                                                    reports=[reports[k] for k in sorted(p['configs'])]))


# --- queue -----------------------------------------------------------------------


def gpu_free_gib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
                                   '-i', '0'], text=True)
    return float(out.strip()) / 1024


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
    # ALFNet ablations are the longest; start them first, then baselines by model.
    order = sorted(p['configs'], key=lambda k: (p['configs'][k]['kind'] != 'ablation', k))
    jobs = [dict(key=k, status='queued') for k in order]
    active, last = {}, 0.
    base_env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
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
            free = gpu_free_gib()
            if free > need:
                key = pending['key']
                env = dict(base_env)
                if p['configs'][key]['model'] == 'HDTTS':
                    env['PYTHONPATH'] = str(PYDEPS)
                log = (root / f'{key}.log').open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need, log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_extension_20261103.py worker --key {key}')
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
        if e['kind'] == 'ablation':
            assert r['model_method_contract']['paired_ablation'] == e['ablation']
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'],
                         teacher_report=r.get('teacher_report'), teacher_checkpoint_sha256=r.get('teacher_checkpoint_sha256'))
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                                                 anchors=p['anchors'], reuse=p['reuse'], test_accessed=False))
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
