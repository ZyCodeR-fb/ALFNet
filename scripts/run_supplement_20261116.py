"""2026-11-16 supplement: OS decomposition and the partition algorithm.

Modes: freeze | check | preflight | queue | worker --key K | collect.
Training/validation only; the authorized test pass is scripts/test_supplement_20261116.py.

Runs (seed 32, current final anchors, paired initialization):
  w/o Observation Role  x {METRLA, PEMSBAY, PEMS04, PEMS08}
  w/o Observation Mask  x {METRLA, PEMSBAY, PEMS04, PEMS08}
  BFS partition         x {METRLA, PEMSBAY}   (the two datasets partitioned by the KD-tree)
"""
import argparse
import fcntl
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
from scripts import run_extension_20261103 as ext  # noqa: E402

NAME = 'supplement_20261116'
DIR = ROOT / 'configs/paper' / NAME
PLAN = DIR / 'plan.json'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
LABELS = {'no_role': 'w/o Observation Role', 'no_mask': 'w/o Observation Mask', 'bfs': 'BFS partition'}
SCOPE = (('no_role', DATASETS), ('no_mask', DATASETS), ('bfs', ('METRLA', 'PEMSBAY')))
MAX_CONCURRENT = 5
AUTHORITY = ('2026-11-16 user: run w/o Observation Role and w/o Observation Mask on four datasets and the BFS '
             'partition on the speed datasets, seed 32 only; test evaluation authorized for these runs; '
             'max 5 concurrent processes. Test never used for selection, tuning or rerun decisions.')


def output(p):
    return ROOT / 'artifacts/experiments' / NAME / p['plan_sha256']


def run_config(kind, anchor_path, key):
    c = yaml.safe_load((ROOT / anchor_path).read_text())
    expected = ext.canonical(c)
    c['runtime']['model_variant'] = f'{NAME}_{key}'
    name, value = ('study_layout', 'graph_native_component_bfs') if kind == 'bfs' else ('study_ablation', kind)
    c['family']['recipe'][name] = value
    expected['family']['recipe'][name] = value
    path = DIR / f'{key}.yaml'
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    assert ext.canonical(yaml.safe_load(path.read_text())) == expected, key
    return path


def freeze():
    import torch
    assert not PLAN.exists(), 'plan already frozen; a new identity needs a new name'
    DIR.mkdir(parents=True, exist_ok=True)
    anchors = read(ext.ANCHOR_MANIFEST)['entries']
    files = [ext.ANCHOR_MANIFEST, Path(__file__)]
    configs, anchor_rows = {}, {}
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
        folder, _ = ext.DATA[ds]
        files += [ROOT / f'data_v4/{folder}/raw_data.pkl', ROOT / f'data_v4/{folder}/unknown_nodes.npy',
                  ROOT / f'configs/paper/v4_data_manifests/{ds}_run_manifest.json']
    index = 0
    for kind, scope in SCOPE:
        for ds in scope:
            key = f'{index:02d}_{ds}_{kind}'
            path = run_config(kind, anchor_rows[ds]['config'], key)
            configs[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), dataset=ds, model='ALFNet',
                                kind='layout' if kind == 'bfs' else 'ablation', ablation=kind, seed=32,
                                label=LABELS[kind])
            files.append(path)
            index += 1
    for folder in ('model', 'layers', 'lib', 'experiments', 'utils', 'data_provider'):
        files.extend(p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts)
    files += [ROOT / 'args.py'] + [ROOT / 'scripts' / n for n in (
        'train_v4.py', 'run_extension_20261103.py', 'run_metrla_hparams.py', 'eval_release.py')]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    p = dict(kind=NAME, status='authorized_training_validation', authority=AUTHORITY, seed=32,
             max_concurrent=MAX_CONCURRENT, test_authorized_after_freeze=True,
             selection='validation 33-cell main/LifeAvg/MAE, strict improvement, earlier epoch on ties',
             budget='max_epoch 100, patience 10, early stopping from epoch 40 (the anchors\' budget)',
             recovery_policy='no automatic retry; a failure is recorded first',
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
def run_context(row=None):
    """Flow input adapters + study interventions (+ the BFS partition for a layout run)."""
    from experiments import v4_runner
    from experiments.alfnet_supplement_20261116_adapter import apply_intervention, bfs_layout
    from experiments.flow_support_transfer import adapter as support
    from experiments.flow_unified import adapter as unified
    from scripts import train_v4
    with ExitStack() as stack:
        stack.enter_context(support())
        stack.enter_context(unified())
        if row is not None and row['kind'] == 'layout':
            stack.enter_context(bfs_layout())
        build, contract = v4_runner.build_v4_session, train_v4.model_method_contract

        def wrapped(*args, **kwargs):
            return apply_intervention(build(*args, **kwargs))

        def describe(s):
            r = contract(s)
            r['paired_ablation'] = getattr(s.model, 'study_ablation', 'reference')
            r['layout_algorithm'] = s.train_context.layout_algorithm
            return r
        for obj, name, value in [(v4_runner, 'build_v4_session', wrapped), (train_v4, 'build_v4_session', wrapped),
                                 (train_v4, 'model_method_contract', describe)]:
            stack.enter_context(patch.object(obj, name, value))
        yield


def train_run(p, key, root, **kwargs):
    row = p['configs'][key]
    return ext.safe_train(ROOT / row['path'], root, p, p['anchors'][row['dataset']]['table'], **kwargs)


# --- audits ------------------------------------------------------------------------


def _forward(session, batch):
    return session.model.forward_v4(batch, session.validation_context)['pred']


def role_audit(session):
    """w/o Observation Role: the prediction cannot depend on the role channels."""
    import torch
    from experiments.v4_runner import to_device
    batch = to_device(next(iter(session.validation_loader)), session.device)
    session.model.eval()
    with torch.no_grad():
        reference = _forward(session, batch)
        changed = dict(batch)
        context = batch['lifecycle_context'].clone()
        g = torch.Generator().manual_seed(7)
        context[..., :4] = torch.rand(context[..., :4].shape, generator=g).to(context)
        changed['lifecycle_context'] = context
        by_role = _forward(session, changed)
        context = batch['lifecycle_context'].clone()
        context[..., 4] = torch.rand(context[..., 4].shape, generator=g).to(context)
        changed['lifecycle_context'] = context
        by_progress = _forward(session, changed)
    result = dict(output_invariant_to_role=bool(torch.equal(reference, by_role)),
                  output_depends_on_progress=not bool(torch.equal(reference, by_progress)))
    assert all(result.values()), result
    return result


def mask_audit(session):
    """w/o Observation Mask: no learned path sees the mask; role and progress still arrive."""
    import torch
    from experiments.v4_runner import to_device
    model = session.model
    batch = to_device(next(iter(session.validation_loader)), session.device)
    raw, mask_index = model.raw_enc_in, model.mask_index
    bigril = model.shared_input_imputer.bigril
    hidden = bigril.fwd_rnn.hidden_size
    seen = {}
    hooks = [
        bigril.fwd_rnn.spatial_decoder.register_forward_hook(
            lambda m, a, k, o: (seen.setdefault('decoder_m', []).append(float(k['m'].abs().sum())),
                                seen.setdefault('decoder_u', []).append(float(k['u'].abs().sum())))[0],
            with_kwargs=True),
        bigril.fwd_rnn.cells[0].register_forward_pre_hook(
            lambda m, a: (seen.setdefault('cell_m', []).append(float(a[0][:, raw:2 * raw].abs().sum())),
                          seen.setdefault('cell_u', []).append(float(a[0][:, 2 * raw:].abs().sum())))[0]),
        bigril.out.register_forward_pre_hook(
            lambda m, a: seen.setdefault('merge_m', []).append(float(a[0][:, 4 * hidden:].abs().sum()))),
    ]
    original_patch, original_graph = model.patch_expert.forward_v4, model.graph_expert.forward

    def spy(x, marks):
        seen['patch_mask'] = float(x[..., mask_index].abs().sum())
        seen['patch_role'] = float(x[..., raw:mask_index].abs().sum())
        return original_patch(x, marks)

    def graph_spy(adjacency, x_enc, *args, **kwargs):
        seen['graph_mask'] = float(x_enc[..., mask_index].abs().sum())
        seen['graph_role'] = float(x_enc[..., raw:mask_index].abs().sum())
        return original_graph(adjacency, x_enc, *args, **kwargs)
    model.patch_expert.forward_v4, model.graph_expert.forward = spy, graph_spy
    model.eval()
    try:
        with torch.no_grad():
            reference = _forward(session, batch)
            changed = dict(batch)
            context = batch['lifecycle_context'].clone()
            context[..., :4] = torch.rand(context[..., :4].shape, generator=torch.Generator().manual_seed(7)).to(context)
            changed['lifecycle_context'] = context
            by_role = _forward(session, changed)
    finally:
        for h in hooks:
            h.remove()
        del model.patch_expert.forward_v4
        del model.graph_expert.forward
    result = dict(grin_decoder_mask_zero=all(v == 0 for v in seen['decoder_m']),
                  grin_cell_mask_zero=all(v == 0 for v in seen['cell_m']),
                  grin_merge_mask_zero=all(v == 0 for v in seen['merge_m']),
                  grin_role_progress_kept=all(v > 0 for v in seen['decoder_u'] + seen['cell_u']),
                  temporal_mask_zero=seen['patch_mask'] == 0, graph_mask_zero=seen['graph_mask'] == 0,
                  temporal_role_progress_kept=seen['patch_role'] > 0, graph_role_progress_kept=seen['graph_role'] > 0,
                  output_depends_on_role=not bool(torch.equal(reference, by_role)))
    assert all(result.values()), result
    return result


def layout_audit(session, anchor_session):
    import numpy as np
    from lib.graph_native_layout import layout_parts
    ctx, old = session.train_context, anchor_session.train_context
    parts = [list(part) for part in ctx.layout_parts]
    reference = [[int(n) for n in part] for part in layout_parts(ctx.adjacency.cpu().numpy(), 5)]
    moved = sum(1 for a, b in zip(parts, old.layout_parts) for n in a if n not in set(b))
    adjacency = ctx.adjacency.cpu().numpy()
    view = ((adjacency + adjacency.T) > 0)
    np.fill_diagonal(view, False)

    def inside(layout):
        index = np.empty(ctx.num_nodes, dtype=int)
        for i, part in enumerate(layout):
            index[list(part)] = i
        rows, cols = np.nonzero(view)
        return float((index[rows] == index[cols]).mean())
    result = dict(algorithm=ctx.layout_algorithm, layout_sha256=ctx.layout_sha256, parts=len(parts),
                  largest_part=max(len(part) for part in parts), equals_graph_native_bfs=parts == reference,
                  anchor_algorithm=old.layout_algorithm, nodes_in_a_different_part_index=moved,
                  edge_share_inside_a_part=dict(bfs=inside(parts), kdtree=inside(old.layout_parts)),
                  positions_kept=ctx.positions is not None,
                  same_layout_in_validation=session.validation_context.layout_sha256 == ctx.layout_sha256,
                  patch_expert_parts_match=[int(n) for n in session.model.patch_expert.ori_parts_idx] == [
                      n for part in parts for n in part])
    assert result['algorithm'] == 'graph_native_component_bfs' and result['equals_graph_native_bfs']
    assert result['anchor_algorithm'] == 'kdtree_on_verified_wgs84' and moved > 0
    assert result['positions_kept'] and result['same_layout_in_validation'] and result['patch_expert_parts_match']
    return result


def preflight(keys=None):
    import torch
    from experiments import v4_runner
    from scripts import train_v4
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')} if (root / 'preflight').exists() else {}
    anchors = {}
    with run_context():
        for ds, a in p['anchors'].items():
            s = v4_runner.build_v4_session(ROOT / a['config'], device='cpu', run_seed=32, num_workers=0)
            cp = torch.load(ROOT / a['checkpoint'], map_location='cpu', weights_only=False)
            ids = identity(s)
            assert all(cp[k] == v for k, v in ids.items()), ds
            anchors[ds] = dict(identity=ids, session=s,
                               parameters={n: t.detach().clone() for n, t in s.model.named_parameters()})
            del cp
    for key, row in sorted(p['configs'].items()):
        if (keys and key not in keys) or reports.get(key, {}).get('passed'):
            continue
        print('PREFLIGHT_START', key, flush=True)
        info = dict(key=key)
        anchor = anchors[row['dataset']]
        with run_context(row):
            s = v4_runner.build_v4_session(ROOT / row['path'], device='cpu', run_seed=32, num_workers=0)
            ids = identity(s)
            differing = sorted(k for k in ids if ids[k] != anchor['identity'][k])
            if row['kind'] == 'layout':
                # The hashes embed the partition; everything else must be the anchor's.
                import numpy as np
                assert differing == ['static_asset_hash', 'train_exposure_hash', 'validation_exposure_hash'], differing
                layout_fields = {'layout_algorithm', 'layout_sha256', 'static_asset_hash'}
                for phase in ('train_context', 'validation_context'):
                    mine, theirs = getattr(s, phase), getattr(anchor['session'], phase)
                    for name, values in ((('static', (mine.static_asset_identity(), theirs.static_asset_identity()))),
                                         ('exposure', (mine.exposure_identity(), theirs.exposure_identity()))):
                        changed = {k for k in values[0] if values[0][k] != values[1].get(k)}
                        assert changed <= layout_fields, (key, phase, name, changed)
                    for field in ('known_nodes', 'validation_unseen', 'test_unseen', 'signal_scope'):
                        assert np.array_equal(getattr(mine, field), getattr(theirs, field)), (key, phase, field)
            else:
                assert differing == [], (key, differing)
            info['identity'] = 'equal to the dataset anchor' + (
                ' except the partition membership and the hashes that embed it' if differing else '')
            names = dict(s.model.named_parameters())
            assert set(names) == set(anchor['parameters'])
            for n, t in names.items():
                assert torch.equal(t, anchor['parameters'][n]), (key, n)
            info['paired_initialization'] = 'every parameter equals the anchor initial value'
            info['parameter_count'] = sum(t.numel() for t in s.model.parameters())
            assert (s.args.max_epoch, s.args.patience, s.args.early_stopping_start_epoch) == (100, 10, 40)
            opt = train_v4.build_optimizer(s)
            info['optimizer'] = dict(cls=type(opt).__name__, learning_rates=[g['lr'] for g in opt.param_groups],
                                     weight_decay=[g['weight_decay'] for g in opt.param_groups])
            reference = train_v4.build_optimizer(anchor['session'])
            assert [(g['lr'], g['weight_decay']) for g in opt.param_groups] == [
                (g['lr'], g['weight_decay']) for g in reference.param_groups]
            if row['kind'] == 'layout':
                assert getattr(s.model, 'study_ablation', 'reference') == 'reference'
                info['layout'] = layout_audit(s, anchor['session'])
            else:
                assert s.model.study_ablation == row['ablation']
                info['audit'] = (role_audit if row['ablation'] == 'no_role' else mask_audit)(s)
            del s
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            report = train_run(p, key, None, device='cuda:0', smoke=True, max_train_batches=20,
                               max_validation_batches=4)
            info.update(passed=True, smoke_epochs=report['epochs_run'],
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3, seconds=time.time() - start)
            torch.cuda.empty_cache()
        write(root / 'preflight' / f'{key}.json', info)
        print('PREFLIGHT_OK', key, f"{info['peak_reserved_gib']:.2f}GiB {info['seconds']:.0f}s", flush=True)
    reports = {f.stem: read(f) for f in (root / 'preflight').glob('*.json')}
    if all(reports.get(k, {}).get('passed') for k in p['configs']):
        write(root / 'PREFLIGHT_COMPLETE.json', dict(passed=True, plan_sha256=p['plan_sha256'],
                                                    reports=[reports[k] for k in sorted(p['configs'])]))


# --- queue -------------------------------------------------------------------------


def worker(key):
    import torch
    torch.set_num_threads(4)
    p = check()
    root = output(p)
    assert read(root / 'PREFLIGHT_COMPLETE.json')['plan_sha256'] == p['plan_sha256']
    assert not (root / f'{key}_complete.json').exists()
    with run_context(p['configs'][key]):
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
    # PEMS-BAY is the longest; start it first.
    order = sorted(p['configs'], key=lambda k: (p['configs'][k]['dataset'] != 'PEMSBAY', k))
    jobs = [dict(key=k, status='queued') for k in order]
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
        # First fit: a small run is not held back by a large one waiting for memory on the shared GPU.
        free = ext.gpu_free_gib()
        pending = next((j for j in jobs if j['status'] == 'queued' and free > peaks[j['key']] + 5.), None)
        if pending and len(active) < MAX_CONCURRENT and time.monotonic() - last > 120:
            need = peaks[pending['key']] + 5.
            if free > need:
                key = pending['key']
                log = (root / f'{key}.log').open('x')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--key', key],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                pending.update(status='running', pid=proc.pid, started=time.time(), gpu='cuda:0',
                               free_gib_at_launch=free, required_gib=need,
                               log=str((root / f'{key}.log').relative_to(ROOT)),
                               command=f'{sys.executable} -u scripts/run_supplement_20261116.py worker --key {key}')
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
        contract = r['model_method_contract']
        if e['kind'] == 'ablation':
            assert contract['paired_ablation'] == e['ablation']
        else:
            assert contract['layout_algorithm'] == 'graph_native_component_bfs'
        best = report_path.with_name('best.pt')
        rows[key] = dict(e, report=str(report_path.relative_to(ROOT)), checkpoint=str(best.relative_to(ROOT)),
                         checkpoint_sha256=sha(best), best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         table=r['development_validation_table'])
    write(root / 'validation_results.json', dict(plan_sha256=p['plan_sha256'], split='validation', rows=rows,
                                                 anchors=p['anchors'], test_accessed=False))
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
