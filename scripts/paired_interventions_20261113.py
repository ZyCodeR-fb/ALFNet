"""2026-11-13: paired observation interventions on every eligible validation window, four datasets.

Validation only, frozen seed-32 final weights, no training, no test access.
Differences from the 64-window runs (scripts/final_figures_20261017.py, scripts/paired_interventions_20261112.py):
  * windows: every fault-track validation schedule entry that has a focal node (one of its changed nodes,
    all known) whose 12 history + 12 future readings are naturally complete; one focal node per window drawn
    with a fixed seed. No subsampling, nothing is selected by prediction or error.
  * models: ALFNet and, per dataset, the method of each of the four comparison groups (Section 2) with the
    lowest seed-32 validation LifeAvg MAE on that dataset (rule fixed before the interventions are run).
The intervention arms are those of scripts/analyze_observation_support.py (6 trajectory + 14 support arms).
Before the interventions each checkpoint replays the full validation table and must reproduce the stored one.
Outputs: paper/experiments_20261113/interventions/validation/<DATASET>/ (windows.json, selection.json,
<model>.npz with per-window predictions, <model>_audit.json). Modes: select | run --dataset D.
Authority: user, 2026-11-13 (items 3-4: show 4.5/4.6/4.8 on four datasets, more windows).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_release import read, sha, write  # noqa: E402

OUT = ROOT / 'paper/experiments_20261113/interventions/validation'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
GROUPS = {'unseen': ('MoGERNN', 'SLPF'), 'incomplete': ('GinAR', 'Merlin'), 'shift': ('STONE', 'MCPST', 'STBP'),
          'backbone': ('STGCN', 'STID', 'PatchSTG', 'MetaDG')}  # make_tables.py GROUPS
PLANS = ('configs/paper/seeds_test_20261106/plan.json', 'configs/paper/seeds2_test_20261110/plan.json')
WINDOW_SEED = 20261113
BATCH = 32


def final_seed32():
    """Final seed-32 checkpoint of every (dataset, model): a rerun entry if there is one, else the reused run."""
    rows = {}
    for i, f in enumerate(PLANS):
        p = read(ROOT / f)
        for v in p['reuse'].values():
            rows[(v['dataset'], v['model'])] = dict(
                config=v['config'], config_sha256=v['config_sha256'], checkpoint=v['checkpoint'],
                checkpoint_sha256=v['checkpoint_sha256'], epoch=v['epoch'], table=v['validation_table'],
                study_context=v.get('study_context'), plan=f, second=bool(i))
        for v in p['entries'].values():
            if v['seed'] == 32:
                rows[(v['dataset'], v['model'])] = dict(
                    config=v['path'], config_sha256=v['sha256'], checkpoint=v['checkpoint'],
                    checkpoint_sha256=v['checkpoint_sha256'], epoch=v['epoch'], table=v['table'],
                    study_context=v.get('study_context'), plan=f, second=bool(i))
    return rows


def select():
    rows = final_seed32()
    for ds in DATASETS:
        dest = OUT / ds
        dest.mkdir(parents=True, exist_ok=True)
        assert not (dest / 'selection.json').exists()
        models, scores = {'ALFNet': dict(rows[(ds, 'ALFNet')], group='ours')}, {}
        for g, members in GROUPS.items():
            scores[g] = {m: rows[(ds, m)]['table']['main']['LifeAvg']['MAE'] for m in members}
            best = min(scores[g], key=scores[g].get)
            models[best] = dict(rows[(ds, best)], group=g)
        for r in models.values():
            assert sha(ROOT / r['checkpoint']) == r['checkpoint_sha256'] and sha(ROOT / r['config']) == r['config_sha256']
        write(dest / 'selection.json', dict(
            dataset=ds, rule='per comparison group, lowest seed-32 validation LifeAvg MAE on this dataset',
            validation_lifeavg_mae=scores, models=models, split='validation', test_accessed=False))
        print(ds, {m: r['group'] for m, r in models.items()})


def context(model, r):
    from scripts import run_seeds_20261106 as seeds1
    from scripts import run_seeds2_20261110 as seeds2
    return seeds2.context(model, r['study_context']) if r['second'] else seeds1.context(model)


def windows(data):
    rng = np.random.default_rng(WINDOW_SEED)
    out = []
    for i, e in enumerate(data.schedule):
        if e['scenario'] != 'fault':
            continue
        start = int(data.origins[i])
        nodes = [int(n) for n in e['changed_nodes'] if data.natural[start:start + 24, n].all()]
        if nodes:
            node = int(rng.choice(nodes))
            assert node in data.context.known_nodes
            out.append(dict(index=int(i), node=node, origin=start, regime=e['calendar_regime']))
    return out


def run(ds, only=None, device='cuda:0'):
    import torch
    from torch.utils.data import default_collate
    from experiments import v4_runner as vr
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.analyze_observation_support import INPUT_FIELDS, arms, intervene
    from scripts.eval_release import validation_agrees
    torch.set_num_threads(4)
    dest = OUT / ds
    sel = read(dest / 'selection.json')
    arm_list = list(arms())
    for model, r in sel['models'].items():
        path = dest / f'{model}.npz'
        if path.exists() or (only and model not in only):
            continue
        try:  # one worker per model when several processes share a dataset
            (dest / f'{model}.lock').open('x').close()
        except FileExistsError:
            continue
        cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
        assert sha(ROOT / r['checkpoint']) == r['checkpoint_sha256'] and cp['seed'] == 32 and cp['epoch'] == r['epoch']
        with context(model, r):
            s = vr.build_v4_session(ROOT / r['config'], device=device, run_seed=32, num_workers=0)  # default batch: exact replay
            assert cp['scaler'] == s.scaler and cp['static_asset_hash'] == s.train_context.static_asset_hash()
            assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
            s.model.load_state_dict(cp['model_state'], strict=True)
            if model == 'Merlin':
                assert bool(s.model.teacher_ready)
            s.model.eval()
            seed = s.seed_contract['derived']['training_runtime']
            seed_process(seed)
            ok, delta = validation_agrees(collect_validation_table(s, epoch=cp['epoch']), r['table'])
            assert ok, (ds, model, delta)
            seed_process(seed)
            ctx, data = s.validation_context, s.validation_loader.dataset
            assert data.context.phase == 'validation'
            wins = windows(data)
            try:  # every model must see the same windows
                with (dest / 'windows.json').open('x') as handle:
                    json.dump(wins, handle)
            except FileExistsError:
                pass
            assert read(dest / 'windows.json') == wins
            std, mean = float(data.scaler['std']), float(data.scaler['mean'])
            pred = np.zeros((len(arm_list), len(wins), 12), dtype=np.float32)
            target = np.zeros((len(wins), 12), dtype=np.float32)
            history = np.zeros((len(wins), 12), dtype=np.float32)
            q = np.zeros((len(arm_list), len(wins)), dtype=np.int16)
            for a, arm in enumerate(arm_list):
                for start in range(0, len(wins), BATCH):
                    part = wins[start:start + BATCH]
                    items = [intervene(data, x['index'], x['node'], arm['scenario'], arm['evidence'], arm['keep'])
                             for x in part]
                    batch = vr.to_device(default_collate(items), device)
                    teacher = vr.build_model_teacher_mask(s, batch, cp['epoch'], False)
                    assert not teacher.any()
                    with torch.inference_mode():
                        out = vr._forward(s, batch, ctx, cp['epoch'], teacher)['pred'].cpu().numpy()
                    assert np.isfinite(out).all()
                    for j, (x, item) in enumerate(zip(part, items)):
                        n, w = x['node'], start + j
                        pred[a, w] = out[j, :, n, 0] * std + mean
                        q[a, w] = int(item['input_available_mask'][:, n].sum())
                        if arm['q'] >= 0:
                            assert q[a, w] == arm['q']
                        truth = item['target'][:, n, 0].numpy() * std + mean
                        if a == 0:
                            assert arm['condition'] == 'full' and q[a, w] == 12
                            target[w] = truth
                            history[w] = item['base_visible_input'][:, n, 0].numpy() * std + mean
                        else:
                            assert np.allclose(truth, target[w], atol=1e-4)
                print(ds, model, arm['condition'], flush=True)
        np.savez_compressed(path.with_suffix('.tmp.npz'), pred=pred, target=target, history=history, q=q,
                            conditions=np.array([x['condition'] for x in arm_list]),
                            index=np.array([x['index'] for x in wins]), node=np.array([x['node'] for x in wins]))
        path.with_suffix('.tmp.npz').replace(path)
        write(dest / f'{model}_audit.json', dict(
            dataset=ds, model=model, group=r['group'], checkpoint=r['checkpoint'],
            checkpoint_sha256=r['checkpoint_sha256'], config=r['config'], config_sha256=r['config_sha256'],
            epoch=int(cp['epoch']), windows=len(wins), arms=len(arm_list), teacher_forcing=0, split='validation',
            window_seed=WINDOW_SEED, validation_replay_max_abs_difference=delta, input_fields=list(INPUT_FIELDS),
            test_accessed=False))
        del s, cp
        torch.cuda.empty_cache()
    if all((dest / f'{m}.npz').exists() for m in sel['models']):
        write(dest / 'COMPLETE.json', dict(models=list(sel['models']), windows=len(read(dest / 'windows.json')),
                                           arms=len(arm_list), test_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['select', 'run'])
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--models', help='comma-separated subset (parallel workers)')
    a = parser.parse_args()
    select() if a.mode == 'select' else run(a.dataset, a.models.split(',') if a.models else None)
