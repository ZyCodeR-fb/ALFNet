"""2026-11-12: paired interventions of Merlin and STGCN on the METR-LA validation windows of the mechanism figures.

Validation only, frozen seed-32 weights, no training, no test access. The 64 fault-track windows are the ones
frozen in paper/final_figures_20261017/validation/METRLA/pairs.json (not re-selected); the procedure is that of
scripts/final_figures_20261017.py::run. Before the interventions each checkpoint replays the full validation
table and must reproduce its training-time LifeAvg MAE. Outputs go to paper/experiments_20261112/ only.
Authority: user, 2026-11-12 (inference with frozen weights for the paper's analyses; validation preferred).
"""
import json
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_release import read, sha, write  # noqa: E402

DS = 'METRLA'
OLD = ROOT / 'paper/final_figures_20261017/validation' / DS
OUT = ROOT / 'paper/experiments_20261112/interventions/validation' / DS
MERLIN = next((ROOT / 'artifacts/experiments/extension_20261103').glob(
    '*/runs/00_METRLA_Merlin/student/METRLA/Merlin/*/seed32/best.pt'))
ENTRIES = {
    'STGCN': dict(config='artifacts/models/main/METRLA/STGCN/seed32/config.yaml',
                  checkpoint='artifacts/models/main/METRLA/STGCN/seed32/best.pt', context='main'),
    'Merlin': dict(config='configs/paper/extension_20261103/00_METRLA_Merlin.yaml',
                   checkpoint=str(MERLIN.relative_to(ROOT)), context='extension'),
}


def context(kind):
    if kind == 'extension':
        from scripts.run_extension_20261103 import run_context
        return run_context()
    from experiments.flow_support_transfer import adapter as support
    from experiments.flow_unified import adapter as unified
    stack = ExitStack()
    stack.enter_context(support())
    stack.enter_context(unified())
    return stack


def run(model, device='cuda:0'):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from experiments import v4_runner as vr
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.analyze_observation_support import INPUT_FIELDS, arms, intervene, write_csv
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{model}.csv'
    assert not path.exists(), path
    r = ENTRIES[model]
    pairs = read(OLD / 'pairs.json')
    assert len(pairs) == 64 and all(x['track'] == 'fault' for x in pairs)
    cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    with context(r['context']):
        s = vr.build_v4_session(ROOT / r['config'], device=device, run_seed=32, batch_size=16, num_workers=0)
        assert cp['seed'] == 32 and cp['dataset'] == DS and cp['model'] == model and cp['scaler'] == s.scaler
        assert cp['static_asset_hash'] == s.train_context.static_asset_hash()
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
        s.model.load_state_dict(cp['model_state'], strict=True)
        if model == 'Merlin':
            assert bool(s.model.teacher_ready)
        s.model.eval()
        seed = s.seed_contract['derived']['training_runtime']
        seed_process(seed)
        replay = collect_validation_table(s, epoch=cp['epoch'])
        delta = abs(replay['main']['LifeAvg']['MAE'] - cp['best_validation_life_avg_mae'])
        assert delta < 1e-4, (model, delta)
        seed_process(seed)
        ctx, data = s.validation_context, s.validation_loader.dataset
        assert data.context.phase == 'validation'
        # same windows as the stored runs: de-normalised targets must equal the stored ALFNet rows
        old = {}
        import csv
        csv.field_size_limit(10 ** 9)
        for row in csv.DictReader((OLD / 'ALFNet.csv').open(encoding='utf-8')):
            old[(row['condition'], int(row['index']), int(row['node']))] = json.loads(row['target'])
        rows = []
        for arm in arms():
            for start in range(0, len(pairs), 16):
                part = pairs[start:start + 16]
                items = [intervene(data, x['index'], x['node'], arm['scenario'], arm['evidence'], arm['keep'])
                         for x in part]
                batch = vr.to_device(default_collate(items), device)
                teacher = vr.build_model_teacher_mask(s, batch, cp['epoch'], False)
                assert not teacher.any()
                with torch.inference_mode():
                    pred = vr._forward(s, batch, ctx, cp['epoch'], teacher)['pred'].cpu().numpy()
                assert np.isfinite(pred).all()
                for j, (x, item) in enumerate(zip(part, items)):
                    n, std, mean = x['node'], data.scaler['std'], data.scaler['mean']
                    truth = item['target'][:, n, 0].numpy() * std + mean
                    forecast = pred[j, :, n, 0] * std + mean
                    diff = forecast - truth
                    q = int(item['input_available_mask'][:, n].sum())
                    if arm['q'] >= 0:
                        assert q == arm['q']
                    assert np.allclose(truth, old[(arm['condition'], x['index'], n)], rtol=0, atol=1e-9)
                    rows.append(dict(dataset=DS, model=model, condition=arm['condition'],
                                     experiment=arm['experiment'], layout=arm['layout'], q=q, **x,
                                     MAE=float(abs(diff).mean()), MSE=float((diff ** 2).mean()),
                                     MAPE=float((abs(diff) / np.maximum(abs(truth), 1e-6)).mean() * 100),
                                     prediction=json.dumps(forecast.tolist()), target=json.dumps(truth.tolist())))
            print(model, arm['condition'], flush=True)
        assert len(rows) == 1280
        write_csv(path.with_suffix('.tmp'), rows)
        path.with_suffix('.tmp').replace(path)
        write(OUT / f'{model}_audit.json', dict(
            checkpoint=r['checkpoint'], checkpoint_sha256=sha(ROOT / r['checkpoint']), config=r['config'],
            config_sha256=sha(ROOT / r['config']), epoch=int(cp['epoch']), rows=len(rows), teacher_forcing=0,
            split='validation', pairs=str((OLD / 'pairs.json').relative_to(ROOT)), pairs_sha256=sha(OLD / 'pairs.json'),
            validation_replay_lifeavg_mae=replay['main']['LifeAvg']['MAE'], validation_replay_abs_difference=delta,
            input_fields=list(INPUT_FIELDS), test_accessed=False))


if __name__ == '__main__':
    run(sys.argv[1])
