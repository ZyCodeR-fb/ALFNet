"""2026-11-13: layout arms of ALFNet with an observation state that follows the visible readings.

Validation only, frozen seed-32 final weights, no training, no test access. Same windows, focal nodes, checkpoints
and layout masks as scripts/paired_interventions_20261113.py. That run kept the focal node in the ON role with
progress 1 while its readings were deleted; the comparison methods do not read the role, so their predictions on
the layout arms do not depend on it (checked: identical to their join / fault arms), and they are not rerun.
Here the role and progress of the focal node are derived step by step from the layout mask, with the conventions
of the loader (data_provider/v4_lifecycle_loader.py::_lifecycle_state). The rule was fixed before this run:
  * steps before the first visible reading: UNO (VS), progress 0;
  * visible steps from the first step up to the first gap: ON (AAS), progress 1;
  * invisible steps after a visible one: OFF (FS), progress k/12, k = steps since the gap began;
  * visible steps after an invisible one: NEW (NAS), progress k/12, k = steps since readings resumed.
Recent-q is then exactly the Join arm with m = q and Early-q the Fault arm with m = 12 - q (asserted against the
stored predictions); Middle gap and Dispersed are compositions of the same states that training did not contain.
Output: paper/experiments_20261113/interventions_state/validation/<DATASET>/ALFNet.npz (+ audit).
Authority: user, 2026-11-13 ("switch the scheme" for the layout figures).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_release import read, sha, write  # noqa: E402
from scripts.paired_interventions_20261113 import BATCH, DATASETS, OUT as SRC, context  # noqa: E402

OUT = ROOT / 'paper/experiments_20261113/interventions_state/validation'
LENGTH = 12


def state(keep):
    """(role, progress) per step for a visibility mask, see the module docstring."""
    from data_provider.lifecycle_roles import ROLE_AAS, ROLE_FS, ROLE_NAS, ROLE_VS
    role, age = np.zeros(LENGTH, dtype=np.int64), np.zeros(LENGTH, dtype=np.float32)
    seen = gap = False
    run = 0
    for t, visible in enumerate(keep):
        run = run + 1 if t and visible == keep[t - 1] else 1
        if visible:
            seen = True
            role[t], age[t] = (ROLE_NAS, run / LENGTH) if gap else (ROLE_AAS, 1.0)
        else:
            gap = True
            role[t], age[t] = (ROLE_FS, run / LENGTH) if seen else (ROLE_VS, 0.0)
    return role, age


def intervene(dataset, index, node, keep):
    import torch
    from data_provider.lifecycle_roles import ROLE_CHANNEL_COUNT
    from scripts.analyze_observation_support import INPUT_FIELDS
    from scripts.analyze_observation_support import intervene as steady
    result = steady(dataset, index, node)  # focal node ON with its 12 natural readings, everything else untouched
    assert bool(result['input_available_mask'][:, node].all())
    role, age = state(keep)
    available = torch.as_tensor(keep, dtype=torch.bool)
    result['input_available_mask'][:, node] = available
    result['base_visible_input'][:, node, 0] = torch.where(available, result['base_visible_input'][:, node, 0], 0.)
    result['lifecycle_role_ids'][:, node] = torch.from_numpy(role)
    result['lifecycle_state_age'][:, node] = torch.from_numpy(age)
    result['lifecycle_context'][:, node] = torch.cat([
        torch.nn.functional.one_hot(torch.from_numpy(role), ROLE_CHANNEL_COUNT).float(),
        torch.from_numpy(age).unsqueeze(-1), available.float().unsqueeze(-1)], dim=-1)
    result['decoder_available_mask'][:, node] = available[-1] & result['target_available_mask'][:, node]
    base = dataset[index]
    others = torch.arange(dataset.num_nodes) != node
    for key in INPUT_FIELDS:
        assert torch.equal(result[key][:, others], base[key][:, others]), key
    return result


def run(ds, device='cuda:0'):
    import torch
    from torch.utils.data import default_collate
    from experiments import v4_runner as vr
    from experiments.v4_validation_collector import collect_validation_table
    from lib.v4_seeds import seed_process
    from scripts.analyze_observation_support import LAYOUTS, layout_mask
    from scripts.eval_release import validation_agrees
    torch.set_num_threads(4)
    dest = OUT / ds
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'ALFNet.npz'
    assert not path.exists(), path
    r = read(SRC / ds / 'selection.json')['models']['ALFNet']
    wins = read(SRC / ds / 'windows.json')
    old = np.load(SRC / ds / 'ALFNet.npz')
    old_cond = [str(c) for c in old['conditions']]
    arm_list = [(f'q{q}_{k}', layout_mask(q, k)) for q in (3, 6, 9) for k in LAYOUTS]
    cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    assert sha(ROOT / r['checkpoint']) == r['checkpoint_sha256'] and cp['seed'] == 32 and cp['epoch'] == r['epoch']
    with context('ALFNet', r):
        s = vr.build_v4_session(ROOT / r['config'], device=device, run_seed=32, num_workers=0)
        assert cp['scaler'] == s.scaler and cp['static_asset_hash'] == s.train_context.static_asset_hash()
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
        s.model.load_state_dict(cp['model_state'], strict=True)
        s.model.eval()
        seed = s.seed_contract['derived']['training_runtime']
        seed_process(seed)
        ok, delta = validation_agrees(collect_validation_table(s, epoch=cp['epoch']), r['table'])
        assert ok, (ds, delta)
        seed_process(seed)
        ctx, data = s.validation_context, s.validation_loader.dataset
        assert data.context.phase == 'validation'
        std, mean = float(data.scaler['std']), float(data.scaler['mean'])
        pred = np.zeros((len(arm_list), len(wins), 12), dtype=np.float32)
        roles = np.zeros((len(arm_list), 12), dtype=np.int64)
        ages = np.zeros((len(arm_list), 12), dtype=np.float32)
        for a, (name, keep) in enumerate(arm_list):
            roles[a], ages[a] = state(keep)
            for start in range(0, len(wins), BATCH):
                part = wins[start:start + BATCH]
                items = [intervene(data, x['index'], x['node'], keep) for x in part]
                batch = vr.to_device(default_collate(items), device)
                teacher = vr.build_model_teacher_mask(s, batch, cp['epoch'], False)
                assert not teacher.any()
                with torch.inference_mode():
                    out = vr._forward(s, batch, ctx, cp['epoch'], teacher)['pred'].cpu().numpy()
                assert np.isfinite(out).all()
                for j, (x, item) in enumerate(zip(part, items)):
                    pred[a, start + j] = out[j, :, x['node'], 0] * std + mean
                    truth = item['target'][:, x['node'], 0].numpy() * std + mean
                    assert np.allclose(truth, old['target'][start + j], atol=1e-4)
            print(ds, name, flush=True)
    # Recent-q is the Join arm with m = q, Early-q the Fault arm with m = 12 - q: same inputs, same predictions.
    replay = {}
    for new, stored in (('q3_recent', 'join_3'), ('q6_recent', 'join_6'), ('q9_early', 'fault_3'), ('q6_early', 'fault_6')):
        a = [n for n, _ in arm_list].index(new)
        replay[f'{new}~{stored}'] = float(np.abs(pred[a] - old['pred'][old_cond.index(stored)]).max())
    assert max(replay.values()) < 0.05, replay
    np.savez_compressed(path.with_suffix('.tmp.npz'), pred=pred, target=old['target'], roles=roles, ages=ages,
                        conditions=np.array([n for n, _ in arm_list]), index=old['index'], node=old['node'])
    path.with_suffix('.tmp.npz').replace(path)
    write(dest / 'ALFNet_audit.json', dict(
        dataset=ds, model='ALFNet', checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
        config=r['config'], config_sha256=r['config_sha256'], epoch=int(cp['epoch']), windows=len(wins),
        arms=[n for n, _ in arm_list], teacher_forcing=0, split='validation', test_accessed=False,
        validation_replay_max_abs_difference=delta, replay_of_stored_join_fault_arms_max_abs_difference=replay,
        state_rule='leading invisible: UNO, 0; visible before the first gap: ON, 1; invisible after visible: OFF, k/12; '
                   'visible after invisible: NEW, k/12', comparison_methods='not rerun: they do not read the role'))
    print(ds, 'done', replay, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    run(parser.parse_args().dataset)
