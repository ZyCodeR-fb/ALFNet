"""CPU checks of the 2026-11-10 three-seed study of ALFNet, MCPST and MetaDG (no training, no test access)."""
import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_seeds2_20261110 as study  # noqa: E402


def test_epoch0_comparison_tolerates_only_small_float_noise():
    old = {'train': {'base': 1.0, 'aux': 5e-7, 'batches': 10, 'ratio': 1.0, 'policy': 'x', 'scale_mix': None},
           'validation': {'mae': 40.0, 'count': 100}}
    same = json.loads(json.dumps(old))
    same['train']['nonfinite_gradient_steps'] = []
    assert study.compare_epoch0(same, old)[0] == []
    noisy = json.loads(json.dumps(same))
    noisy['train']['base'] = 1.0 + 3e-4  # observed run-to-run spread of epoch-mean training losses
    noisy['train']['aux'] = 5e-7 * (1 + 1e-2)  # tiny auxiliary term: within the absolute tolerance
    noisy['validation']['mae'] = 40.0 * (1 + 4e-2)  # observed one-epoch nondeterministic spread
    assert study.compare_epoch0(noisy, old)[0] == []
    drift = json.loads(json.dumps(same))
    drift['train']['base'] = 1.0 * (1 + 5e-3)
    assert study.compare_epoch0(drift, old)[0]
    gross = json.loads(json.dumps(same))
    gross['validation']['mae'] = 40.0 * 1.2
    assert study.compare_epoch0(gross, old)[0]
    count = json.loads(json.dumps(same))
    count['train']['batches'] = 11
    assert study.compare_epoch0(count, old)[0]
    guarded = json.loads(json.dumps(same))
    guarded['train']['nonfinite_gradient_steps'] = [{'epoch': 0}]
    assert study.compare_epoch0(guarded, old)[0]


def test_contexts_follow_the_seed32_training_path():
    from experiments import v4_runner
    base = v4_runner.build_v4_session
    with study.context('ALFNet', 'bare'):
        assert v4_runner.build_v4_session is base
    for name in ('support', 'unified', 'support+unified'):
        with study.context('ALFNet', name):
            assert v4_runner.build_v4_session is not base
    with pytest.raises(AssertionError):
        with study.context('ALFNet', 'repair'):
            pass
    with study.context('MCPST'):
        assert 'MCPST' in v4_runner.MODEL_ENTRY_POINTS and 'MetaDG' in v4_runner.MODEL_ENTRY_POINTS
    assert 'MCPST' not in v4_runner.MODEL_ENTRY_POINTS


def test_every_final_study_has_a_known_context():
    final = json.loads((ROOT / 'paper/pems08_wd_test_20261012/selected_configuration_manifest.json').read_text())
    for e in final['entries'].values():
        assert Path(e['checkpoint']).parts[2] in study.STUDY_CONTEXT


@pytest.mark.skipif(not study.PLAN.exists(), reason='plan not frozen yet')
def test_frozen_plan_covers_three_methods_four_datasets_three_seeds():
    p = json.loads(study.PLAN.read_text())
    pairs = {(c['dataset'], c['model']) for c in p['configs'].values()}
    assert len(pairs) == 12 and len(p['reuse']) == 12
    assert collections.Counter(c['seed'] for c in p['configs'].values()) == {32: 12, 42: 12, 52: 12}
    for c in p['configs'].values():
        r = p['reuse'][f"{c['dataset']}_{c['model']}_seed32"]
        assert c['conditional'] == (c['seed'] == 32)
        assert c['path'] == r['config'] and c['sha256'] == r['config_sha256'] == p['files'][c['path']]
        assert p['files'][r['checkpoint']] == r['checkpoint_sha256']
    assert not any(r['reusable'] for r in p['historical_seed_runs'])
    sel = p['alfnet_selection']['entries']
    assert {d: p['reuse'][f'{d}_ALFNet_seed32']['checkpoint_sha256'] for d in sel} == \
        {d: e['checkpoint_sha256'] for d, e in sel.items()}


def _toy_training(nan_batches):
    import types
    import torch

    def build_optimizer(session):
        return torch.optim.SGD(session.model.parameters(), lr=0.1)

    def train_one_epoch(session, opt, epoch):
        for i in range(len(session.train_loader)):
            opt.zero_grad()
            session.model(torch.ones(1, 2)).sum().backward()
            if i in nan_batches:
                session.model.weight.grad[0, 0] = float('nan')
            opt.step()
        return dict(batches=len(session.train_loader))
    return types.SimpleNamespace(build_optimizer=build_optimizer, train_one_epoch=train_one_epoch)


def test_recovery_guard_skips_records_and_enforces_the_10_percent_limit(tmp_path):
    import types
    import torch
    from scripts import recover_seeds2_metadg_20261110 as recovery
    assert recovery.RECOVERY_EVENT_FRACTION == 0.10
    session = types.SimpleNamespace(model=torch.nn.Linear(2, 1), train_loader=list(range(10)))
    training = _toy_training({3})
    with recovery.relaxed_guard(training, tmp_path / 'events.json'):
        r = training.train_one_epoch(session, training.build_optimizer(session), 0)
    assert len(r['nonfinite_gradient_steps']) == 1 and r['nonfinite_gradient_fraction'] == 0.1
    assert r['nonfinite_gradient_steps'][0]['optimiser_call'] == 3 and torch.isfinite(session.model.weight).all()
    assert json.loads((tmp_path / 'events.json').read_text())[0]['parameters'] == ['weight']
    training = _toy_training({3, 7})
    with pytest.raises(AssertionError):
        with recovery.relaxed_guard(training, tmp_path / 'events2.json'):
            training.train_one_epoch(session, training.build_optimizer(session), 0)
