import copy
from pathlib import Path
import pytest
import yaml
from scripts.run_metrla_hparams import BASE, ROOT, SPECS, candidate_config


@pytest.mark.parametrize('key', list(SPECS))
def test_each_candidate_changes_only_declared_factor(key):
    base = yaml.safe_load((BASE / 'config.yaml').read_text())
    patch = yaml.safe_load((ROOT / base['family']['experts']['patch']['config_path']).read_text())
    original = copy.deepcopy((base, patch))
    outer, nested = candidate_config(base, patch, key)
    capacity, multiplier, dropout, decay = SPECS[key]
    assert (base, patch) == original
    assert sum((capacity != 8, multiplier != 1, dropout != .1, decay != 1e-4)) == 1
    assert nested['family']['model']['spa_patchsize'] == capacity
    assert nested['family']['model']['crossformer_dropout'] == dropout
    assert outer['training']['weight_decay'] == decay
    assert outer['training']['lr'] == base['training']['lr'] * multiplier
    for group in ('graph_expert_lr', 'patch_expert_lr', 'gate_lr'):
        assert outer['family']['recipe'][group] == base['family']['recipe'][group] * multiplier
        outer['family']['recipe'][group] = base['family']['recipe'][group]
    nested['family']['model']['spa_patchsize'] = 8
    nested['family']['model']['crossformer_dropout'] = .1
    outer['training']['weight_decay'] = 1e-4
    outer['training']['lr'] = base['training']['lr']
    outer['runtime']['model_variant'] = base['runtime']['model_variant']
    outer['family']['experts']['patch']['config_path'] = base['family']['experts']['patch']['config_path']
    assert (outer, nested) == original


def test_collection_does_not_overwrite_bay_results():
    import inspect
    from scripts.run_metrla_hparams import collect, freeze
    assert 'hparams_pemsbay' not in inspect.getsource(collect)
    assert 'hparams_metrla_20260920/PLAN.md' in inspect.getsource(freeze)
