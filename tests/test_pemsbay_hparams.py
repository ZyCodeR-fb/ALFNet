import copy

from scripts.run_pemsbay_hparams import candidate_config, digest


def fixture_config():
    return {'runtime': {'model_variant': 'baseline'}, 'training': {'lr': 5e-5, 'loss': 'mae_focal_raw02'},
            'family': {'experts': {'patch': {'config_path': 'old.yaml'}},
                       'recipe': {'graph_expert_lr': 5e-4, 'patch_expert_lr': 1e-4, 'gate_lr': 5e-5}}}


def test_lr_scan_changes_actual_groups_without_mutating_baseline():
    base = fixture_config()
    original = copy.deepcopy(base)
    patch = {'family': {'model': {'spa_patchsize': 11}}}
    outer, nested = candidate_config(base, patch, 'lr_double')
    assert base == original and patch['family']['model']['spa_patchsize'] == 11
    assert outer['family']['recipe'] == {'graph_expert_lr': 1e-3, 'patch_expert_lr': 2e-4, 'gate_lr': 1e-4}
    assert nested == patch and outer['training']['loss'] == base['training']['loss']


def test_capacity_scan_does_not_change_training_recipe_or_other_patch_parameters():
    base = fixture_config()
    patch = {'family': {'model': {'spa_patchsize': 11, 'layers': 1, 'crossformer_dropout': .1}}}
    outer, nested = candidate_config(base, patch, 'patch12')
    assert outer['family']['recipe'] == base['family']['recipe'] and outer['training'] == base['training']
    assert nested['family']['model'] == {'spa_patchsize': 12, 'layers': 1, 'crossformer_dropout': .1}


def test_plan_binds_scope_and_does_not_include_its_own_hash():
    p = {'test_authorized': False, 'max_concurrent': 5}
    h = digest(p)
    p['plan_sha256'] = h
    assert digest(p) == h
    p['test_authorized'] = True
    assert digest(p) != h
