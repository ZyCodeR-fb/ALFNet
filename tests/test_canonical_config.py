"""A run's configuration must have an identity that the run cannot change.

The previous hash was computed from `vars(args)` at checkpoint time. `args` is a
live namespace -- the trainer writes `best_val_metric`, per-epoch history,
resolved output paths and `exp_id` onto it -- so that hash identified the run,
not the configuration: two identical experiments disagreed, and the value moved
during training.

`utils.canonical_config` freezes it the moment parsing finishes, from an
explicit allowlist. These tests pin both halves of the contract: runtime state
must not move it, and anything that defines the experiment must.
"""

import copy

import pytest

from args import load_args_from_config
from utils.canonical_config import (
    build_canonical_config,
    canonical_config_hash,
    freeze_canonical_config,
)


def _args(config="configs/ALFNet/fg1.yaml"):
    return load_args_from_config(config, [])


def _hash(args):
    return canonical_config_hash(build_canonical_config(args))


# --- runtime state must not move the hash ------------------------------------


def test_hash_is_frozen_at_parse_time():
    args = _args()
    assert getattr(args, "canonical_config_hash", None), "parsing did not freeze a hash"
    assert args.canonical_config_hash == _hash(args)


@pytest.mark.parametrize("field,value", [
    ("run_id", "some-other-run"),
    ("output_root", "./elsewhere"),
    ("device", "cpu"),
    ("num_workers", 99),
    ("exp_id", "x" * 40),
    ("best_checkpoint_path", "/tmp/whatever"),
    ("log_dir", "/tmp/logs"),
])
def test_runtime_fields_do_not_change_the_hash(field, value):
    args = _args()
    before = args.canonical_config_hash
    setattr(args, field, value)
    assert _hash(args) == before, f"{field} leaked into the configuration identity"


def test_training_writebacks_do_not_change_the_hash():
    """Exactly what broke the old implementation."""
    args = _args()
    before = args.canonical_config_hash
    args.best_val_metric = 4.2
    args.best_val_metric_epoch = 17
    args.validation_group_history = {"unseen_mae": [1.0, 2.0]}
    args.validation_lifecycle_history = [{"epoch": 1}]
    args.best_routing = {"router/x": 0.5}
    args.num_params = 1234
    assert _hash(args) == before


# --- anything that defines the experiment must move it -----------------------


def test_nested_expert_yaml_changes_the_hash():
    """fg1, gs and g1 differ *only* inside the graph expert's own file.

    A hash that read the top-level YAML alone would call all three the same
    configuration, which is precisely the confusion provenance exists to stop.
    """
    hashes = {
        name: _args(f"configs/ALFNet/{name}.yaml").canonical_config_hash
        for name in ("fg1", "gs", "g1")
    }
    assert len(set(hashes.values())) == 3, hashes


def test_nested_expert_config_contents_are_captured():
    args = _args()
    nested = args.canonical_config["nested_expert_configs"]
    assert set(nested) == {"graph", "patch"}
    assert nested["graph"]["resolved"], "the graph expert YAML was not inlined"


@pytest.mark.parametrize("field,value", [
    ("seed", 999),
    ("lr", 0.5),
    ("max_epoch", 7),
    ("d_model", 128),
    ("input_len", 24),
    ("lcbt_join_ratio", 0.4),
])
def test_experiment_defining_fields_change_the_hash(field, value):
    args = _args()
    before = args.canonical_config_hash
    setattr(args, field, value)
    assert _hash(args) != before, f"{field} does not affect the configuration identity"


def test_model_family_recipe_change_moves_the_hash():
    args = _args()
    before = args.canonical_config_hash
    modified = copy.deepcopy(args)
    modified.model_family_cfg["recipe"]["fusion_mode"] = "residual_gate"
    assert _hash(modified) != before


def test_data_fingerprints_enter_the_hash():
    """Same configuration, different graph, must not share an identity."""
    args = _args()
    first = freeze_canonical_config(args, {"adjacency": "aaa", "data": "bbb"})
    second = freeze_canonical_config(args, {"adjacency": "ccc", "data": "bbb"})
    assert first != second


def test_protocol_versions_enter_the_hash():
    args = _args()
    before = _hash(args)
    args.lce_protocol_version = args.lce_protocol_version + 1
    assert _hash(args) != before
