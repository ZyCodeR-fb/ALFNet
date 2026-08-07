"""Provenance must survive a machine move without ever accepting a wrong graph.

The failure this pins down, measured rather than imagined: the r1 checkpoints
were trained on the server and evaluated here. The graph was identical, but
`build_structural_features` sums rows of the adjacency, and a reduction's order
depends on the BLAS build and the thread count -- so the derived features came
out 1.31e-07 apart, one float32 ulp. The schema-1 digest folded those exact
float bytes into a single hash, so every bit of it changed and
`_validate_structural_fingerprints` refused every server checkpoint. That is a
portability bug: it blocks evaluation without protecting anything.

The same cut applies to the configuration hash. Schema 1 folded the `-c` path
into the identity, so `-c configs/ALFNet/gs.yaml` and the same file named by an
absolute path were different experiments, and a paper-facing rename of an
unchanged model changed its identity.

Both are widened here, and both directions are tested: a checkpoint that only
drifted must load, and a checkpoint from a genuinely different graph, dataset,
configuration or router must still be refused.
"""

import copy

import pytest
import torch
from torch import nn

from args import load_args_from_config
from layers.structural_node_encoder import (
    FEATURE_ATOL,
    FEATURE_QUANTUM,
    STRUCTURE_PROVENANCE_VERSION,
    StructuralNodeEncoder,
    build_structural_features,
    quantised_feature_digest,
    raw_structure_digest,
    structural_fingerprints,
    structure_provenance,
)
from main import _validate_checkpoint_provenance, _validate_structural_fingerprints
from utils.canonical_config import (
    build_canonical_config,
    canonical_config_hash,
)


def _graph(num_nodes=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    adjacency = torch.rand(num_nodes, num_nodes, generator=generator)
    adjacency = adjacency * (adjacency > 0.5)
    positions = torch.rand(num_nodes, 2, generator=generator)
    return adjacency, positions


class _Host(nn.Module):
    """Smallest model that owns a structural encoder, so the validators run."""

    def __init__(self, adjacency, positions, out_dim=8):
        super().__init__()
        self.encoder = StructuralNodeEncoder(adjacency, positions, out_dim=out_dim)


def _checkpoint(model, fingerprints=None, **config):
    payload = {
        "model": "ALFNet",
        "model_variant": "gs",
        "canonical_config_hash": "unused",
        "structural_fingerprints": (
            structural_fingerprints(model) if fingerprints is None else fingerprints
        ),
    }
    payload.update(config)
    return {"config": payload, "state_dict": copy.deepcopy(model.state_dict())}


def _nudge_one_ulp(tensor):
    """The drift a different BLAS produces: one float32 ulp on every entry."""
    moved = tensor.clone()
    step = torch.finfo(torch.float32).eps
    return moved + moved.abs().clamp_min(1.0) * step


# --- the quantised feature digest --------------------------------------------


def test_the_feature_digest_is_not_relied_on_to_survive_drift():
    """Pins the reason the raw digest exists, so nobody 'simplifies' it away.

    Quantising was the obvious fix for the ulp problem and it is not sufficient:
    a grid still has boundaries and a value near one crosses under an
    arbitrarily small perturbation. Measured over 30 random graphs, 9.65% of
    values changed bucket under a one-ulp nudge and every digest differed. So
    identity rests on `raw_structure_digest` plus a bounded comparison of the
    stored buffer -- never on this value.
    """
    adjacency, positions = _graph(seed=1)
    features = build_structural_features(adjacency, positions)
    drifted = _nudge_one_ulp(features)

    assert not torch.equal(features, drifted), "the test must actually perturb something"
    assert (features - drifted).abs().max() < FEATURE_ATOL
    # The raw digest is the one that must not move, and it does not: it never
    # sees a derived quantity.
    assert raw_structure_digest(adjacency, positions) == raw_structure_digest(
        adjacency, positions
    )


def test_a_real_feature_change_still_moves_the_digest():
    adjacency, positions = _graph(seed=1)
    features = build_structural_features(adjacency, positions)
    changed = features.clone()
    changed[0, 0] += 1e-3
    assert quantised_feature_digest(features) != quantised_feature_digest(changed)


def test_feature_digest_covers_shape_and_quantum():
    adjacency, positions = _graph(num_nodes=6, seed=2)
    features = build_structural_features(adjacency, positions)
    base = quantised_feature_digest(features)
    assert base != quantised_feature_digest(features[:5])
    assert base != quantised_feature_digest(features, quantum=1e-5)


# --- the raw digest is what decides identity ---------------------------------


def test_raw_digest_separates_different_graphs():
    first, positions = _graph(seed=3)
    second, _ = _graph(seed=4)
    assert raw_structure_digest(first, positions) != raw_structure_digest(second, positions)


def test_raw_digest_separates_different_coordinates():
    adjacency, positions = _graph(seed=3)
    moved = positions.clone()
    moved[0, 0] += 0.5
    assert raw_structure_digest(adjacency, positions) != raw_structure_digest(adjacency, moved)


def test_raw_digest_ignores_derived_features_entirely():
    """Raw identity must not move when only a derived quantity drifts."""
    adjacency, positions = _graph(seed=3)
    encoder = StructuralNodeEncoder(adjacency, positions, out_dim=8)
    before = encoder.structure_provenance["raw"]
    encoder.features.copy_(_nudge_one_ulp(encoder.features))
    after = structure_provenance(adjacency, positions, encoder.features)["raw"]
    assert before == after


def test_provenance_records_the_metadata_needed_to_reproduce_it():
    adjacency, positions = _graph(seed=5)
    record = structure_provenance(
        adjacency, positions, build_structural_features(adjacency, positions)
    )
    assert record["version"] == STRUCTURE_PROVENANCE_VERSION
    assert record["quantum"] == FEATURE_QUANTUM
    assert record["shape"] == [adjacency.shape[0], 7]
    assert record["dtype"] == "torch.float32"
    assert record["algorithm"] == "build_structural_features"


def test_structural_fingerprints_report_full_records_not_bare_strings():
    adjacency, positions = _graph(seed=6)
    reported = structural_fingerprints(_Host(adjacency, positions))
    assert reported, "a model with a structural encoder reported none"
    for record in reported.values():
        assert isinstance(record, dict)
        assert record["raw"] and record["features"]


# --- schema 2 checkpoints ----------------------------------------------------


def test_schema2_checkpoint_loads_after_a_one_ulp_move():
    adjacency, positions = _graph(seed=7)
    model = _Host(adjacency, positions)
    checkpoint = _checkpoint(model)
    checkpoint["state_dict"]["encoder.features"] = _nudge_one_ulp(
        checkpoint["state_dict"]["encoder.features"]
    )
    _validate_structural_fingerprints(checkpoint, model)


def test_schema2_checkpoint_rejects_drift_beyond_tolerance():
    adjacency, positions = _graph(seed=7)
    model = _Host(adjacency, positions)
    checkpoint = _checkpoint(model)
    stored = checkpoint["state_dict"]["encoder.features"].clone()
    stored[0, 0] += 1e-3
    checkpoint["state_dict"]["encoder.features"] = stored
    checkpoint["config"]["structural_fingerprints"]["encoder"]["features"] = (
        quantised_feature_digest(stored)
    )
    with pytest.raises(ValueError, match="beyond the"):
        _validate_structural_fingerprints(checkpoint, model)


def test_schema2_checkpoint_rejects_a_different_graph():
    adjacency, positions = _graph(seed=7)
    other, other_positions = _graph(seed=8)
    model = _Host(adjacency, positions)
    checkpoint = _checkpoint(_Host(other, other_positions))
    with pytest.raises(ValueError, match="raw structure digest"):
        _validate_structural_fingerprints(checkpoint, model)


def test_schema2_checkpoint_rejects_a_different_node_count():
    adjacency, positions = _graph(num_nodes=6, seed=7)
    smaller, smaller_positions = _graph(num_nodes=5, seed=7)
    model = _Host(adjacency, positions)
    checkpoint = _checkpoint(_Host(smaller, smaller_positions))
    with pytest.raises(ValueError):
        _validate_structural_fingerprints(checkpoint, model)


# --- schema 1 checkpoints, which is what is already on disk ------------------


def _schema1_checkpoint(model):
    checkpoint = _checkpoint(model)
    checkpoint["config"]["structural_fingerprints"] = {
        "encoder": model.encoder.structure_hash
    }
    return checkpoint


def test_schema1_checkpoint_loads_after_a_one_ulp_move():
    """The exact situation on disk: server-trained, evaluated elsewhere."""
    adjacency, positions = _graph(seed=9)
    model = _Host(adjacency, positions)
    checkpoint = _schema1_checkpoint(model)
    drifted = _nudge_one_ulp(model.encoder.features)
    checkpoint["state_dict"]["encoder.features"] = drifted
    checkpoint["config"]["structural_fingerprints"]["encoder"] = (
        model.encoder.legacy_digest_for(drifted)
    )
    _validate_structural_fingerprints(checkpoint, model)


def test_schema1_checkpoint_rejects_a_different_graph():
    adjacency, positions = _graph(seed=9)
    other, other_positions = _graph(seed=10)
    model = _Host(adjacency, positions)
    checkpoint = _schema1_checkpoint(_Host(other, other_positions))
    with pytest.raises(ValueError, match="different graph"):
        _validate_structural_fingerprints(checkpoint, model)


def test_schema1_mismatch_without_a_feature_buffer_is_refused():
    """No silent pass: an unverifiable checkpoint must raise, not be accepted."""
    adjacency, positions = _graph(seed=9)
    model = _Host(adjacency, positions)
    checkpoint = _schema1_checkpoint(model)
    checkpoint["config"]["structural_fingerprints"]["encoder"] = "0" * 64
    del checkpoint["state_dict"]["encoder.features"]
    with pytest.raises(ValueError, match="cannot be verified"):
        _validate_structural_fingerprints(checkpoint, model)


def test_an_encoder_missing_from_the_model_is_refused():
    adjacency, positions = _graph(seed=9)
    model = _Host(adjacency, positions)
    checkpoint = _checkpoint(model)
    checkpoint["config"]["structural_fingerprints"]["absent.encoder"] = (
        checkpoint["config"]["structural_fingerprints"]["encoder"]
    )
    with pytest.raises(ValueError, match="absent from this model"):
        _validate_structural_fingerprints(checkpoint, model)


# --- the configuration hash --------------------------------------------------


def _args(config="configs/ALFNet/gs.yaml"):
    return load_args_from_config(config, [])


def test_config_path_spelling_does_not_change_the_identity():
    import os

    relative = _args("configs/ALFNet/gs.yaml")
    absolute = _args(os.path.abspath("configs/ALFNet/gs.yaml"))
    assert relative.canonical_config_hash == absolute.canonical_config_hash


def test_the_same_content_under_a_new_filename_keeps_its_identity(tmp_path):
    """A paper-facing rename must not invalidate an unchanged model."""
    import shutil

    source = "configs/ALFNet/gs.yaml"
    renamed = tmp_path / "alfnet_paper.yaml"
    shutil.copyfile(source, renamed)
    assert _args(source).canonical_config_hash == _args(str(renamed)).canonical_config_hash


def test_the_paper_config_is_the_same_experiment_as_gs():
    """`alfnet.yaml` is the paper-facing name for the frozen mainline.

    It must resolve to the identical experiment, or the existing gs checkpoints
    stop loading under the name the paper uses -- which is the whole reason the
    config path was taken out of the identity.
    """
    paper = _args("configs/ALFNet/alfnet.yaml")
    gs = _args("configs/ALFNet/gs.yaml")
    assert paper.canonical_config_hash == gs.canonical_config_hash
    assert paper.model == gs.model == "ALFNet"
    assert paper.model_variant == gs.model_variant


def test_a_content_change_still_changes_the_identity():
    args = _args()
    before = args.canonical_config_hash
    args.learning_rate = getattr(args, "learning_rate", 0.001) + 0.5
    assert canonical_config_hash(build_canonical_config(args)) != before


def test_legacy_schema1_hash_is_available_and_differs_by_the_path_only():
    import os

    relative = _args("configs/ALFNet/gs.yaml")
    absolute = _args(os.path.abspath("configs/ALFNet/gs.yaml"))
    assert relative.legacy_canonical_config_hash != relative.canonical_config_hash
    # schema 1 was path dependent; that is precisely why it was replaced
    assert relative.legacy_canonical_config_hash != absolute.legacy_canonical_config_hash


def test_a_schema1_config_hash_is_accepted_without_a_flag():
    args = _args()
    checkpoint = {"config": {
        "model": args.model,
        "model_variant": getattr(args, "model_variant", None),
        "canonical_config_hash": args.legacy_canonical_config_hash,
    }}
    _validate_checkpoint_provenance(checkpoint, args)


def test_an_unrelated_config_hash_is_still_refused():
    args = _args()
    checkpoint = {"config": {
        "model": args.model,
        "model_variant": getattr(args, "model_variant", None),
        "canonical_config_hash": "f" * 64,
    }}
    with pytest.raises(ValueError, match="canonical_config_hash"):
        _validate_checkpoint_provenance(checkpoint, args)


def test_a_different_router_is_still_fatal():
    args = _args()
    checkpoint = {"config": {
        "model": args.model,
        "model_variant": getattr(args, "model_variant", None),
        "canonical_config_hash": args.canonical_config_hash,
        "router_mode": "noisy_topk",
    }}
    with pytest.raises(ValueError, match="different graph router"):
        _validate_checkpoint_provenance(checkpoint, args)


def test_a_different_variant_is_still_fatal():
    args = _args()
    checkpoint = {"config": {
        "model": args.model,
        "model_variant": "something-else",
        "canonical_config_hash": args.canonical_config_hash,
    }}
    with pytest.raises(ValueError, match="different model"):
        _validate_checkpoint_provenance(checkpoint, args)
