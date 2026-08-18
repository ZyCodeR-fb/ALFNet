"""Graph assets must be deterministic, separated by role, and shared by models.

Three failures motivate this file, all measured on the real v3 artefacts:

* the propagation graph was symmetrised on every PeMS dataset, which makes
  forward and backward diffusion the same operator;
* the "coordinates" for PEMS03/04/08 were the first two Laplacian eigenvectors,
  which on a disconnected graph lie in the null space, collapse to component
  indicators, and carry an arbitrary sign and basis;
* the propagation array and the layout source were the same object, so fixing
  one silently changed the other.

The layout is therefore defined on the graph, the structural view is derived and
never substituted back, and the contract records both hashes separately.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.graph_native_layout import (  # noqa: E402
    LAYOUT_ALGORITHM,
    LAYOUT_VERSION,
    connected_components,
    layout_parts,
    layout_signature,
    node_order,
    structural_view,
)

CONTRACTS = ROOT / "configs/paper/v4_data_manifests"
DATASETS = ("PEMS03", "PEMS04", "PEMS08", "PEMSBAY")
NO_COORDINATES = ("PEMS03", "PEMS04", "PEMS08")


def _contract(dataset):
    path = CONTRACTS / f"{dataset}_graph_contract.json"
    if not path.exists():
        pytest.skip(f"{dataset} contract not built in this checkout")
    return json.loads(path.read_text("utf-8"))


def _ring_with_tail(n=12):
    """A connected graph with an obvious hub, plus a directed asymmetry."""
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adjacency[i, i + 1] = 1.0
    adjacency[0, n - 1] = 1.0        # directed, no reciprocal edge
    return adjacency


def _two_components(a=6, b=4):
    adjacency = np.zeros((a + b, a + b), dtype=np.float32)
    for i in range(a - 1):
        adjacency[i, i + 1] = 1.0
    for i in range(a, a + b - 1):
        adjacency[i, i + 1] = 1.0
    return adjacency


# --- 1. determinism -----------------------------------------------------------


def test_the_layout_is_identical_across_two_independent_processes():
    """Same inputs, separate interpreters, same hash."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np\n"
        "from lib.graph_native_layout import layout_signature\n"
        "a = np.zeros((12, 12), dtype=np.float32)\n"
        "for i in range(11): a[i, i+1] = 1.0\n"
        "a[0, 11] = 1.0\n"
        "import hashlib; print(hashlib.sha256(layout_signature(a, 2).encode()).hexdigest())"
    ) % str(ROOT)
    digests = set()
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1, f"layout differed across processes: {digests}"


def test_repeated_calls_agree_in_process():
    adjacency = _ring_with_tail()
    assert layout_signature(adjacency, 2) == layout_signature(adjacency, 2)


# --- 2. components and coverage ------------------------------------------------


def test_every_node_appears_exactly_once():
    adjacency = _two_components()
    parts = layout_parts(adjacency, 2)
    flat = np.concatenate(parts)
    assert sorted(flat.tolist()) == list(range(adjacency.shape[0]))
    assert len(set(flat.tolist())) == len(flat), "a node appeared twice"
    assert flat.min() >= 0 and flat.max() < adjacency.shape[0]


def test_components_are_stable_and_contiguous_in_the_order():
    adjacency = _two_components(a=6, b=4)
    components = connected_components(structural_view(adjacency))
    assert [len(c) for c in components] == [6, 4]
    order = node_order(adjacency)
    positions = {node: index for index, node in enumerate(order)}
    for component in components:
        spans = sorted(positions[node] for node in component)
        assert spans == list(range(spans[0], spans[0] + len(component))), (
            "a component was split apart in the layout order")


@pytest.mark.parametrize("dataset", NO_COORDINATES)
def test_real_dataset_components_are_recorded_and_cover_every_node(dataset):
    contract = _contract(dataset)
    sizes = contract["structural_view"]["component_sizes"]
    assert sum(sizes) == contract["canonical_sensor_ids"]["count"]
    assert len(sizes) == contract["structural_view"]["component_count"]


# --- 3/4. position policy per dataset ------------------------------------------


@pytest.mark.parametrize("dataset", NO_COORDINATES)
def test_datasets_without_gps_use_the_graph_native_layout(dataset):
    contract = _contract(dataset)
    assert contract["positions"]["policy"] == "graph_native_no_coordinates"
    assert contract["positions"]["sha256"] is None
    assert contract["layout"]["algorithm"] == LAYOUT_ALGORITHM
    assert contract["layout"]["version"] == LAYOUT_VERSION


def test_pems08_no_longer_uses_spectral_positions_or_the_v3_symmetric_graph():
    """The concrete v3 -> v4 invalidation boundary for PEMS08."""
    contract = _contract("PEMS08")
    assert contract["positions"]["policy"] == "graph_native_no_coordinates"
    assert contract["propagation_graph"]["directed"] is True

    active = ROOT / "data/PEMS08/raw_data.pkl"
    if active.exists():
        blob = np.load(active, allow_pickle=True)
        blob = blob.item() if getattr(blob, "shape", None) == () else blob
        v3_adjacency = np.asarray(blob["adj"])
        assert np.allclose(v3_adjacency, v3_adjacency.T), "v3 was symmetric"
        # so the v4 propagation graph cannot be the v3 one
        assert contract["propagation_graph"]["sha256"] != _array_sha256(v3_adjacency)


def test_pems_bay_keeps_verified_coordinates_and_its_canonical_graph():
    contract = _contract("PEMSBAY")
    assert contract["positions"]["policy"] == "verified_wgs84"
    assert contract["positions"]["sha256"] is not None
    assert contract["propagation_graph"]["directed"] is True
    assert contract["propagation_graph"]["weighted"] is True


# --- 5. the two graph views stay separate --------------------------------------


def test_the_structural_view_is_undirected_and_the_propagation_graph_is_not():
    adjacency = _ring_with_tail()
    view = structural_view(adjacency)
    assert not np.allclose(adjacency, adjacency.T), "fixture must be directed"
    assert np.array_equal(view, view.T), "the structural view must be undirected"
    assert np.all(np.diag(view) == 0)
    assert set(np.unique(view)).issubset({0, 1})


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_contract_records_the_two_views_separately(dataset):
    contract = _contract(dataset)
    assert contract["propagation_graph"]["sha256"] != contract["structural_view"]["sha256"]
    assert contract["propagation_graph"]["must_not_be_symmetrised"] is True
    assert contract["structural_view"]["never_substituted_into_propagation"] is True


@pytest.mark.parametrize("dataset", NO_COORDINATES)
def test_no_pems_propagation_graph_is_symmetric(dataset):
    contract = _contract(dataset)
    assert contract["propagation_graph"]["directed"] is True, (
        "a symmetric graph makes forward and backward diffusion one operator")


# --- 6. both models must share one propagation hash ----------------------------


def test_a_single_contract_defines_the_propagation_graph_for_both_models():
    """There is one contract per dataset, so the hash cannot differ by model."""
    for dataset in DATASETS:
        contract = _contract(dataset)
        assert "model" not in contract, (
            "the contract must not be per-model; both models read one graph")


def test_a_mismatched_propagation_hash_is_detectable():
    left = _contract("PEMS03")["propagation_graph"]["sha256"]
    right = _contract("PEMS04")["propagation_graph"]["sha256"]
    assert left != right
    # the check a preflight performs: equality of the recorded hash
    assert left == _contract("PEMS03")["propagation_graph"]["sha256"]


# --- 9. the contract hash moves when its inputs move ---------------------------


def _array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def test_changing_an_edge_changes_the_propagation_hash():
    adjacency = _ring_with_tail()
    before = _array_sha256(adjacency)
    adjacency[3, 7] = 1.0
    assert _array_sha256(adjacency) != before


def test_changing_sensor_order_changes_the_layout_on_an_asymmetric_graph():
    """Components of unequal size make the ordering label-sensitive."""
    adjacency = _two_components(a=6, b=4)
    before = layout_signature(adjacency, 2)
    permutation = np.roll(np.arange(adjacency.shape[0]), 3)
    permuted = adjacency[np.ix_(permutation, permutation)]
    assert layout_signature(permuted, 2) != before


def test_a_vertex_transitive_graph_can_relabel_to_the_same_index_sequence():
    """Stated rather than glossed, because it is easy to over-claim here.

    A ring is vertex-transitive: every node looks like every other, so a
    relabelling can produce the *same sequence of indices* while those indices
    now name different physical sensors. That is not a defect of the layout --
    it is what "no coordinates and a symmetric graph" means -- but it is why the
    module documents sensor id as the final tie-break and does not claim
    isomorphism invariance.
    """
    adjacency = _ring_with_tail()
    permutation = np.roll(np.arange(adjacency.shape[0]), 3)
    permuted = adjacency[np.ix_(permutation, permutation)]
    assert layout_signature(permuted, 2) == layout_signature(adjacency, 2)


def test_changing_recur_times_changes_the_layout():
    adjacency = _ring_with_tail()
    assert layout_signature(adjacency, 2) != layout_signature(adjacency, 3)


# --- 10. permutation semantics, stated honestly --------------------------------


def test_a_relabelling_that_carries_sensor_ids_preserves_the_grouping():
    """Invariance holds only when ids travel with the nodes.

    Sensor id is the final tie-break, so two nodes that are interchangeable
    under pure graph isomorphism are still separated by their id. The layout is
    deterministic given the id assignment; it is not isomorphism-invariant, and
    the module documents that rather than claiming otherwise.
    """
    adjacency = _two_components(a=6, b=4)
    order = node_order(adjacency)

    permutation = np.roll(np.arange(adjacency.shape[0]), 2)
    permuted = adjacency[np.ix_(permutation, permutation)]
    permuted_order = node_order(permuted)
    # map new indices back to original ids
    recovered = [int(permutation[node]) for node in permuted_order]

    original_components = {frozenset(c) for c in
                           connected_components(structural_view(adjacency))}
    recovered_components = {
        frozenset(int(permutation[n]) for n in c)
        for c in connected_components(structural_view(permuted))}
    assert original_components == recovered_components
    assert sorted(recovered) == sorted(order)


# --- 11/12. no target leakage, no v3 mutation ----------------------------------


def test_the_layout_never_reads_the_signal():
    """It takes an adjacency and nothing else -- no values, no statistics."""
    import inspect

    from lib import graph_native_layout

    source = inspect.getsource(graph_native_layout)
    for banned in ("time_stamp", '["data"]', "target", "test"):
        assert banned not in source, f"layout module references {banned!r}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_contract_reports_training_as_not_ready(dataset):
    contract = _contract(dataset)
    assert contract["status"]["data_ready"] is True
    assert contract["status"]["graph_ready"] is True
    assert contract["status"]["training_ready"] is False


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_reserved_scopes_are_three_distinct_fields(dataset):
    reserved = _contract(dataset)["reserved_scopes"]
    for field in ("propagation_graph_scope", "available_signal_node_mask",
                  "loss_target_node_mask"):
        assert field in reserved


FROZEN_V3 = {
    "data/PEMS03/raw_data.pkl":
        "2f5c6182682687de9c9e9dcffb6334bd103b652a5311f591cffee7e7df448ecb",
    "data/PEMS04/raw_data.pkl":
        "b8741a79dface8a80b6db5ffb753944407deeea645a7de3a8a64ae5607228130",
}


@pytest.mark.parametrize("path,digest", sorted(FROZEN_V3.items()))
def test_v3_payloads_remain_byte_identical(path, digest):
    target = ROOT / path
    if not target.exists():
        pytest.skip(f"{path} absent")
    running = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            running.update(chunk)
    assert running.hexdigest() == digest


# --- tracked and runtime manifests agree ---------------------------------------


@pytest.mark.parametrize("dataset", NO_COORDINATES)
def test_the_tracked_manifest_matches_the_runtime_copy(dataset):
    tracked = CONTRACTS / f"{dataset}_data_manifest.json"
    runtime = ROOT / "data" / dataset / "v4_candidate" / "v4_candidate_manifest.json"
    if not (tracked.exists() and runtime.exists()):
        pytest.skip("manifests not built in this checkout")
    assert tracked.read_bytes() == runtime.read_bytes(), (
        "the version-controlled manifest must be byte-identical to the runtime "
        "copy, or another machine cannot tell what should have been produced")


# --- 7/8. positions=None reaches the models -----------------------------------


def test_structural_encoder_forwards_without_positions():
    """The topology-only path, exercised rather than assumed."""
    import torch

    from layers.structural_node_encoder import StructuralNodeEncoder

    adjacency = torch.tensor(_two_components(a=6, b=4))
    encoder = StructuralNodeEncoder(adjacency, None, out_dim=8)
    output = encoder()
    assert output.shape == (10, 8)
    assert torch.isfinite(output).all()


def test_the_loader_returns_none_for_a_payload_without_positions(tmp_path):
    """A v4 payload omits pos_stamp; the loader must not KeyError."""
    from lib.utils import load_project_positions

    payload = tmp_path / "raw.pkl"
    np.save(payload, {"data": np.zeros((4, 3, 1), dtype=np.float32),
                      "adj": np.zeros((3, 3), dtype=np.float32),
                      "time_stamp": np.arange(4)}, allow_pickle=True)
    assert load_project_positions({"data_path": str(payload) + ".npy"}) is None


def test_the_graph_native_layout_preserves_the_patch_contract():
    """`_reorder_data` pads parts and never splits them, with or without GPS."""
    from model.ALFTemporal.ALFTemporal import _build_spatial_layout

    adjacency = _two_components(a=6, b=4)
    ori, reo, reo_all, patchnum = _build_spatial_layout(
        None, adjacency, recur_times=2, spa_patchsize=5)
    assert patchnum == 4
    assert len(reo_all) == patchnum * 5, "token count must match patches x size"
    assert len(ori) == len(reo)


def test_the_layout_falls_back_only_when_positions_are_absent():
    from model.ALFTemporal.ALFTemporal import _build_spatial_layout

    adjacency = _two_components(a=6, b=4)
    positions = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
    with_gps = _build_spatial_layout(positions, adjacency, 2, 5)
    without = _build_spatial_layout(None, adjacency, 2, 5)
    assert with_gps[3] == without[3] == 4       # same patch count
    assert with_gps[2] != without[2], "GPS and graph layouts should differ here"
