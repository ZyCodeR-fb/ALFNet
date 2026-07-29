"""PEMS-BAY is the confirmation dataset, so its configs must be twins.

METR-LA's test split has been inspected many times and is a development set.
A second dataset only confirms anything if nothing was retuned for it, so every
PEMS-BAY config here must differ from its METR-LA counterpart in the dataset
block alone -- never in capacity, learning rate, or a protocol ratio.

The protocol *ratios* are shared deliberately; the absolute numbers they produce
are not, and are checked here to be genuinely different so that nobody can quietly
reuse METR-LA's manifest. Node counts, split boundaries, unseen strata, cohort
sizes and the training_protocol_hash are all derived per dataset by
`data_provider/lcbt_protocol.py`.
"""

import numpy as np
import pytest
import yaml

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# (METR-LA config, PEMS-BAY config)
TWINS = [
    ("configs/ALFNet/fg1.yaml", "configs/ALFNet/fg1_bay.yaml"),
    ("configs/ALFGraph/METRLA_alfgraph_k1.yaml", "configs/ALFGraph/PEMSBAY_alfgraph_k1.yaml"),
    ("configs/ALFTemporal/METRLA_cond1.yaml", "configs/ALFTemporal/PEMSBAY_cond1.yaml"),
    ("configs/MoGERNN/METRLA_lcbt_v3.yaml", "configs/MoGERNN/PEMSBAY_lcbt_v3.yaml"),
]

# A twin may move the dataset block, its own name, and the paths of the expert
# files it composes -- nothing else.
ALLOWED_PREFIXES = ("dataset.",)
ALLOWED_KEYS = {
    "runtime.model_variant",
    "family.experts.graph.config_path",
    "family.experts.patch.config_path",
    # Geometry, not capacity: the KD-tree partition of a larger node set has
    # larger parts, and `_reorder_data` pads a part but never splits one.
    # test_spa_patchsize_follows_the_partition below pins it to a derived value
    # so it cannot quietly become a tuning knob.
    "family.model.spa_patchsize",
}


def _flat(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


def _load(relative):
    return _flat(yaml.safe_load((ROOT / relative).read_text(encoding="utf-8")))


@pytest.mark.parametrize("metrla,pemsbay", TWINS, ids=[t[1].split("/")[-1] for t in TWINS])
def test_pemsbay_config_differs_only_in_the_dataset_block(metrla, pemsbay):
    left, right = _load(metrla), _load(pemsbay)
    drift = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    unexpected = {
        key for key in drift
        if key not in ALLOWED_KEYS and not key.startswith(ALLOWED_PREFIXES)
    }
    assert not unexpected, (
        f"{pemsbay} was retuned relative to {metrla} at {sorted(unexpected)}"
    )
    # and it really did switch datasets, rather than silently pointing at METR-LA
    assert right["dataset.dloader_name"] == "PEMSBAY"
    assert "PEMS-BAY" in str(right["dataset.data_path"])
    assert "PEMS-BAY" in str(right["dataset.unknown_nodes_path"])


def _protocol_keys(flat):
    """The lcbt / ples ratios. Expert configs (ALFTemporal) carry none."""
    return [
        key for key in flat
        if key.startswith(("lifecycle.lcbt_", "async_test."))
        and not key.endswith(("data_path", "unknown_nodes_path", "manifest_root"))
    ]


@pytest.mark.parametrize("metrla,pemsbay", TWINS, ids=[t[1].split("/")[-1] for t in TWINS])
def test_pemsbay_config_keeps_every_protocol_ratio(metrla, pemsbay):
    """The ratios are the protocol; only what they are applied to may change."""
    left, right = _load(metrla), _load(pemsbay)
    differing = {
        key: (left[key], right.get(key))
        for key in _protocol_keys(left) if left[key] != right.get(key)
    }
    assert not differing, f"{pemsbay} changed protocol ratios: {differing}"


def test_the_twin_table_actually_covers_the_protocol_configs():
    """Guard against the table drifting onto files that carry no protocol at all,
    which would make the check above vacuously pass everywhere."""
    covered = [pair for pair in TWINS if _protocol_keys(_load(pair[0]))]
    assert len(covered) >= 3, (
        f"only {len(covered)} twin pairs carry lcbt/ples keys: {[p[1] for p in covered]}"
    )


def test_spa_patchsize_follows_the_partition():
    """spa_patchsize is geometry, so it must be derivable, not chosen.

    `_reorder_data` pads a KD-tree part up to spa_patchsize but never splits
    one, so a patch smaller than the largest part yields the wrong token count.
    The convention across these configs is `largest part + 1`, which gives every
    part at least one augmented neighbour -- and which reproduces METR-LA's
    existing 8 from its largest part of 7. Pinning it here stops the value from
    drifting into a per-dataset tuning knob.
    """
    from lib.utils import load_project_adjacency, load_project_positions
    from model.ALFTemporal.ALFTemporal import _kd_tree

    cases = [
        ("configs/ALFTemporal/METRLA_cond1.yaml", "./data/METR-LA/raw_data.pkl"),
        ("configs/ALFTemporal/PEMSBAY_cond1.yaml", "./data/PEMS-BAY/raw_data.pkl"),
    ]
    observed = {}
    for config_path, data_path in cases:
        config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
        model_cfg = config["family"]["model"]
        positions = load_project_positions({"data_path": str(ROOT / data_path)})
        parts = _kd_tree(positions.T, int(model_cfg["recur_times"]), 0)
        largest = max(len(part) for part in parts)
        patch = int(model_cfg["spa_patchsize"])
        assert patch == largest + 1, (
            f"{config_path}: spa_patchsize={patch} but the largest KD-tree part "
            f"is {largest}; the derived value is {largest + 1}"
        )
        observed[config_path] = (largest, patch, len(parts))
        # and the adjacency has to agree on the node count the partition covers
        adjacency = load_project_adjacency({"data_path": str(ROOT / data_path)})
        assert sum(len(part) for part in parts) == adjacency.shape[0]

    # the two datasets must genuinely need different geometry, otherwise the
    # exemption in ALLOWED_KEYS is unnecessary and should be removed
    sizes = {value[1] for value in observed.values()}
    assert len(sizes) == 2, f"spa_patchsize is the same on both datasets: {observed}"


def test_reorder_data_cannot_silently_undersize_a_patch():
    """The bug this guards: it fails at forward time, three frames deep.

    `_reorder_data` pads a part up to patch_size and leaves an over-sized part
    alone, so an under-sized patch produces fewer tokens than
    spa_patchnum * spa_patchsize and the model only complains later, inside
    WindowAttBlock's reshape, with no hint of which YAML key is wrong.
    ALFTemporal now checks this at construction; this pins the underlying
    behaviour that makes the check necessary.
    """
    from model.ALFTemporal.ALFTemporal import _reorder_data

    adjacency = np.zeros((12, 12), dtype=np.float32)
    for node in range(12):
        adjacency[node, (node + 1) % 12] = 1.0
    parts = [np.arange(0, 5), np.arange(5, 12)]      # sizes 5 and 7

    # patch_size 7 fits the largest part: 2 * 7 tokens
    _, _, fits = _reorder_data(parts, adjacency, 7)
    assert len(fits) == 2 * 7

    # patch_size 6 does not: the size-7 part is passed through, not split
    _, _, undersized = _reorder_data(parts, adjacency, 6)
    assert len(undersized) != 2 * 6, (
        "an under-sized patch silently produced the right token count"
    )


def test_pemsbay_data_files_exist_and_are_not_metrla():
    """The absolute numbers must be genuinely different, not a copied manifest."""
    shapes = {}
    for name in ("METR-LA", "PEMS-BAY"):
        unknown = np.load(ROOT / f"data/{name}/unknown_nodes.npy", allow_pickle=True)
        shapes[name] = int(unknown.shape[0])
        assert (ROOT / f"data/{name}/raw_data.pkl").exists()
    assert shapes["PEMS-BAY"] != shapes["METR-LA"], (
        "the two datasets report the same unseen count -- check the data files"
    )
    # 40% validation split of the master unseen set, per lcbt_unseen_validation_ratio
    validation = int(round(shapes["PEMS-BAY"] * 0.40))
    assert 1 <= validation < shapes["PEMS-BAY"]
    # PEMS-BAY's validation-unseen group is larger than METR-LA's 20 nodes, which
    # is the group whose small size made METR-LA validation ranking unreliable.
    assert validation > int(round(shapes["METR-LA"] * 0.40))
