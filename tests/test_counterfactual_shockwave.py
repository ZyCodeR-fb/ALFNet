import importlib.util
from pathlib import Path

import numpy as np
import torch

from data_provider.graph_data_loader import ROLE_AAS, ROLE_FS, ROLE_NAS, ROLE_VS


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shockwave = _load_module("counterfactual_shockwave", "experiments/counterfactual_shockwave.py")
plot_shockwave = _load_module("plot_counterfactual_shockwave", "scripts/plot_counterfactual_shockwave.py")
build_source_hop_partition = shockwave.build_source_hop_partition
lifecycle_phase = shockwave.lifecycle_phase
aggregate_counterfactual_gain = plot_shockwave.aggregate_counterfactual_gain


class _TinyLifecycleDataset:
    def __init__(self):
        self.step = 1
        self.input_len = 4
        self.look_back = 0
        self.pred_len = 2
        self.in_feas = [0]
        self.lifecycle_role_features = True
        self.lifecycle_history_age_feature = True
        self.data = np.arange(20, dtype=np.float32).reshape(10, 2, 1) + 1.0
        self.data_y = self.data.copy()
        self.availability_mask = np.ones((10, 2), dtype=bool)
        self.availability_mask[:4, 0] = False
        self.availability_mask[4:, 1] = False

    def _build_role_ids_for_window(self, input_begin, input_end):
        times = np.arange(input_begin, input_end)[:, None]
        roles = np.full((input_end - input_begin, 2), ROLE_AAS, dtype=np.int64)
        roles[:, 0] = np.where(times[:, 0] >= 4, ROLE_NAS, ROLE_VS)
        roles[:, 1] = np.where(times[:, 0] >= 4, ROLE_FS, ROLE_AAS)
        return roles

    def __getitem__(self, index):
        input_begin = index
        input_end = input_begin + self.input_len
        output_end = input_end + self.pred_len
        raw_x = self.data[input_begin:input_end]
        raw_y = self.data_y[input_end:output_end]
        return raw_x, raw_y, np.zeros_like(raw_x), np.zeros_like(raw_y)


def test_lifecycle_phase_uses_all_six_paper_facing_states():
    assert lifecycle_phase("commissioning", -1, input_len=12) == "Cold-start"
    assert lifecycle_phase("commissioning", 0, input_len=12) == "Warm-up"
    assert lifecycle_phase("commissioning", 11, input_len=12) == "Warm-up"
    assert lifecycle_phase("commissioning", 12, input_len=12) == "Adapted"

    assert lifecycle_phase("failure", -1, input_len=12) == "Steady"
    assert lifecycle_phase("failure", 0, input_len=12) == "Degrading"
    assert lifecycle_phase("failure", 11, input_len=12) == "Degrading"
    assert lifecycle_phase("failure", 12, input_len=12) == "Outage"


def test_source_hop_partition_assigns_each_node_once_to_nearest_source():
    adjacency = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    partition = build_source_hop_partition(adjacency, sources=[0, 5], max_hop=3)

    assert partition[0][0] == [0]
    assert partition[0][1] == [1]
    assert partition[0][2] == [2]
    assert partition[5][0] == [5]
    assert partition[5][1] == [4]
    assert partition[5][2] == [3]
    assigned = [node for hop_groups in partition.values() for nodes in hop_groups.values() for node in nodes]
    assert sorted(assigned) == list(range(6))


def test_intervention_sample_separates_commissioning_and_failure_shocks():
    dataset = _TinyLifecycleDataset()
    event = {"event_time_index": 4, "nas_nodes": [0], "fs_nodes": [1]}

    control = shockwave.build_intervention_sample(dataset, index=2, event=event, trajectory="control")[0]
    commissioning = shockwave.build_intervention_sample(dataset, index=2, event=event, trajectory="commissioning")[0]
    failure = shockwave.build_intervention_sample(dataset, index=2, event=event, trajectory="failure")[0]

    assert np.isnan(control[:, 0, 0]).all()
    assert np.isfinite(control[:, 1, 0]).all()
    assert np.isnan(commissioning[:2, 0, 0]).all()
    assert np.isfinite(commissioning[2:, 0, 0]).all()
    assert np.isfinite(commissioning[:, 1, 0]).all()
    assert np.isnan(failure[:, 0, 0]).all()
    assert np.isfinite(failure[:2, 1, 0]).all()
    assert np.isnan(failure[2:, 1, 0]).all()

    raw_channels = 1
    role_channels = commissioning[..., raw_channels : raw_channels + 4]
    assert np.all(role_channels[2:, 0, ROLE_NAS] == 1.0)
    assert np.all(role_channels[:, 1, ROLE_AAS] == 1.0)
    assert np.allclose(commissioning[:, 0, -1], [0.0, 0.0, 0.25, 0.5])


def test_counterfactual_gain_is_a_paired_difference_in_differences():
    keys = {
        "event_id": 1,
        "trajectory": "failure",
        "source_sensor": 7,
        "tau": 0,
        "phase": "Degrading",
        "hop": 0,
        "count": 10,
    }
    ours_rows = [
        {
            **keys,
            "treated_ape_sum": 1.2,
            "control_ape_sum": 1.0,
            "treated_ae_sum": 12.0,
            "control_ae_sum": 10.0,
            "treated_se_sum": 20.0,
            "control_se_sum": 16.0,
        }
    ]
    baseline_rows = [
        {
            **keys,
            "treated_ape_sum": 1.5,
            "control_ape_sum": 1.0,
            "treated_ae_sum": 15.0,
            "control_ae_sum": 10.0,
            "treated_se_sum": 25.0,
            "control_se_sum": 16.0,
        }
    ]

    cells = aggregate_counterfactual_gain(ours_rows, baseline_rows, bootstrap_samples=0)

    assert len(cells) == 1
    assert np.isclose(cells[0]["ours_ace_mape"], 2.0)
    assert np.isclose(cells[0]["baseline_ace_mape"], 5.0)
    assert np.isclose(cells[0]["attenuation_gain_mape"], 3.0)


def test_cluster_bootstrap_resamples_changed_sensors_as_paired_units():
    ours_rows = []
    baseline_rows = []
    for source_sensor in range(4):
        shared = {
            "event_id": 1,
            "trajectory": "commissioning",
            "source_sensor": source_sensor,
            "tau": 3,
            "phase": "Warm-up",
            "hop": 1,
            "count": 10,
            "control_ape_sum": 2.0,
            "control_ae_sum": 20.0,
            "control_se_sum": 40.0,
        }
        ours_rows.append(
            {
                **shared,
                "treated_ape_sum": 1.0,
                "treated_ae_sum": 10.0,
                "treated_se_sum": 20.0,
            }
        )
        baseline_rows.append(
            {
                **shared,
                "treated_ape_sum": 1.5,
                "treated_ae_sum": 15.0,
                "treated_se_sum": 30.0,
            }
        )

    cells = aggregate_counterfactual_gain(ours_rows, baseline_rows, bootstrap_samples=100, seed=7)

    assert np.isclose(cells[0]["attenuation_gain_mape"], 5.0)
    assert np.isclose(cells[0]["ci_low"], 5.0)
    assert np.isclose(cells[0]["ci_high"], 5.0)
    assert cells[0]["significant"] is True


def test_paired_error_sums_use_one_common_valid_mask():
    treated = torch.tensor([[[[12.0], [float("nan")]]]])
    control = torch.tensor([[[[11.0], [9.0]]]])
    target = torch.tensor([[[[10.0], [10.0]]]])

    sums = shockwave.paired_error_sums(treated, control, target, nodes=[0, 1])

    assert sums["count"] == 1
    assert np.isclose(sums["treated_ape_sum"], 0.2)
    assert np.isclose(sums["control_ape_sum"], 0.1)
    assert np.isclose(sums["treated_ae_sum"], 2.0)
    assert np.isclose(sums["control_ae_sum"], 1.0)
