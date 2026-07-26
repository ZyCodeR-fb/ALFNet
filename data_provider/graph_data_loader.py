import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import warnings
from data_provider.calendar_strata import (
    DAY_TYPES,
    STRATA,
    TOD_BINS,
    calendar_stratum,
    select_stratified_origins,
    stratum_weights,
    strip_diagnostics,
    summarize_origins,
)
from data_provider.lcbt_protocol import (
    LCBT_GROUPS,
    LCBT_PHASES,
    LCBT_PROTOCOL_NAME,
    LCBT_PROTOCOL_VERSION,
    array_hash,
    build_lcbt_schedule,
    compose_effective_lifecycle_features,
    lcbt_manifest_hash,
    resolve_lcbt_boundaries,
    select_master_unseen_nodes,
    split_unseen_nodes,
    stable_hash,
)
from data_provider.ples_protocol import build_protocol_manifest, main_table_cohort_plan
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_CHANNEL_COUNT = 4


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

def _unwrap_raw_data(raw_data):
    if isinstance(raw_data, np.ndarray) and raw_data.shape == ():
        return raw_data.item()
    return raw_data


def _lcbt_window_regimes(timestamps, segment_start, segment_end, input_len, pred_len):
    segment_length = int(segment_end) - int(segment_start)
    num_samples = (segment_length - int(input_len) - int(pred_len)) + 1
    if num_samples <= 0:
        raise ValueError("LCBT segment is too short for the requested input/prediction lengths")
    if timestamps is None:
        return [STRATA[0]] * num_samples
    return [
        calendar_stratum(timestamps[int(segment_start) + index + int(input_len) - 1])
        for index in range(num_samples)
    ]


def _lcbt_validation_origins(
    timestamps,
    segment_start,
    segment_end,
    input_len,
    pred_len,
    per_stratum,
    values=None,
    min_observed_fraction=0.30,
):
    """Stratified low-discrepancy validation origins.

    The event instant is the last input step, so the selector works on window
    indices and the caller adds the segment offset.
    """
    segment_length = int(segment_end) - int(segment_start)
    if segment_length - int(input_len) - int(pred_len) + 1 <= 0:
        raise ValueError("LCBT validation segment is too short")
    segment_values = (
        None
        if values is None
        else np.asarray(values)[int(segment_start):int(segment_end)]
    )
    origins = select_stratified_origins(
        None if timestamps is None else np.asarray(timestamps)[int(segment_start):int(segment_end)],
        segment_length,
        input_len,
        pred_len,
        pre_steps=0,
        post_steps=0,
        per_stratum=per_stratum,
        values=segment_values,
        min_observed_fraction=min_observed_fraction,
        # Every record of a validation origin replays the same window with a
        # different lifecycle mask, so one origin touches exactly one episode
        # rather than the tau sweep the test protocol runs.
        min_spacing=int(input_len) + int(pred_len),
    )
    # The selector reports the event instant; the replay needs the window start.
    for origin in origins:
        origin["window_index"] = int(origin["local_index"]) - int(input_len) + 1
    return [origin for origin in origins if origin["window_index"] >= 0]


def _lcbt_schedule_summary(schedule, num_nodes):
    scenario_histogram = {name: 0 for name in LCBT_GROUPS}
    phase_histogram = {phase: 0 for phase in LCBT_PHASES}
    calendar_crosstab = {}
    cohort_histogram = {}
    topology_histogram = {"clustered": 0, "scattered": 0}
    evidence_histogram = {"join": {}, "fault": {}}
    evidence_calendar_crosstab = {}
    exposure = np.zeros(int(num_nodes), dtype=np.int64)
    for item in schedule:
        scenario = str(item["scenario"])
        scenario_histogram[scenario] += 1
        regime = str(item["calendar_regime"])
        calendar_crosstab.setdefault(regime, {name: 0 for name in LCBT_GROUPS})
        calendar_crosstab[regime][scenario] += 1
        if int(item.get("phase_id", -1)) >= 0:
            phase_histogram[LCBT_PHASES[int(item["phase_id"])]] += 1
        fraction_key = f"{float(item['cohort_fraction']):.6f}"
        cohort_histogram[fraction_key] = cohort_histogram.get(fraction_key, 0) + 1
        topology_histogram[str(item["topology_mode"])] += 1
        if scenario in evidence_histogram:
            evidence = str(int(item["evidence_steps"]))
            evidence_histogram[scenario][evidence] = (
                evidence_histogram[scenario].get(evidence, 0) + 1
            )
            evidence_calendar_crosstab.setdefault(regime, {}).setdefault(
                scenario,
                {},
            )
            evidence_calendar_crosstab[regime][scenario][evidence] = (
                evidence_calendar_crosstab[regime][scenario].get(evidence, 0) + 1
            )
        exposure[np.asarray(item["changed_nodes"], dtype=np.int64)] += 1
    return {
        "num_samples": int(len(schedule)),
        "scenario_histogram": scenario_histogram,
        "phase_histogram": phase_histogram,
        "cohort_fraction_histogram": cohort_histogram,
        "topology_histogram": topology_histogram,
        "calendar_crosstab": calendar_crosstab,
        "evidence_histogram": evidence_histogram,
        "evidence_calendar_crosstab": evidence_calendar_crosstab,
        "node_exposure": {
            "minimum": int(exposure.min()) if len(exposure) else 0,
            "mean": float(exposure.mean()) if len(exposure) else 0.0,
            "maximum": int(exposure.max()) if len(exposure) else 0,
            "counts": exposure.astype(int).tolist(),
        },
    }


class LifecycleCoveragePEMS(Dataset):
    """Versioned LCBT train/validation dataset with strict chronological splits."""

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path="",
        unknown_nodes_list=None,
        unknown_nodes_num=0,
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        lifecycle_observation_mask_feature=True,
        lcbt_manifest_root="./protocol_manifests",
        lcbt_protocol_seed=2026,
        lcbt_validation_seed=12026,
        lcbt_train_end_ratio=0.50,
        lcbt_val_end_ratio=0.60,
        lcbt_test_start_ratio=0.60,
        lcbt_scaler_end_ratio=0.50,
        lcbt_join_ratio=0.25,
        lcbt_fault_ratio=0.25,
        lcbt_steady_ratio=0.25,
        lcbt_zero_history_ratio=0.25,
        lcbt_cohort_fractions=(0.025, 0.05, 0.10),
        lcbt_cohort_weights=(0.25, 0.50, 0.25),
        lcbt_clustered_ratio=0.50,
        lcbt_unseen_validation_ratio=0.40,
        lcbt_generated_unseen_fraction=0.25,
        lifecycle_age_mode="state_duration",
        lcbt_overwrite_manifest=False,
        lifecycle_training_strategy="lcbt_v3",
        lcbt_validation_origins_per_stratum=2,
        lcbt_min_observed_fraction=0.30,
    ):
        if flag not in {"train", "val"}:
            raise ValueError("LifecycleCoveragePEMS only supports train/val splits")
        self.flag = flag
        self.data_path = data_path
        self.input_len, self.look_back, self.pred_len = [int(value) for value in size]
        if self.look_back > self.input_len:
            raise ValueError("look_back should be smaller than input_len")
        self.scale = bool(scale)
        self.inverse = inverse
        self.freq = freq
        self.dloader_name = dloader_name
        self.step = 1
        self.unknown_nodes_path = unknown_nodes_path
        self.unknown_nodes = list(unknown_nodes_list or [])
        self.unknown_nodes_num = int(unknown_nodes_num)
        self.lifecycle_role_features = bool(lifecycle_role_features)
        self.lifecycle_history_age_feature = bool(lifecycle_history_age_feature)
        self.lifecycle_observation_mask_feature = bool(lifecycle_observation_mask_feature)
        self.input_features = input_features
        self.target_features = target_features
        self.manifest_root = Path(lcbt_manifest_root)
        self.protocol_seed = int(lcbt_protocol_seed)
        self.validation_seed = int(lcbt_validation_seed)
        self.overwrite_manifest = bool(lcbt_overwrite_manifest)
        self.unseen_validation_ratio = float(lcbt_unseen_validation_ratio)
        self.generated_unseen_fraction = float(lcbt_generated_unseen_fraction)
        self.lifecycle_age_mode = str(lifecycle_age_mode)
        self.lifecycle_training_strategy = str(lifecycle_training_strategy)
        self.validation_origins_per_stratum = int(lcbt_validation_origins_per_stratum)
        self.min_observed_fraction = float(lcbt_min_observed_fraction)
        self.validation_origins = []
        self.validation_origin_summary = {}
        if self.lifecycle_training_strategy != "lcbt_v3":
            raise ValueError("LifecycleCoveragePEMS requires lcbt_v3")
        if self.lifecycle_age_mode != "state_duration":
            raise ValueError(
                f"{self.lifecycle_training_strategy} requires lifecycle_age_mode=state_duration"
            )
        self.split_ratios = {
            "train_end_ratio": float(lcbt_train_end_ratio),
            "val_end_ratio": float(lcbt_val_end_ratio),
            "test_start_ratio": float(lcbt_test_start_ratio),
            "scaler_end_ratio": float(lcbt_scaler_end_ratio),
        }
        self.scenario_ratios = (
            float(lcbt_zero_history_ratio),
            float(lcbt_steady_ratio),
            float(lcbt_join_ratio),
            float(lcbt_fault_ratio),
        )
        self.cohort_fractions = tuple(float(value) for value in lcbt_cohort_fractions)
        self.cohort_weights = tuple(float(value) for value in lcbt_cohort_weights)
        self.clustered_ratio = float(lcbt_clustered_ratio)
        self.current_epoch = 0
        self._read_data(input_features, target_features)
        self.protocol_manifest = self._ensure_protocol_manifest()
        self.training_protocol_hash = self.protocol_manifest["protocol_hash"]
        self._refresh_schedule()

    def _read_data(self, input_features, target_features):
        feature_map = {
            "q": [0],
            "o": [1],
            "v": [2],
            "qv": [0, 2],
            "qov": [0, 1, 2],
        }
        self.in_feas = feature_map[input_features]
        self.tar_feas = feature_map[target_features]
        payload = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))
        raw_source = np.asarray(payload["data"])
        self.raw_data_shape = [int(value) for value in raw_source.shape]
        self.raw_data_hash = array_hash(raw_source)
        raw_values = raw_source.astype(np.float32, copy=True)
        raw_values[raw_values <= 0] = np.nan
        self.full_timestamps = payload.get("time_stamp")
        self.timestamp_hash = (
            array_hash(np.asarray(self.full_timestamps))
            if self.full_timestamps is not None
            else None
        )
        self.full_num_nodes = int(raw_values.shape[1])
        self.full_adjacency = np.asarray(payload["adj"])
        self.adjacency_hash = array_hash(self.full_adjacency)
        # Targets on the full timeline, used only by the origin availability
        # guard so that an origin buried in a network-wide outage is rejected.
        self.guard_values = raw_values[..., self.tar_feas]

        if self.unknown_nodes_path:
            self.unknown_nodes = np.load(self.unknown_nodes_path).astype(int).tolist()
        elif self.unknown_nodes:
            self.unknown_nodes = [int(node) for node in self.unknown_nodes]
        else:
            self.unknown_nodes = select_master_unseen_nodes(
                self.full_adjacency,
                fraction=self.generated_unseen_fraction,
                seed=self.protocol_seed,
            ).astype(int).tolist()
        self.master_unseen_nodes = sorted({int(node) for node in self.unknown_nodes})
        self.known_nodes = np.setdiff1d(
            np.arange(self.full_num_nodes, dtype=np.int64),
            np.asarray(self.master_unseen_nodes, dtype=np.int64),
        ).astype(int).tolist()
        (
            validation_unseen,
            test_unseen,
            self.unseen_split_metadata,
        ) = split_unseen_nodes(
            self.master_unseen_nodes,
            self.full_adjacency,
            validation_ratio=self.unseen_validation_ratio,
            seed=self.protocol_seed,
        )
        self.validation_unseen_nodes = validation_unseen.astype(int).tolist()
        self.test_unseen_nodes = test_unseen.astype(int).tolist()
        self.boundaries = resolve_lcbt_boundaries(
            len(raw_values),
            train_end_ratio=self.split_ratios["train_end_ratio"],
            val_end_ratio=self.split_ratios["val_end_ratio"],
            test_start_ratio=self.split_ratios["test_start_ratio"],
            scaler_end_ratio=self.split_ratios["scaler_end_ratio"],
        )

        scaler_data = raw_values[
            self.boundaries["scaler_start"]:self.boundaries["scaler_end"],
            self.known_nodes,
        ]
        self.mean_ = np.nanmean(scaler_data, axis=(0, 1), keepdims=True)
        self.std_ = np.nanstd(scaler_data, axis=(0, 1), keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        if not self.scale:
            self.mean_ = np.zeros_like(self.mean_)
            self.std_ = np.ones_like(self.std_)

        if self.flag == "train":
            self.segment_start = self.boundaries["train_start"]
            self.segment_end = self.boundaries["train_end"]
        else:
            self.segment_start = self.boundaries["val_start"]
            self.segment_end = self.boundaries["val_end"]
        self.active_nodes = (
            list(self.known_nodes)
            if self.flag == "train"
            else list(range(self.full_num_nodes))
        )
        self.full_to_local = {
            int(node): int(index) for index, node in enumerate(self.active_nodes)
        }
        segment = raw_values[self.segment_start:self.segment_end, self.active_nodes]
        segment = (segment - self.mean_) / self.std_
        self.data = segment[..., self.in_feas]
        self.data_y = segment[..., self.tar_feas]
        self.num_nodes = int(self.data.shape[1])

        raw_time_mark = payload.get("time_stamp")
        self.time_mark = None
        if raw_time_mark is not None:
            transformed = time_features(raw_time_mark, freq=self.freq)
            self.time_mark = transformed[self.segment_start:self.segment_end]
        self.pos_mark = payload.get("pos_stamp")
        if self.pos_mark is not None:
            self.pos_mark = np.asarray(self.pos_mark)[self.active_nodes]
            denominator = np.max(self.pos_mark, keepdims=True) - np.min(self.pos_mark, keepdims=True)
            denominator[denominator == 0] = 1.0
            self.pos_mark = (self.pos_mark - np.min(self.pos_mark, keepdims=True)) / denominator - 0.5
        self.adj = self.full_adjacency[np.ix_(self.active_nodes, self.active_nodes)]
        self.calendar_regimes = _lcbt_window_regimes(
            self.full_timestamps,
            self.segment_start,
            self.segment_end,
            self.input_len,
            self.pred_len,
        )

    def _build_schedule(self, flag, seed, epoch):
        if flag == "train":
            start, end = self.boundaries["train_start"], self.boundaries["train_end"]
        else:
            start, end = self.boundaries["val_start"], self.boundaries["val_end"]
        regimes = _lcbt_window_regimes(
            self.full_timestamps,
            start,
            end,
            self.input_len,
            self.pred_len,
        )
        if flag == "val":
            return self._build_validation_records(seed)
        train_adjacency = self.full_adjacency[np.ix_(self.known_nodes, self.known_nodes)]
        return build_lcbt_schedule(
            len(regimes),
            self.input_len,
            train_adjacency,
            regimes,
            seed=seed,
            epoch=epoch,
            scenario_ratios=self.scenario_ratios,
            cohort_fractions=self.cohort_fractions,
            cohort_weights=self.cohort_weights,
            clustered_ratio=self.clustered_ratio,
            cohort_size_base=self.full_num_nodes,
            topology_adjacency=self.full_adjacency,
            # Training runs on the known-node induced graph, so its schedule
            # indices are local; this maps them back onto the full network.
            node_ids=self.known_nodes,
        )

    @property
    def validation_cohort_plan(self):
        pools = self._validation_node_pools()
        return main_table_cohort_plan(
            self.full_num_nodes,
            [len(nodes) for nodes in pools.values()],
            cohort_fractions=self.cohort_fractions,
            cohort_weights=self.cohort_weights,
        )

    def _validation_node_pools(self):
        return {
            "zero_history": list(self.validation_unseen_nodes),
            "steady": list(self.known_nodes),
            "join": list(self.validation_unseen_nodes),
            "fault": list(self.known_nodes),
        }

    def _build_validation_records(self, seed):
        origins = _lcbt_validation_origins(
            self.full_timestamps,
            self.boundaries["val_start"],
            self.boundaries["val_end"],
            self.input_len,
            self.pred_len,
            per_stratum=self.validation_origins_per_stratum,
            values=self.guard_values,
            min_observed_fraction=self.min_observed_fraction,
        )
        if not origins:
            raise ValueError("LCBT validation segment produced no usable origins")
        self.validation_origin_summary = summarize_origins(origins)
        self.validation_origins = strip_diagnostics(origins)
        origins = self.validation_origins
        regimes = [str(origin["stratum"]) for origin in origins]
        selection_schedule = build_lcbt_schedule(
            len(origins) * 4,
            self.input_len,
            self.full_adjacency,
            [regime for regime in regimes for _ in range(4)],
            seed=seed,
            epoch=0,
            scenario_ratios=(0.25, 0.25, 0.25, 0.25),
            # Same tiers the LCE main table pools, so the checkpoint metric and
            # the reported table never optimise different conditions.
            cohort_fractions=self.validation_cohort_plan["fractions"],
            cohort_weights=self.validation_cohort_plan["weights"],
            clustered_ratio=self.clustered_ratio,
            scenario_node_pools=self._validation_node_pools(),
            cohort_size_base=self.full_num_nodes,
            topology_adjacency=self.full_adjacency,
            # One cohort per origin is replayed across every k, so sizes are
            # balanced across origins in a stratum, not inside each k.
            stratify_by_evidence=False,
        )
        # The schedule balances cohort size and topology *within* a calendar
        # stratum, so a source item is only valid for an origin of the same
        # stratum. Filtering the schedule globally and zipping it against the
        # origins destroyed that: the k-th join item and the k-th origin sit in
        # different strata whenever the scenario draw is not block-aligned.
        by_stratum = {}
        for item in selection_schedule:
            scenario = str(item["scenario"])
            if scenario in {"join", "fault"}:
                by_stratum.setdefault((scenario, str(item["calendar_regime"])), []).append(item)
        pending = {key: list(items) for key, items in by_stratum.items()}

        def take(scenario, regime):
            queue = pending.get((scenario, regime))
            if not queue:
                raise RuntimeError(
                    f"{self.lifecycle_training_strategy} validation replay has no "
                    f"{scenario} cohort left for stratum {regime}"
                )
            return queue.pop(0)

        records = []
        for origin_rank, (origin, regime) in enumerate(zip(origins, regimes)):
            join_item = take("join", regime)
            fault_item = take("fault", regime)
            if (
                str(join_item["calendar_regime"]) != regime
                or str(fault_item["calendar_regime"]) != regime
            ):
                raise RuntimeError("validation replay paired a cohort from another stratum")
            calendar_id = (
                STRATA.index(regime) if regime in STRATA else len(STRATA)
            )

            def append_record(scenario, evidence, phase_id, group_id, source_item, pair):
                records.append(
                    {
                        "window_index": int(origin["window_index"]),
                        "scenario": scenario,
                        "evidence_steps": int(evidence),
                        "phase_id": int(phase_id),
                        "group_id": int(group_id),
                        "changed_nodes": list(source_item["changed_nodes"]),
                        "cohort_fraction": float(source_item["cohort_fraction"]),
                        "topology_mode": str(source_item["topology_mode"]),
                        "calendar_regime": str(regime),
                        "calendar_id": int(calendar_id),
                        "calendar_day": str(origin["calendar_day"]),
                        "matched_pair_id": f"val-{origin_rank:05d}-{pair}",
                    }
                )

            append_record("zero_history", 0, -1, 0, join_item, "join")
            append_record("steady", 0, -1, 1, fault_item, "fault")
            for evidence in range(1, self.input_len + 1):
                join_phase = 0 if evidence == 1 else 2 if evidence == self.input_len else 1
                fault_phase = 3 if evidence == 1 else 5 if evidence == self.input_len else 4
                append_record("join", evidence, join_phase, 2, join_item, "join")
                append_record("fault", evidence, fault_phase, 3, fault_item, "fault")
        return records

    def _manifest_payload(self):
        train_schedule = self._build_schedule("train", self.protocol_seed, 0)
        validation_records = self._build_schedule("val", self.validation_seed, 0)
        plan = self.validation_cohort_plan
        lce_semantics = {
            "protocol_name": "LCE v3",
            "validation_time_range": [
                self.boundaries["val_start"],
                self.boundaries["val_end"],
            ],
            "anchors": ["Unseen", "Steady"],
            "join_phases": ["Cold", "Warm", "Ready"],
            "fault_phases": ["Onset", "Decay", "Outage"],
            "aggregation": "LifeAvg=mean(Unseen,Steady,mean_k(Join),mean_k(Fault)); k=1..L",
            "strata_weighting": "equal over tod within day_type; day types by calendar share 5/7 and 2/7",
            "checkpoint_metric": "original-scale val_lifecycle_mae",
            "checkpoint_tie_break": "worst_group_mae over the four semantic groups",
        }
        ples_semantics = {
            "protocol_name": "PLES v3",
            "test_time_range": [
                self.boundaries["test_start"],
                self.boundaries["test_end"],
            ],
            "event_clock": "input_end=event_time+tau+1",
            "effects": ["JoinLift", "FaultPenalty", "OutageFaultPenalty"],
            "effect_metrics": ["MAPE", "MAE"],
            "bootstrap": "stratified_origin_then_cohort",
        }
        payload = {
            "schema_version": 3,
            "protocol_name": LCBT_PROTOCOL_NAME,
            "protocol_version": LCBT_PROTOCOL_VERSION,
            "dataset": self.dloader_name,
            "data_path": str(Path(self.data_path).resolve()),
            "data_shape": self.raw_data_shape,
            "data_hash": self.raw_data_hash,
            "timestamp_hash": self.timestamp_hash,
            "adjacency_hash": self.adjacency_hash,
            "boundaries": self.boundaries,
            "nodes": {
                "known": list(self.known_nodes),
                "master_unseen": list(self.master_unseen_nodes),
                "validation_unseen": list(self.validation_unseen_nodes),
                "test_unseen": list(self.test_unseen_nodes),
                "known_hash": stable_hash(self.known_nodes),
                "master_unseen_hash": stable_hash(self.master_unseen_nodes),
                "validation_unseen_hash": stable_hash(self.validation_unseen_nodes),
                "test_unseen_hash": stable_hash(self.test_unseen_nodes),
                "split": self.unseen_split_metadata,
            },
            "scaler": {
                "fit_start": self.boundaries["scaler_start"],
                "fit_end": self.boundaries["scaler_end"],
                "node_pool": "known_only",
                "mean_hash": array_hash(self.mean_),
                "std_hash": array_hash(self.std_),
            },
            "input_semantics": {
                "input_features": self.input_features,
                "target_features": self.target_features,
                "scale": self.scale,
                "effective_mask": "M_natural_and_M_lifecycle",
                "availability_context_channels": 4,
                "state_age": "event_duration_clipped_to_input_len",
                "visibility": "window_mean_of_effective_observation_mask",
                "observation_mask_feature": self.lifecycle_observation_mask_feature,
                "phase_labels_are_model_inputs": False,
                "grin_is_protocol_preprocessing": False,
            },
            "objective": {
                "forecast_mask": "target_cohort_mask",
                "auxiliary_mask": "target_cohort_mask_intersection",
                "scenario_ratios": dict(zip(LCBT_GROUPS, self.scenario_ratios)),
                "event_evidence_distribution": f"uniform_k=1..{self.input_len}",
            },
            "input_len": self.input_len,
            "pred_len": self.pred_len,
            "window_step": 1,
            "seeds": {
                "training": self.protocol_seed,
                "validation": self.validation_seed,
            },
            "calendar_strata": {
                "day_types": list(DAY_TYPES),
                "tod_bins": [[name, start, end] for name, start, end in TOD_BINS],
                "strata": list(STRATA),
                "weighting": "equal over tod within day_type; day types by calendar share 5/7 and 2/7",
                "weights": stratum_weights(),
            },
            "validation_origin_selection": {
                "rule": "stratified_low_discrepancy",
                "sequence": "golden_ratio_additive_recurrence",
                "stratum_order": "chronological",
                "per_stratum": self.validation_origins_per_stratum,
                "min_spacing": self.input_len + self.pred_len,
                "min_spacing_scope": "global; one replay window per origin",
                "min_observed_fraction": self.min_observed_fraction,
                "summary": self.validation_origin_summary,
            },
            "cohort_fractions": list(self.cohort_fractions),
            "cohort_weights": list(self.cohort_weights),
            "cohort_size_base": self.full_num_nodes,
            "validation_cohort_plan": {
                "fractions": list(plan["fractions"]),
                "weights": list(plan["weights"]),
                "realised_weights": plan["realised_weights"],
                "dropped_for_pool_saturation": plan["dropped"],
                "pool_saturation_limit": plan["pool_saturation_limit"],
            },
            "clustered_ratio": self.clustered_ratio,
            "reference_training_schedule": _lcbt_schedule_summary(
                train_schedule,
                len(self.known_nodes),
            ),
            "validation_replay_summary": _lcbt_schedule_summary(
                validation_records,
                self.full_num_nodes,
            ),
            "validation_replay": validation_records,
            "lce": lce_semantics,
            "ples": ples_semantics,
            "lce_protocol_hash": stable_hash(lce_semantics),
            "ples_protocol_hash": stable_hash(ples_semantics),
        }
        payload["protocol_hash"] = lcbt_manifest_hash(payload)
        payload["training_protocol_hash"] = payload["protocol_hash"]
        return payload

    def _ensure_protocol_manifest(self):
        manifest = self._manifest_payload()
        manifest_path = (
            self.manifest_root
            / self.dloader_name
            / self.lifecycle_training_strategy
            / "protocol_manifest.json"
        )
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("protocol_hash") != manifest["protocol_hash"] and not self.overwrite_manifest:
                raise ValueError(
                    f"LCBT manifest mismatch at {manifest_path}. "
                    "Use lcbt_overwrite_manifest=true only for an intentional protocol replacement."
                )
            if existing.get("protocol_hash") == manifest["protocol_hash"]:
                return existing
        _write_json_atomic(manifest_path, manifest)
        return manifest

    def _refresh_schedule(self):
        seed = self.protocol_seed if self.flag == "train" else self.validation_seed
        epoch = self.current_epoch if self.flag == "train" else 0
        self.schedule = self._build_schedule(self.flag, seed, epoch)

    def set_epoch(self, epoch):
        if self.flag == "train":
            self.current_epoch = int(epoch)
            self._refresh_schedule()

    def _lifecycle_state(self, item):
        role_ids = np.full((self.input_len, self.num_nodes), ROLE_AAS, dtype=np.int64)
        lifecycle_mask = np.ones((self.input_len, self.num_nodes), dtype=bool)
        state_age = np.ones((self.input_len, self.num_nodes), dtype=np.float32)
        if self.flag == "val":
            unseen = np.asarray(self.master_unseen_nodes, dtype=np.int64)
            role_ids[:, unseen] = ROLE_VS
            lifecycle_mask[:, unseen] = False
            state_age[:, unseen] = 0.0

        changed_nodes = list(item["changed_nodes"])
        scenario = str(item["scenario"])
        evidence_steps = int(item.get("evidence_steps", 0))
        if scenario == "zero_history":
            role_ids[:, changed_nodes] = ROLE_VS
            lifecycle_mask[:, changed_nodes] = False
            state_age[:, changed_nodes] = 0.0
        elif scenario == "steady":
            role_ids[:, changed_nodes] = ROLE_AAS
            lifecycle_mask[:, changed_nodes] = True
            state_age[:, changed_nodes] = 1.0
        elif scenario == "join":
            cutoff = self.input_len - evidence_steps
            role_ids[:, changed_nodes] = ROLE_VS
            lifecycle_mask[:, changed_nodes] = False
            state_age[:, changed_nodes] = 0.0
            role_ids[cutoff:, changed_nodes] = ROLE_NAS
            lifecycle_mask[cutoff:, changed_nodes] = True
            state_age[cutoff:, changed_nodes] = (
                np.arange(1, evidence_steps + 1, dtype=np.float32)[:, None]
                / float(self.input_len)
            )
        elif scenario == "fault":
            cutoff = self.input_len - evidence_steps
            role_ids[:, changed_nodes] = ROLE_AAS
            lifecycle_mask[:, changed_nodes] = True
            state_age[:, changed_nodes] = 1.0
            role_ids[cutoff:, changed_nodes] = ROLE_FS
            lifecycle_mask[cutoff:, changed_nodes] = False
            state_age[cutoff:, changed_nodes] = (
                np.arange(1, evidence_steps + 1, dtype=np.float32)[:, None]
                / float(self.input_len)
            )
        else:
            raise ValueError(
                f"unsupported {self.lifecycle_training_strategy} scenario: {scenario}"
            )
        return role_ids, lifecycle_mask, state_age

    def __getitem__(self, index):
        item = self.schedule[int(index)]
        input_begin = (
            int(item["window_index"])
            if self.flag == "val"
            else int(index)
        )
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        raw_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end]
        role_ids, lifecycle_mask, state_age = self._lifecycle_state(item)
        seq_x, _ = compose_effective_lifecycle_features(
            raw_x,
            lifecycle_mask,
            role_ids,
            append_role=self.lifecycle_role_features,
            append_age=self.lifecycle_history_age_feature,
            append_observation_mask=self.lifecycle_observation_mask_feature,
            state_age=state_age,
        )
        target_weights = np.ones_like(seq_y, dtype=np.float32)
        target_cohort_mask = np.zeros(self.num_nodes, dtype=np.float32)
        target_cohort_mask[item["changed_nodes"]] = 1.0
        group_id = int(
            item.get("group_id", LCBT_GROUPS.index(str(item["scenario"])))
        )
        calendar_id = int(
            item.get(
                "calendar_id",
                STRATA.index(item["calendar_regime"])
                if item["calendar_regime"] in STRATA
                else len(STRATA),
            )
        )
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        sample = (
            seq_x,
            seq_y,
            x_t_mark,
            y_t_mark,
            target_weights,
            np.int64(item["phase_id"]),
            np.int64(group_id),
            target_cohort_mask,
            np.int64(calendar_id),
        )
        return sample + (np.int64(item.get("evidence_steps", 0)),)

    def __len__(self):
        if self.flag == "val" and hasattr(self, "schedule"):
            return len(self.schedule)
        return (len(self.data) - self.input_len - self.pred_len) + 1

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        if self.flag == "val":
            return (
                list(self.known_nodes),
                list(self.validation_unseen_nodes),
                [],
                [],
            )
        return list(range(self.num_nodes)), [], [], []

    def get_protocol_manifest(self):
        return self.protocol_manifest


class AsyncLifecyclePEMS(Dataset):
    """Dataset-adaptive paired lifecycle event study test provider."""

    def __init__(
        self,
        data_path: str,
        size: list[int],
        flag: str,
        scale: bool,
        freq: str,
        target_features: str,
        input_features: str,
        dloader_name: str,
        splits: list,
        step: int = 1,
        inverse=False,
        unknown_nodes_path="",
        unknown_nodes_list=None,
        unknown_nodes_num=0,
        lifecycle_role_features: bool = True,
        lifecycle_history_age_feature: bool = True,
        lifecycle_observation_mask_feature: bool = True,
        lifecycle_age_mode: str = "state_duration",
        lcbt_manifest_root: str = "./protocol_manifests",
        lcbt_scaler_end_ratio=None,
        lcbt_test_start_ratio=None,
        lcbt_cohort_fractions=(0.025, 0.05, 0.10),
        lcbt_cohort_weights=(0.25, 0.50, 0.25),
        ples_pre_steps: int = 12,
        ples_post_steps: int = 36,
        ples_seed: int = 2026,
        ples_bootstrap_samples: int = 2000,
        ples_bootstrap_seed: int = 2026,
        ples_protocol_version: int = 3,
        lce_protocol_version: int = 3,
        ples_manifest_root: str = "./protocol_manifests",
        ples_overwrite_manifest: bool = False,
        lifecycle_training_strategy: str = "lcbt_v3",
        lce_origins_per_stratum: int = 1,
        lcbt_min_observed_fraction: float = 0.30,
    ):
        if flag != "test":
            raise ValueError("AsyncLifecyclePEMS only supports the test split")
        fea_dict = {
            "q": [0],
            "o": [1],
            "v": [2],
            "qv": [0, 2],
            "qov": [0, 1, 2],
        }
        self.tar_feas = fea_dict[target_features]
        self.in_feas = fea_dict[input_features]
        self.target_features = target_features
        self.input_features = input_features
        self.raw_enc_in = len(self.in_feas)
        self.flag = flag
        self.input_len = int(size[0])
        self.look_back = int(size[1])
        if self.look_back > self.input_len:
            raise ValueError("look_back should be smaller than input_len")
        self.pred_len = int(size[2])
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.data_path = data_path
        self.train_unknown_nodes = list(unknown_nodes_list or [])
        self.unseen_nodes_were_explicit = bool(unknown_nodes_path) or bool(
            unknown_nodes_list
        )
        self.unknown_nodes_num = unknown_nodes_num
        self.unknown_nodes_path = unknown_nodes_path
        self.dloader_name = dloader_name
        self.step = int(step)
        self.train_ratio, self.valid_ratio = splits
        self.lifecycle_role_features = lifecycle_role_features
        self.lifecycle_history_age_feature = lifecycle_history_age_feature
        self.lifecycle_observation_mask_feature = lifecycle_observation_mask_feature
        self.lifecycle_age_mode = str(lifecycle_age_mode)
        self.lifecycle_training_strategy = str(lifecycle_training_strategy)
        self.lcbt_manifest_root = lcbt_manifest_root
        self.lcbt_scaler_end_ratio = lcbt_scaler_end_ratio
        self.lcbt_test_start_ratio = lcbt_test_start_ratio
        self.lcbt_cohort_fractions = tuple(float(value) for value in lcbt_cohort_fractions)
        self.lcbt_cohort_weights = tuple(float(value) for value in lcbt_cohort_weights)
        self.ples_protocol_version = int(ples_protocol_version)
        self.lce_protocol_version = int(lce_protocol_version)
        self.origins_per_stratum = int(lce_origins_per_stratum)
        self.min_observed_fraction = float(lcbt_min_observed_fraction)
        self.ples_manifest_root = Path(ples_manifest_root)
        self.ples_overwrite_manifest = bool(ples_overwrite_manifest)
        if self.lifecycle_training_strategy != "lcbt_v3":
            raise ValueError("async evaluation requires lcbt_v3")
        if self.lifecycle_age_mode != "state_duration":
            raise ValueError("lifecycle async evaluation requires state_duration age")
        if (
            self.ples_protocol_version != 3
            or self.lce_protocol_version != 3
        ):
            raise ValueError(
                "lcbt_v3 async evaluation requires LCE v3 and PLES v3"
            )
        self.training_protocol_hash = None
        self.ples_options = {
            "pre_steps": int(ples_pre_steps),
            "post_steps": int(ples_post_steps),
            "seed": int(ples_seed),
            "bootstrap_samples": int(ples_bootstrap_samples),
            "bootstrap_seed": int(ples_bootstrap_seed),
        }
        self.__read_data__()

    def __read_data__(self):
        raw_data = _unwrap_raw_data(np.load(self.data_path, allow_pickle=True))
        self.time_stamp_raw = raw_data.get("time_stamp")
        full_time_mark = (
            time_features(self.time_stamp_raw, freq=self.freq)
            if self.time_stamp_raw is not None
            else None
        )

        self.pos_mark = raw_data.get("pos_stamp")
        if self.pos_mark is not None:
            denominator = np.max(self.pos_mark, keepdims=True) - np.min(self.pos_mark, keepdims=True)
            denominator = np.where(denominator == 0, 1.0, denominator)
            self.pos_mark = (self.pos_mark - np.min(self.pos_mark, keepdims=True)) / denominator - 0.5

        self.adj = raw_data["adj"]
        raw_source = np.asarray(raw_data["data"])
        raw_series_hash = array_hash(raw_source)
        raw_series = raw_source.astype(np.float32)
        self.full_num_nodes = raw_series.shape[1]
        self.num_nodes = raw_series.shape[1]
        raw_series[raw_series <= 0] = np.nan

        if len(self.unknown_nodes_path):
            self.train_unknown_nodes = np.load(self.unknown_nodes_path)
        elif len(self.train_unknown_nodes):
            self.train_unknown_nodes = np.asarray(self.train_unknown_nodes)
        else:
            rng = np.random.default_rng(0)
            self.train_unknown_nodes = rng.choice(
                range(raw_series.shape[1]),
                self.unknown_nodes_num,
                replace=False,
            )
        self.train_known_nodes = np.setdiff1d(range(raw_series.shape[1]), self.train_unknown_nodes)
        self.train_unknown_nodes = sorted(np.asarray(self.train_unknown_nodes, dtype=int).tolist())
        self.train_known_nodes = sorted(np.asarray(self.train_known_nodes, dtype=int).tolist())
        self.validation_unseen_nodes = self.train_unknown_nodes.copy()
        self.test_unseen_nodes = self.train_unknown_nodes.copy()

        if self.lcbt_scaler_end_ratio is None or self.lcbt_test_start_ratio is None:
            raise ValueError("LCE/PLES evaluation requires explicit LCBT split ratios")
        scaler_cutoff = int(float(self.lcbt_scaler_end_ratio) * len(raw_series))
        test_cutoff = int(float(self.lcbt_test_start_ratio) * len(raw_series))
        manifest_path = (
            Path(self.lcbt_manifest_root)
            / self.dloader_name
            / self.lifecycle_training_strategy
            / "protocol_manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"LCBT training manifest not found: {manifest_path}. "
                "Build the train/validation loaders before running test_async."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            training_manifest = json.load(handle)
        expected = training_manifest.get("boundaries", {})
        if expected.get("scaler_end") != scaler_cutoff or expected.get("test_start") != test_cutoff:
            raise ValueError("LCBT train manifest split boundaries do not match async-test settings")
        if training_manifest.get("data_hash") != raw_series_hash:
            raise ValueError("LCBT train manifest data fingerprint does not match async-test data")
        if training_manifest.get("adjacency_hash") != array_hash(self.adj):
            raise ValueError("LCBT train manifest adjacency fingerprint does not match async-test graph")
        self.training_protocol_hash = training_manifest.get("protocol_hash")
        node_manifest = training_manifest.get("nodes", {})
        manifest_known = sorted(int(node) for node in node_manifest.get("known", []))
        manifest_master = sorted(
            int(node) for node in node_manifest.get("master_unseen", [])
        )
        manifest_validation = sorted(
            int(node) for node in node_manifest.get("validation_unseen", [])
        )
        manifest_test = sorted(
            int(node) for node in node_manifest.get("test_unseen", [])
        )
        if not manifest_known or not manifest_master:
            raise ValueError("LCBT training manifest is missing frozen node pools")
        if (
            self.unseen_nodes_were_explicit
            and sorted(self.train_unknown_nodes) != manifest_master
        ):
            raise ValueError(
                "Configured unseen nodes do not match the frozen LCBT master unseen pool"
            )
        self.train_known_nodes = manifest_known
        self.train_unknown_nodes = manifest_master
        self.validation_unseen_nodes = manifest_validation
        self.test_unseen_nodes = manifest_test

        self.initial_aas_nodes = self.train_known_nodes.copy()
        self.initial_vs_nodes = self.train_unknown_nodes.copy()

        train_data = raw_series[:scaler_cutoff][:, self.train_known_nodes]
        test_data = raw_series[test_cutoff:]
        self.time_mark = (
            full_time_mark[test_cutoff:]
            if full_time_mark is not None
            else None
        )
        self.time_stamp_raw = (
            np.asarray(self.time_stamp_raw)[test_cutoff:]
            if self.time_stamp_raw is not None
            else None
        )

        if self.scale:
            self.mean_ = np.nanmean(train_data, axis=(0, 1), keepdims=True)
            self.std_ = np.nanstd(train_data, axis=(0, 1), keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        else:
            self.mean_ = np.zeros((1, 1, train_data.shape[-1]), dtype=np.float32)
            self.std_ = np.ones((1, 1, train_data.shape[-1]), dtype=np.float32)

        scaled_test = (test_data - self.mean_) / self.std_
        self.data = scaled_test[..., self.in_feas]
        self.data_y = scaled_test[..., self.tar_feas]
        self.protocol_manifest = build_protocol_manifest(
            dataset_name=self.dloader_name,
            timestamps=self.time_stamp_raw,
            adjacency=self.adj,
            known_nodes=self.train_known_nodes,
            master_unseen_nodes=self.train_unknown_nodes,
            validation_unseen_nodes=self.validation_unseen_nodes,
            test_unseen_nodes=self.test_unseen_nodes,
            test_values=test_data[..., self.tar_feas],
            input_features=self.input_features,
            target_features=self.target_features,
            input_len=self.input_len,
            pred_len=self.pred_len,
            training_protocol_hash=self.training_protocol_hash,
            pre_steps=self.ples_options["pre_steps"],
            post_steps=self.ples_options["post_steps"],
            cohort_fractions=self.lcbt_cohort_fractions,
            cohort_weights=self.lcbt_cohort_weights,
            seed=self.ples_options["seed"],
            bootstrap_samples=self.ples_options["bootstrap_samples"],
            bootstrap_seed=self.ples_options["bootstrap_seed"],
            lce_protocol_version=3,
            origins_per_stratum=self.origins_per_stratum,
            min_observed_fraction=self.min_observed_fraction,
        )
        self.protocol_manifest = self._ensure_evaluation_manifest(
            self.protocol_manifest
        )

    def _ensure_evaluation_manifest(self, manifest):
        manifest_path = (
            self.ples_manifest_root
            / self.dloader_name
            / "lifecycle_v3"
            / "protocol_manifest.json"
        )
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("protocol_hash") == manifest.get("protocol_hash"):
                return existing
            if not self.ples_overwrite_manifest:
                raise ValueError(
                    f"Lifecycle v3 manifest mismatch at {manifest_path}. "
                    "Use ples_overwrite_manifest=true only for an intentional protocol replacement."
                )
        _write_json_atomic(manifest_path, manifest)
        return manifest

    def _base_visibility_and_roles(self):
        visibility = np.zeros((self.input_len, self.num_nodes), dtype=bool)
        visibility[:, self.initial_aas_nodes] = True
        role_ids = np.full((self.input_len, self.num_nodes), ROLE_AAS, dtype=np.int64)
        role_ids[:, self.initial_vs_nodes] = ROLE_VS
        state_age = np.ones((self.input_len, self.num_nodes), dtype=np.float32)
        state_age[:, self.initial_vs_nodes] = 0.0
        return visibility, role_ids, state_age

    def _time_marks(self, input_begin, input_end, output_begin, output_end):
        if self.time_mark is None:
            return (
                np.zeros((input_end - input_begin, 0), dtype=np.float32),
                np.zeros((output_end - output_begin, 0), dtype=np.float32),
            )
        return self.time_mark[input_begin:input_end], self.time_mark[output_begin:output_end]

    def _compose_input(self, raw_x, visibility, role_ids, state_age=None):
        seq_x, _ = compose_effective_lifecycle_features(
            raw_x,
            visibility,
            role_ids,
            append_role=self.lifecycle_role_features,
            append_age=self.lifecycle_history_age_feature,
            append_observation_mask=self.lifecycle_observation_mask_feature,
            state_age=state_age,
        )
        return seq_x

    def build_ples_sample(self, origin, cohort, track, arm, tau):
        if track not in {"join", "fault"}:
            raise ValueError(f"Unsupported PLES track: {track}")
        if arm not in {"control", "treatment"}:
            raise ValueError(f"Unsupported PLES arm: {arm}")

        event_time = int(origin["event_time_index"])
        input_end = event_time + int(tau) + 1
        input_begin = input_end - self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        if input_begin < 0 or output_end > len(self.data):
            raise IndexError("PLES sample exceeds the dataset timeline")

        raw_x = self.data[input_begin:input_end].copy()
        seq_y = self.data_y[output_begin:output_end].copy()
        visibility, role_ids, state_age = self._base_visibility_and_roles()
        if arm == "treatment":
            post_rows = np.flatnonzero(np.arange(input_begin, input_end) >= event_time)
            event_duration = np.clip(
                np.arange(input_begin, input_end) - event_time + 1,
                0,
                self.input_len,
            ).astype(np.float32) / float(self.input_len)
            if track == "join" and len(post_rows):
                nodes = np.asarray(cohort["join_nodes"], dtype=np.int64)
                visibility[np.ix_(post_rows, nodes)] = True
                role_ids[np.ix_(post_rows, nodes)] = ROLE_NAS
                state_age[np.ix_(post_rows, nodes)] = event_duration[post_rows, None]
            elif track == "fault" and len(post_rows):
                nodes = np.asarray(cohort["fault_nodes"], dtype=np.int64)
                visibility[np.ix_(post_rows, nodes)] = False
                role_ids[np.ix_(post_rows, nodes)] = ROLE_FS
                state_age[np.ix_(post_rows, nodes)] = event_duration[post_rows, None]

        seq_x = self._compose_input(raw_x, visibility, role_ids, state_age)
        x_t_mark, y_t_mark = self._time_marks(input_begin, input_end, output_begin, output_end)
        return seq_x, seq_y, x_t_mark, y_t_mark

    def get_protocol_manifest(self):
        return self.protocol_manifest

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.initial_aas_nodes, self.initial_vs_nodes, [], []

    def __getitem__(self, index):
        input_begin = int(index) * self.step
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        raw_x = self.data[input_begin:input_end].copy()
        visibility, role_ids, state_age = self._base_visibility_and_roles()
        seq_x = self._compose_input(raw_x, visibility, role_ids, state_age)
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = self._time_marks(input_begin, input_end, output_begin, output_end)
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len - self.pred_len) // self.step + 1

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]
