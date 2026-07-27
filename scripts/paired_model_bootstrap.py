"""Offline paired comparison of two frozen LCE v3 / PLES v3 result bundles.

Stage 0 of the protocol audit: read-only. This script never retrains, never
re-runs inference, and never touches a config, manifest or checkpoint. It reads
the frozen ``ples_paired_rows.csv`` of two models that were evaluated under the
same frozen manifest and answers a single question the current reports cannot:
is the gap between the two models larger than origin/cohort resampling noise?

The existing reports give each model its own interval. Two independent intervals
are not a test of a difference. Because both models are evaluated on exactly the
same origins, cohorts, tracks and taus, the correct estimator applies one shared
resample to both models and takes the difference inside each draw.

Aggregation is not re-derived here. The vectorised bootstrap kernel is checked
against ``data_provider.ples_protocol`` on the full sample before any draw is
taken, so the reported point estimates are the production numbers by
construction.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.calendar_strata import stratum_weights  # noqa: E402
from data_provider.ples_protocol import (  # noqa: E402
    PHASE_ORDER,
    PLES_EFFECT_NAMES,
    overlap_blocks,
    summarize_lce_rows,
    summarize_paired_rows,
    trajectory_span_from_manifest,
)
from scripts.collect_lifecycle_results import (  # noqa: E402
    HASH_FIELDS,
    MAIN_ORDER,
    METRICS,
    _load_run,
)

# Fields that the frozen protocol determines. They are model independent, so a
# mismatch means the two bundles did not come from the same manifest.
PROTOCOL_ROW_FIELDS = (
    "dataset",
    "calendar_day",
    "regime",
    "event_time_index",
    "event_timestamp",
    "cohort_fraction",
    "cohort_style",
    "phase",
    "node_count",
    "count",
)
PAIR_KEY_FIELDS = ("origin_id", "cohort_id", "track", "tau")
SUM_FIELDS = (
    "treated_ape_sum",
    "control_ape_sum",
    "treated_ae_sum",
    "control_ae_sum",
    "treated_se_sum",
    "control_se_sum",
)
TRACKS = ("join", "fault")
# LifeAvg is a deterministic mean of the four semantic groups, so the primary
# endpoint family is the four groups on the checkpoint metric, not all 33 cells.
PRIMARY_FAMILY = (("Unseen", "MAE"), ("Steady", "MAE"), ("Join", "MAE"), ("Fault", "MAE"))
LOWER_IS_BETTER_EFFECTS = {"FaultPenalty", "OutageFaultPenalty"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired origin->cohort bootstrap between two frozen LCE/PLES v3 result bundles.",
    )
    parser.add_argument("--model_a", required=True, help="Result directory of the model under test.")
    parser.add_argument("--model_b", required=True, help="Result directory of the reference model.")
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="./reports/paired_bootstrap")
    parser.add_argument(
        "--recompute_tolerance",
        type=float,
        default=1e-9,
        help="Relative tolerance when replaying the frozen summaries from the row file.",
    )
    return parser.parse_args()


def _read_rows(result_dir):
    path = Path(result_dir) / "ples_paired_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"PLES row file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("origin_id", "cohort_id", "tau", "count", "node_count", "event_time_index"):
                row[key] = int(row[key])
            for key in SUM_FIELDS + ("cohort_fraction",):
                row[key] = float(row[key])
            # The CSV round-trip stores these as text, but the production
            # aggregation expects float-or-None.
            for key in ("graph_weight", "patch_residual_weight"):
                row[key] = None if row.get(key) in {None, ""} else float(row[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"PLES paired rows are empty: {path}")
    return rows, path


def _trajectory_span(result_dir, input_len):
    """PLES trajectory length, read from the frozen evaluation manifest.

    The manifest records the span the single-model bootstrap used; reading it
    back is what keeps the paired bootstrap on the same resampling unit instead
    of quietly drifting to a different one. ``input_len`` is only a fallback for
    manifests that predate ``resolved.input_len``; the horizon always comes from
    ``resolved.pred_len``.
    """
    path = Path(result_dir) / "protocol_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    span = trajectory_span_from_manifest(manifest, default_input_len=input_len)
    if not span:
        raise ValueError(f"manifest declares no PLES trajectory span: {path}")
    return int(span)


def _main_cohort_ids(result_dir):
    """Non-saturated cohorts, read from the frozen manifest beside the rows."""
    path = Path(result_dir) / "protocol_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    ids = manifest.get("resolved", {}).get("main_cohort_ids")
    if not ids:
        raise ValueError(f"manifest has no main_cohort_ids: {path}")
    return [int(value) for value in ids]


def _pair_keys(rows):
    return [
        (int(row["origin_id"]), int(row["cohort_id"]), str(row["track"]), int(row["tau"]))
        for row in rows
    ]


def assert_paired_bundles(run_a, run_b, rows_a, rows_b):
    """Fail loudly on any pairing mismatch. Never silently intersect."""
    problems = []

    for field in HASH_FIELDS:
        if run_a["hashes"][field] != run_b["hashes"][field]:
            problems.append(
                f"{field} differs: {run_a['model']}={run_a['hashes'][field]} "
                f"{run_b['model']}={run_b['hashes'][field]}"
            )
    if run_a["dataset"] != run_b["dataset"]:
        problems.append("dataset record differs between the two bundles")
    if run_a["seed"] != run_b["seed"]:
        problems.append(f"seed differs: {run_a['seed']} vs {run_b['seed']}")
    if len(rows_a) != len(rows_b):
        problems.append(f"row count differs: {len(rows_a)} vs {len(rows_b)}")

    keys_a, keys_b = _pair_keys(rows_a), _pair_keys(rows_b)
    for label, keys in ((run_a["model"], keys_a), (run_b["model"], keys_b)):
        if len(set(keys)) != len(keys):
            problems.append(f"{label} contains duplicated pair keys")
    if set(keys_a) != set(keys_b):
        only_a, only_b = set(keys_a) - set(keys_b), set(keys_b) - set(keys_a)
        problems.append(
            f"pair-key sets differ: {len(only_a)} only in {run_a['model']}, "
            f"{len(only_b)} only in {run_b['model']}; refusing to intersect"
        )

    if not problems:
        index_b = {key: row for key, row in zip(keys_b, rows_b)}
        for key, row_a in zip(keys_a, rows_a):
            row_b = index_b[key]
            for field in PROTOCOL_ROW_FIELDS:
                if row_a[field] != row_b[field]:
                    problems.append(
                        f"protocol-determined field {field} differs at {key}: "
                        f"{row_a[field]} vs {row_b[field]}"
                    )
                    break
            if problems:
                break

    if problems:
        raise ValueError(
            "The two result bundles are not a valid pair:\n  - " + "\n  - ".join(problems)
        )
    return keys_a


def assert_pre_event_identity(rows, model):
    """PreEventCheck as a consistency assertion, not a reported score."""
    checked = 0
    for row in rows:
        if int(row["tau"]) >= 0:
            continue
        checked += 1
        for prefix in ("ape", "ae", "se"):
            treated, control = row[f"treated_{prefix}_sum"], row[f"control_{prefix}_sum"]
            if treated != control:
                raise ValueError(
                    f"{model}: pre-event treated/control diverge at "
                    f"origin={row['origin_id']} cohort={row['cohort_id']} "
                    f"track={row['track']} tau={row['tau']} ({prefix}: {treated} vs {control})"
                )
    if checked == 0:
        raise ValueError(f"{model}: no pre-event rows found; PreEventCheck cannot be asserted")
    return checked


class RowTensor:
    """Row-level metrics plus the (track, tau, regime) cell map used by every statistic."""

    def __init__(self, rows, keys, input_len, trajectory_span=None):
        self.input_len = int(input_len)
        self.trajectory_span = None if trajectory_span is None else int(trajectory_span)
        counts = np.array([row["count"] for row in rows], dtype=np.float64)
        if np.any(counts <= 0):
            raise ValueError("rows with non-positive count must be absent from the frozen CSV")

        self.metrics = {}
        for arm in ("treated", "control"):
            self.metrics[(arm, "MAPE")] = (
                np.array([row[f"{arm}_ape_sum"] for row in rows]) / counts * 100.0
            )
            self.metrics[(arm, "MAE")] = np.array([row[f"{arm}_ae_sum"] for row in rows]) / counts
            self.metrics[(arm, "RMSE")] = np.sqrt(
                np.array([row[f"{arm}_se_sum"] for row in rows]) / counts
            )

        regimes = sorted({str(row["regime"]) for row in rows})
        self.regimes = regimes
        regime_slot = {name: index for index, name in enumerate(regimes)}
        weights_by_name = stratum_weights()
        missing = [name for name in regimes if name not in weights_by_name]
        if missing:
            raise ValueError(f"rows carry strata outside the protocol: {missing}")
        self.stratum_weight = np.array(
            [weights_by_name[name] for name in regimes], dtype=np.float64
        )
        taus = sorted({int(row["tau"]) for row in rows})
        self.taus = taus
        tau_slot = {value: index for index, value in enumerate(taus)}
        track_slot = {name: index for index, name in enumerate(TRACKS)}

        self.n_track, self.n_tau, self.n_regime = len(TRACKS), len(taus), len(regimes)
        self.cell_id = np.array(
            [
                (track_slot[str(row["track"])] * self.n_tau + tau_slot[int(row["tau"])])
                * self.n_regime
                + regime_slot[str(row["regime"])]
                for row in rows
            ],
            dtype=np.int64,
        )
        self.n_cells = self.n_track * self.n_tau * self.n_regime
        self.tau_slot = tau_slot
        self.track_slot = track_slot

        # (origin, cohort) resampling units, ragged because empty cells are dropped.
        units = defaultdict(list)
        for index, (origin_id, cohort_id, _track, _tau) in enumerate(keys):
            units[(origin_id, cohort_id)].append(index)
        self.origin_ids = sorted({origin for origin, _ in units})
        # Neighbouring origins share PLES trajectory data, so they are not
        # independent. Calendar days do not contain that dependence -- 24 of the
        # 175 adjacent origin pairs overlap across midnight -- so origins whose
        # pre+input+post+horizon spans touch are chained into one block instead.
        # Blocks are stratified by day type so a draw cannot change the
        # Weekday/Weekend mix. Cohorts are crossed with origins rather than
        # nested in them, so they get one global draw per sample. This must
        # mirror ples_protocol.bootstrap_units exactly, and the evaluation
        # manifest names this script as the paired implementation of it.
        origin_times = {}
        for row in rows:
            origin_times[int(row["origin_id"])] = int(
                row.get("event_time_index", row["origin_id"])
            )
        if self.trajectory_span:
            assignment = overlap_blocks(origin_times, self.trajectory_span)
            label = {origin: f"overlap-{index:04d}" for origin, index in assignment.items()}
        else:  # no span available: fall back to calendar days
            label = {
                int(row["origin_id"]): str(row.get("calendar_day", "all")) for row in rows
            }
        block_rows = defaultdict(list)
        block_stratum = {}
        for index, row in enumerate(rows):
            block = label[int(row["origin_id"])]
            block_rows[block].append(index)
            block_stratum.setdefault(block, str(row.get("regime", "all")).split("/")[0])
        self.block_rows = {name: np.array(items, dtype=np.int64)
                           for name, items in sorted(block_rows.items())}
        by_stratum = defaultdict(list)
        for block, stratum in sorted(block_stratum.items()):
            by_stratum[stratum].append(block)
        self.blocks_by_stratum = {name: sorted(items) for name, items in sorted(by_stratum.items())}
        self.cohort_ids = sorted({int(row["cohort_id"]) for row in rows})
        self.row_cohort = np.array([int(row["cohort_id"]) for row in rows], dtype=np.int64)
        self.n_rows = len(rows)

    def calendar_balanced(self, arm, metric, weights):
        """Row mean inside each stratum, then the production stratum weighting.

        This must match ``weighted_stratum_mean``: time-of-day equal inside a day
        type, day types by calendar share. Averaging the strata equally here
        would silently disagree with the reported tables.
        """
        values = self.metrics[(arm, metric)]
        totals = np.bincount(self.cell_id, weights=weights * values, minlength=self.n_cells)
        counts = np.bincount(self.cell_id, weights=weights, minlength=self.n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_mean = np.where(counts > 0, totals / np.maximum(counts, 1e-12), np.nan)
        shape = (self.n_track, self.n_tau, self.n_regime)
        cell_mean = np.where(counts > 0, cell_mean, np.nan).reshape(shape)
        stratum_weight = self.stratum_weight.reshape(1, 1, self.n_regime)
        present = np.isfinite(cell_mean)
        weight = np.where(present, stratum_weight, 0.0)
        total_weight = weight.sum(axis=2)
        weighted = np.where(present, cell_mean, 0.0) * weight
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total_weight > 0, weighted.sum(axis=2) / total_weight, np.nan)


def _tau_indices(tensor, tau_values):
    return [tensor.tau_slot[value] for value in tau_values if value in tensor.tau_slot]


def _nanmean(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def compute_statistics(tensor, weights):
    """All reported quantities from one weight vector over rows."""
    input_len = tensor.input_len
    lce_taus = list(range(input_len))
    phase_taus = {
        "Cold": ("join", [0]),
        "Warm": ("join", list(range(1, input_len - 1))),
        "Ready": ("join", [input_len - 1]),
        "Onset": ("fault", [0]),
        "Decay": ("fault", list(range(1, input_len - 1))),
        "Outage": ("fault", [input_len - 1]),
    }
    output = {}
    for metric in METRICS:
        treated = tensor.calendar_balanced("treated", metric, weights)
        control = tensor.calendar_balanced("control", metric, weights)
        join_slot, fault_slot = tensor.track_slot["join"], tensor.track_slot["fault"]
        lce_slots = _tau_indices(tensor, lce_taus)

        groups = {
            "Unseen": _nanmean(control[join_slot, lce_slots]),
            "Steady": _nanmean(control[fault_slot, lce_slots]),
            "Join": _nanmean(treated[join_slot, lce_slots]),
            "Fault": _nanmean(treated[fault_slot, lce_slots]),
        }
        groups["LifeAvg"] = _nanmean(list(groups.values()))
        groups["WorstGroup"] = float(
            np.nanmax([groups[name] for name in ("Unseen", "Steady", "Join", "Fault")])
        )
        for name, value in groups.items():
            output[("main", name, metric)] = value
        for phase, (track, taus) in phase_taus.items():
            slots = _tau_indices(tensor, taus)
            output[("phase", phase, metric)] = _nanmean(treated[tensor.track_slot[track], slots])
        for label, track in (("Join", "join"), ("Fault", "fault")):
            for k in range(1, input_len + 1):
                slot = tensor.tau_slot.get(k - 1)
                output[("trajectory", f"{label}_k{k}", metric)] = (
                    float(treated[tensor.track_slot[track], slot]) if slot is not None else float("nan")
                )

    # PLES effects run over every post-event tau, on both reported metrics.
    post_slots = [slot for value, slot in tensor.tau_slot.items() if value >= 0]
    outage_slot = tensor.tau_slot.get(input_len - 1)
    for metric in ("MAPE", "MAE"):
        treated = tensor.calendar_balanced("treated", metric, weights)
        control = tensor.calendar_balanced("control", metric, weights)
        join_lift = (
            control[tensor.track_slot["join"], post_slots]
            - treated[tensor.track_slot["join"], post_slots]
        )
        fault_penalty = (
            treated[tensor.track_slot["fault"], post_slots]
            - control[tensor.track_slot["fault"], post_slots]
        )
        output[("ples", "JoinLift", metric)] = _nanmean(join_lift)
        output[("ples", "FaultPenalty", metric)] = _nanmean(fault_penalty)
        output[("ples", "OutageFaultPenalty", metric)] = (
            float(fault_penalty[post_slots.index(outage_slot)])
            if outage_slot in post_slots
            else float("nan")
        )
    return output


def verify_against_production(tensor, rows, run, tolerance, main_cohort_ids=None):
    """Replay the frozen summaries through the production aggregation code."""
    weights = np.ones(tensor.n_rows, dtype=np.float64)
    fast = compute_statistics(tensor, weights)
    lce = summarize_lce_rows(rows, input_len=tensor.input_len, main_cohort_ids=main_cohort_ids)
    ples = summarize_paired_rows(
        rows, input_len=tensor.input_len, main_cohort_ids=main_cohort_ids
    )["paired_effects"]

    mismatches = []

    def _check(label, got, expected):
        if not np.isclose(got, expected, rtol=tolerance, atol=tolerance):
            mismatches.append(f"{label}: vectorised={got!r} production={expected!r}")

    for name in MAIN_ORDER:
        for metric in METRICS:
            _check(f"main/{name}/{metric}", fast[("main", name, metric)], lce["main_metrics"][name][metric])
            _check(
                f"frozen-main/{name}/{metric}",
                lce["main_metrics"][name][metric],
                run["main"][name][metric],
            )
    for phase in PHASE_ORDER:
        for metric in METRICS:
            _check(f"phase/{phase}/{metric}", fast[("phase", phase, metric)], lce["phase_metrics"][phase][metric])
            _check(
                f"frozen-phase/{phase}/{metric}",
                lce["phase_metrics"][phase][metric],
                run["phases"][phase][metric],
            )
    for label in ("Join", "Fault"):
        for cell in lce["trajectory_metrics"][label]:
            for metric in METRICS:
                _check(
                    f"trajectory/{label}_k{cell['k']}/{metric}",
                    fast[("trajectory", f"{label}_k{cell['k']}", metric)],
                    cell[metric],
                )
    for metric in ("MAPE", "MAE"):
        for name in PLES_EFFECT_NAMES:
            _check(f"ples/{name}/{metric}", fast[("ples", name, metric)], ples[metric][name])

    if mismatches:
        raise ValueError(
            f"{run['model']}: recomputation disagrees with the frozen report:\n  - "
            + "\n  - ".join(mismatches)
        )
    return fast


def draw_weights(tensor, rng):
    """One stratified overlap-block draw crossed with one cohort draw."""
    cohort_multiplicity = np.zeros(int(max(tensor.cohort_ids)) + 1, dtype=np.float64)
    for cohort_id in rng.choice(tensor.cohort_ids, size=len(tensor.cohort_ids), replace=True):
        cohort_multiplicity[int(cohort_id)] += 1.0

    block_multiplicity = defaultdict(float)
    for _stratum, blocks in tensor.blocks_by_stratum.items():
        for block in rng.choice(blocks, size=len(blocks), replace=True):
            block_multiplicity[str(block)] += 1.0

    weights = np.zeros(tensor.n_rows, dtype=np.float64)
    for block, repeats in block_multiplicity.items():
        indices = tensor.block_rows[block]
        weights[indices] += repeats * cohort_multiplicity[tensor.row_cohort[indices]]
    return weights


def paired_bootstrap(tensor_a, tensor_b, samples, seed):
    """One shared resample per draw, applied to both models."""
    rng = np.random.default_rng(seed)
    draws = defaultdict(lambda: np.empty(samples, dtype=np.float64))
    for index in range(samples):
        weights = draw_weights(tensor_a, rng)
        stats_a = compute_statistics(tensor_a, weights)
        stats_b = compute_statistics(tensor_b, weights)
        for key, value in stats_a.items():
            draws[key][index] = value - stats_b[key]
    return dict(draws)


def _bootstrap_p_value(delta_draws):
    finite = delta_draws[np.isfinite(delta_draws)]
    if not len(finite):
        return float("nan")
    below = float(np.mean(finite <= 0.0))
    above = float(np.mean(finite >= 0.0))
    return float(min(1.0, max(2.0 * min(below, above), 1.0 / (len(finite) + 1.0))))


def _holm(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def classify(delta, low, high, lower_is_better):
    if not np.isfinite(delta):
        return "无法判定"
    better = delta < 0 if lower_is_better else delta > 0
    significant = np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
    if better:
        return "显著胜出" if significant else "仅方向胜出"
    return "显著落后" if significant else "仅方向落后"


def build_table(stats_a, stats_b, draws, model_a, model_b):
    records = []
    for key, delta_draws in draws.items():
        table, name, metric = key
        delta = stats_a[key] - stats_b[key]
        low, high = (
            (float(np.nanpercentile(delta_draws, 2.5)), float(np.nanpercentile(delta_draws, 97.5)))
            if np.any(np.isfinite(delta_draws))
            else (float("nan"), float("nan"))
        )
        lower_is_better = table != "ples" or name in LOWER_IS_BETTER_EFFECTS
        records.append(
            {
                "table": table,
                "row": name,
                "metric": metric,
                f"{model_a}": stats_a[key],
                f"{model_b}": stats_b[key],
                "delta": delta,
                "ci_low": low,
                "ci_high": high,
                "p_bootstrap": _bootstrap_p_value(delta_draws),
                "lower_is_better": lower_is_better,
                "verdict": classify(delta, low, high, lower_is_better),
            }
        )
    return records


def main():
    args = parse_args()
    run_a, run_b = _load_run(args.model_a), _load_run(args.model_b)
    rows_a, path_a = _read_rows(args.model_a)
    rows_b, path_b = _read_rows(args.model_b)
    keys = assert_paired_bundles(run_a, run_b, rows_a, rows_b)
    order = {key: index for index, key in enumerate(_pair_keys(rows_a))}
    rows_b = sorted(rows_b, key=lambda row: order[
        (int(row["origin_id"]), int(row["cohort_id"]), str(row["track"]), int(row["tau"]))
    ])

    pre_a = assert_pre_event_identity(rows_a, run_a["model"])
    pre_b = assert_pre_event_identity(rows_b, run_b["model"])

    # Saturated large-event cohorts are excluded from the pooled tables, so the
    # offline kernel has to drop them *before* it builds its statistics tensor;
    # otherwise it silently averages a different cohort set than the evaluator.
    main_cohort_ids = _main_cohort_ids(args.model_a)
    wanted = {int(value) for value in main_cohort_ids}
    main_a = [row for row in rows_a if int(row["cohort_id"]) in wanted]
    main_b = [row for row in rows_b if int(row["cohort_id"]) in wanted]
    main_keys = _pair_keys(main_a)
    span_a = _trajectory_span(args.model_a, args.input_len)
    span_b = _trajectory_span(args.model_b, args.input_len)
    if span_a != span_b:
        raise ValueError(f"PLES trajectory spans differ: {span_a} vs {span_b}")
    tensor_a = RowTensor(main_a, main_keys, args.input_len, span_a)
    tensor_b = RowTensor(main_b, main_keys, args.input_len, span_a)
    stats_a = verify_against_production(
        tensor_a, rows_a, run_a, args.recompute_tolerance, main_cohort_ids
    )
    stats_b = verify_against_production(
        tensor_b, rows_b, run_b, args.recompute_tolerance, main_cohort_ids
    )

    print(f"PairedBootstrap: {run_a['model']}({run_a['model_variant']}) vs {run_b['model']}({run_b['model_variant']})")
    print(f"  seed={run_a['seed']} | rows={len(rows_a)} | origins={len(tensor_a.origin_ids)} "
          f"| day-blocks={len(tensor_a.block_rows)} | cohorts={tensor_a.cohort_ids} "
          f"| strata={len(tensor_a.regimes)}")
    print(f"  protocol_hash={run_a['hashes']['protocol_hash']}")
    print(f"  training_protocol_hash={run_a['hashes']['training_protocol_hash']}")
    print(f"  PreEventCheck: identical treated/control sums on {pre_a}/{pre_b} pre-event rows")
    print(f"  Recomputation from {path_a.name} / {path_b.name} matches both frozen reports "
          f"(rtol={args.recompute_tolerance})")

    draws = paired_bootstrap(tensor_a, tensor_b, args.bootstrap_samples, args.bootstrap_seed)
    records = build_table(draws=draws, stats_a=stats_a, stats_b=stats_b,
                          model_a=run_a["model"], model_b=run_b["model"])

    family = [record for key in PRIMARY_FAMILY for record in records
              if (record["table"], record["row"], record["metric"]) == ("main",) + key]
    adjusted = _holm([record["p_bootstrap"] for record in family])
    for record, value in zip(family, adjusted):
        record["p_holm_primary"] = value

    # Two arms of the same model differ only in their config hash, so the model
    # name alone let one comparison silently overwrite another.
    def _tag(run):
        suffix = run["result_dir"].replace("\\", "/").split("/")[-2]
        return f"{run['model']}-{suffix.split('-cfg')[-1]}" if "-cfg" in suffix else run["model"]

    output_dir = Path(args.output_dir) / f"{_tag(run_a)}-vs-{_tag(run_b)}-seed{run_a['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["table", "row", "metric", run_a["model"], run_b["model"], "delta",
                  "ci_low", "ci_high", "p_bootstrap", "p_holm_primary", "lower_is_better", "verdict"]
    with (output_dir / "paired_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "analysis": "stage-0 offline paired overlap-block x cohort bootstrap",
        "model_a": {key: run_a[key] for key in ("model", "model_variant", "seed", "result_dir")},
        "model_b": {key: run_b[key] for key in ("model", "model_variant", "seed", "result_dir")},
        "hashes": run_a["hashes"],
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                      "outer_unit": "overlap-connected origin block, stratified by day_type",
                      "trajectory_span": span_a,
                      "design": "origins and cohorts are crossed, not nested",
                      "paired_across_models": True},
        "pre_event_rows_checked": {run_a["model"]: pre_a, run_b["model"]: pre_b},
        "primary_endpoint_family": [f"{name}/{metric}" for name, metric in PRIMARY_FAMILY],
        "single_seed_diagnostic_only": True,
        "records": records,
    }
    with (output_dir / "paired_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"-> written to {output_dir}")
    return records


if __name__ == "__main__":
    main()
