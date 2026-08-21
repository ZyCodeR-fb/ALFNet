"""The 36 validation-only development cells, on the frozen aggregation.

What this is, stated where nobody can miss it: **a development diagnostic on the
validation split**. Its structure is isomorphic to the main and phase tables the
paper reports, and that is exactly why it needs the label -- a table with the
same six rows and the same three metrics is the easiest thing in this repo to
mistake for a result. Every payload it writes carries
`development_validation_only: true`, and the schema refuses the words `final`,
`confirmation` and `test`.

Nothing here re-derives an aggregation. The row metrics, the calendar-share
combination, the k-balanced mean, LifeAvg and WorstGroup are imported from
`data_provider.ples_protocol` and called unmodified, because a second
implementation of a protocol definition is how `plot_counterfactual_shockwave`
drifted three separate ways from the frozen one.

Two mappings connect the v4 loader to that aggregation:

* the four LCBT scenarios are the four main groups, with `zero_history` as
  `Unseen`. LifeAvg is their equal-weight mean and WorstGroup their max, both by
  the frozen functions;
* `tau` is the episode's evidence length `k`. Unseen and Steady carry no
  evidence trajectory, so all their rows share one bucket and the k-balanced
  mean collapses to the calendar-balanced one -- which is what it should be,
  and needs no special case.

MAPE uses `clamp(|target|, min=1e-6)` on the original-scale target, the same
denominator `experiments/ples_evaluation.py` uses. It is not softened here to
make a development number look better.
"""

import numpy as np
import torch

from data_provider.calendar_strata import calendar_stratum
from data_provider.ples_protocol import (
    MAIN_TABLE_ROWS,
    PHASE_ORDER,
    _calendar_balanced_metrics_v2,
    _mean_metrics_v2,
    _tau_balanced_metrics_v2,
    _worst_group,
)

SCHEMA_VERSION = 1
METRICS = ("MAE", "RMSE", "MAPE")
SCENARIO_TO_GROUP = {"zero_history": "Unseen", "steady": "Steady",
                     "join": "Join", "fault": "Fault"}
BASE_GROUPS = ("Unseen", "Steady", "Join", "Fault")
# The frozen `_phase_id`: -1 for the two static anchors, then the join track
# 0/1/2 and the fault track 3/4/5. One mapping, taken from the protocol.
PHASE_ID_TO_NAME = {0: "Cold", 1: "Warm", 2: "Ready",
                    3: "Onset", 4: "Decay", 5: "Outage"}
MAPE_FLOOR = 1e-6
FORBIDDEN_SCHEMA_WORDS = ("final", "confirmation", "test")


class CollectorRefused(ValueError):
    """The collector will not produce a table from this input."""


def _refuse_non_validation(context, allow_phase="validation"):
    """Validation unless a caller names another phase explicitly.

    `allow_phase` exists for the final test evaluator, which is the one place a
    test split may be read and which has eight gates of its own. It is a named
    argument rather than a bypass so that every caller reading test appears in a
    grep for it; the default stays validation-only.
    """
    if context is None:
        raise CollectorRefused("the collector requires a context")
    if context.phase != allow_phase:
        raise CollectorRefused(
            f"this collector reads {allow_phase} only, got phase "
            f"{context.phase!r}; a test split is read once, by the final test "
            "evaluator, and never by a development diagnostic")


class ValidationCellCollector:
    """Accumulate error sums per (group, k, calendar regime) row.

    One row per (group, evidence length, regime, origin): the same granularity
    the frozen aggregation expects, so `_tau_balanced_metrics_v2` can be handed
    them unchanged.
    """

    def __init__(self, context, scaler, allow_phase="validation"):
        _refuse_non_validation(context, allow_phase)
        self.context = context
        self.mean = float(scaler["mean"])
        self.std = float(scaler["std"])
        self.rows = {"main": {name: [] for name in BASE_GROUPS},
                     "phase": {name: [] for name in PHASE_ORDER}}
        self.origins = {"main": {name: set() for name in BASE_GROUPS},
                        "phase": {name: set() for name in PHASE_ORDER}}
        self.samples = {"main": {name: 0 for name in BASE_GROUPS},
                        "phase": {name: 0 for name in PHASE_ORDER}}

    def _error_sums(self, prediction, target, valid):
        """Original scale on both sides; the offset cancels in the error."""
        predicted = prediction[valid] * self.std + self.mean
        actual = target[valid] * self.std + self.mean
        difference = predicted - actual
        denominator = torch.clamp(torch.abs(actual), min=MAPE_FLOOR)
        return {
            "count": int(valid.sum().item()),
            "treated_ape_sum": float(torch.sum(torch.abs(difference) / denominator)),
            "treated_ae_sum": float(torch.sum(torch.abs(difference))),
            "treated_se_sum": float(torch.sum(difference ** 2)),
        }

    def update(self, batch, prediction):
        """One validation batch. Only masked, finite cells contribute."""
        if prediction.shape != batch["target"].shape:
            raise CollectorRefused(
                f"prediction {tuple(prediction.shape)} does not match target "
                f"{tuple(batch['target'].shape)}")
        mask = batch["loss_mask"].unsqueeze(-1)
        valid = mask & torch.isfinite(prediction) & torch.isfinite(batch["target"])
        for index in range(prediction.shape[0]):
            sample_valid = valid[index]
            if not bool(sample_valid.any()):
                continue
            sums = self._error_sums(prediction[index], batch["target"][index],
                                    sample_valid)
            group = SCENARIO_TO_GROUP[str(batch["scenario"][index])]
            origin = int(batch["origin"][index])
            row = {"tau": int(batch["evidence_steps"][index]),
                   "regime": str(batch["calendar_regime"][index]), **sums}
            self.rows["main"][group].append(row)
            self.origins["main"][group].add(origin)
            self.samples["main"][group] += 1
            phase_id = int(batch["phase_id"][index])
            if phase_id in PHASE_ID_TO_NAME:
                phase = PHASE_ID_TO_NAME[phase_id]
                self.rows["phase"][phase].append(dict(row))
                self.origins["phase"][phase].add(origin)
                self.samples["phase"][phase] += 1

    # --- the table ------------------------------------------------------------

    def _cell(self, table, name, metrics):
        rows = self.rows[table][name]
        count = sum(int(row["count"]) for row in rows)
        if metrics is None or count == 0:
            raise CollectorRefused(
                f"{table} row {name} produced no valid cell; refusing to report "
                "a table with a missing row rather than silently shrinking it")
        cell = {}
        for metric in METRICS:
            value = float(metrics[metric])
            if not np.isfinite(value):
                raise CollectorRefused(f"{table} {name} {metric} is not finite")
            cell[metric] = value
        cell.update({"valid_target_count": count,
                     "origin_count": len(self.origins[table][name]),
                     "sample_count": self.samples[table][name],
                     "finite": True})
        return cell

    def table(self):
        """The 36 cells: six main rows and six phase rows, three metrics each."""
        base = {}
        for name in BASE_GROUPS:
            base[name] = _tau_balanced_metrics_v2(self.rows["main"][name], "treated")
            if base[name] is None:
                raise CollectorRefused(f"main row {name} has no valid cells")
        derived = dict(base)
        derived["LifeAvg"] = _mean_metrics_v2([base[name] for name in BASE_GROUPS])
        derived["WorstGroup"] = _worst_group(base)

        main = {name: self._cell("main", name, derived[name]) for name in BASE_GROUPS}
        for name in ("LifeAvg", "WorstGroup"):
            # Derived rows have no rows of their own; their counts are the union
            # of the four they summarise, recorded so a reader can see what they
            # were computed from.
            cell = {metric: float(derived[name][metric]) for metric in METRICS}
            if not all(np.isfinite(value) for value in cell.values()):
                raise CollectorRefused(f"main row {name} is not finite")
            cell.update({
                "valid_target_count": sum(main[base_name]["valid_target_count"]
                                          for base_name in BASE_GROUPS),
                "origin_count": len(set().union(
                    *(self.origins["main"][base_name] for base_name in BASE_GROUPS))),
                "sample_count": sum(self.samples["main"][base_name]
                                    for base_name in BASE_GROUPS),
                "finite": True,
                "derived_from": list(BASE_GROUPS),
            })
            main[name] = cell

        phase = {name: self._cell(
            "phase", name,
            _tau_balanced_metrics_v2(self.rows["phase"][name], "treated"))
            for name in PHASE_ORDER}

        table = {
            "schema_version": SCHEMA_VERSION,
            "development_validation_only": True,
            "split": "validation",
            "note": ("A development diagnostic on the validation split. Same "
                     "shape as the reported tables on purpose; not a result."),
            "aggregation": {
                "row_metrics": "sum/count per cell (ples_protocol._row_metrics_v2)",
                "within_k": "mean inside calendar stratum, then 5/7-2/7 weighted",
                "across_k": "equal-weight mean over evidence lengths",
                "LifeAvg": "mean(Unseen, Steady, Join, Fault)",
                "WorstGroup": "max(Unseen, Steady, Join, Fault)",
                "mape_denominator": f"clamp(|target|, min={MAPE_FLOOR})",
                "scale": "original units (normalised * std + mean)",
            },
            "main": main,
            "phase": phase,
        }
        _assert_schema_is_clean(table)
        return table


def _assert_schema_is_clean(table):
    """No key may call a validation diagnostic final, confirmation or test."""
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                for word in FORBIDDEN_SCHEMA_WORDS:
                    if word in lowered:
                        raise CollectorRefused(
                            f"schema key {path + [key]} contains {word!r}; a "
                            "validation diagnostic may not be named that")
                walk(value, path + [str(key)])
    walk(table, [])
    if len(table["main"]) != len(MAIN_TABLE_ROWS) or len(table["phase"]) != len(PHASE_ORDER):
        raise CollectorRefused(
            f"expected {len(MAIN_TABLE_ROWS)} main and {len(PHASE_ORDER)} phase "
            f"rows, got {len(table['main'])} and {len(table['phase'])}")
    if set(table["main"]) != set(MAIN_TABLE_ROWS):
        raise CollectorRefused(f"main rows are {sorted(table['main'])}")
    if set(table["phase"]) != set(PHASE_ORDER):
        raise CollectorRefused(f"phase rows are {sorted(table['phase'])}")


def cell_count(table):
    return sum(len(table[section]) * len(METRICS) for section in ("main", "phase"))


@torch.no_grad()
def collect_validation_table(session, epoch=0, max_batches=None):
    """Run the validation loader once and return the 36-cell table."""
    from experiments.v4_runner import to_device  # local: avoids a cycle
    from lib.lcbt_v4 import build_v4_teacher_forcing_mask
    from utils.model_output import unpack_model_output

    _refuse_non_validation(session.validation_context)
    session.model.eval()
    collector = ValidationCellCollector(session.validation_context, session.scaler)
    for index, batch in enumerate(session.validation_loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = to_device(batch, session.device)
        teacher = build_v4_teacher_forcing_mask(batch, session.run_seed, epoch, False)
        if bool(teacher.any()):
            raise CollectorRefused("validation produced a non-empty teacher mask")
        prediction, _ = unpack_model_output(
            session.model.forward_v4(batch, session.validation_context,
                                     epoch=epoch, teacher_forcing_mask=None))
        collector.update(batch, prediction)
    return collector.table()


# --- three-seed aggregation ----------------------------------------------------


def aggregate_seeds(tables):
    """mean, sample std (ddof=1) and the per-seed values, cell by cell.

    Sample std, not population: three runs are a sample of the seed distribution,
    and ddof=0 would understate the spread by ~18% at n=3.
    """
    if not tables:
        raise CollectorRefused("no seed tables to aggregate")
    seeds = sorted(tables)
    summary = {"schema_version": SCHEMA_VERSION,
               "development_validation_only": True,
               "seeds": [int(seed) for seed in seeds], "n": len(seeds),
               "std_ddof": 1}
    for section in ("main", "phase"):
        rows = sorted(tables[seeds[0]][section])
        summary[section] = {}
        for row in rows:
            summary[section][row] = {}
            for metric in METRICS:
                values = [float(tables[seed][section][row][metric]) for seed in seeds]
                summary[section][row][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                    "per_seed": {str(seed): value
                                 for seed, value in zip(seeds, values)},
                }
            counts = {int(tables[seed][section][row]["valid_target_count"])
                      for seed in seeds}
            summary[section][row]["valid_target_count"] = sorted(counts)
            summary[section][row]["valid_target_count_agrees"] = len(counts) == 1
    _assert_schema_is_clean({"main": summary["main"], "phase": summary["phase"]})
    return summary
