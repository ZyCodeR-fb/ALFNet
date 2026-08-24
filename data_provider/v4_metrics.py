"""Canonical v4 table arithmetic, with no derived WorstGroup endpoint."""

from collections import defaultdict

import numpy as np

from data_provider.calendar_strata import weighted_stratum_mean

PHASE_ORDER = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
BASE_GROUPS = ("Unseen", "Steady", "Join", "Fault")
MAIN_TABLE_ROWS = BASE_GROUPS + ("LifeAvg",)
METRICS = ("MAE", "RMSE", "MAPE")


def row_metrics(row, prefix):
    count = int(row.get("count", 0))
    if count <= 0:
        return None
    return {
        "MAPE": float(row[f"{prefix}_ape_sum"]) / count * 100.0,
        "MAE": float(row[f"{prefix}_ae_sum"]) / count,
        "RMSE": float(np.sqrt(float(row[f"{prefix}_se_sum"]) / count)),
    }


def mean_metrics(metrics):
    if not metrics:
        return None
    return {name: float(np.mean([entry[name] for entry in metrics]))
            for name in METRICS}


def calendar_balanced_metrics(rows, prefix):
    by_stratum = defaultdict(list)
    for row in rows:
        metrics = row_metrics(row, prefix)
        if metrics is not None:
            by_stratum[str(row.get("regime", "all"))].append(metrics)
    if not by_stratum:
        return None
    stratum_metrics = {name: mean_metrics(values)
                       for name, values in by_stratum.items() if values}
    combined = {
        metric: weighted_stratum_mean(
            {name: entry[metric] for name, entry in stratum_metrics.items()})
        for metric in METRICS
    }
    return (combined if all(np.isfinite(value) for value in combined.values())
            else None)


def tau_balanced_metrics(rows, prefix):
    by_tau = defaultdict(list)
    for row in rows:
        by_tau[int(row["tau"])].append(row)
    metrics = [calendar_balanced_metrics(by_tau[tau], prefix)
               for tau in sorted(by_tau)]
    return mean_metrics([entry for entry in metrics if entry is not None])
