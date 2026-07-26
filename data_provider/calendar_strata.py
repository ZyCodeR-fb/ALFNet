"""Shared calendar stratification and event-origin selection.

Both the validation replay and the LCE/PLES test set need to answer the same
question: given a segment of the timeline, which instants become event origins?
They used to answer it separately, and both answered it by taking the midpoint
of every (day, regime) group. On METR-LA that collapsed the whole evaluation
onto six clock times -- every AM origin landed at 08:30, every PM origin at
17:30, every Weekend origin at 12:00.

Three things change here.

Stratification
    The old four regimes mixed levels: ``Weekend`` is a day type while
    ``AM``/``PM``/``OffPeak`` are times of day, so weekend mornings and weekend
    nights were the same cell. Day type and time of day are now separate
    factors, giving ``len(DAY_TYPES) * len(TOD_BINS)`` strata and a weekend that
    is covered across the clock like any weekday.

Placement
    Within a stratum the origin is placed by an additive-recurrence
    (golden-ratio) sequence indexed by the stratum's rank. This is a
    one-dimensional low-discrepancy sequence: deterministic, seed-free,
    reproducible, and it spreads origins across the clock as days accumulate
    instead of pinning them to the midpoint.

Availability
    An origin whose trajectory window is mostly unobservable produces no usable
    rows. METR-LA contains 45 blocks of at least three consecutive steps where
    every node is missing, and the old selector walked straight into one: origin
    51 of the frozen manifest sits inside a six-hour network-wide outage, so it
    silently contributed nothing while the manifest still advertised it. Origins
    now have to clear a minimum observed fraction.

Aggregation weights are defined here too, so selection and reporting cannot
drift apart. Time-of-day bins are equally weighted inside a day type, and the
two day types are combined by their true calendar share (5/7, 2/7) rather than
equally. Under the old equal-weight-over-four-regimes rule a validation Weekend
cell built from three origins carried a full quarter of the checkpoint metric.
"""

from datetime import datetime, timezone

import numpy as np


DAY_TYPES = ("Weekday", "Weekend")
# Contiguous within the day so that "place an origin inside this bin" is well
# defined; a wrap-around night bin would make the low-discrepancy offset
# meaningless.
TOD_BINS = (
    ("EarlyAM", 0, 7),
    ("AMPeak", 7, 10),
    ("Midday", 10, 16),
    ("PMPeak", 16, 19),
    ("Evening", 19, 24),
)
TOD_NAMES = tuple(name for name, _, _ in TOD_BINS)
STRATA = tuple(f"{day}/{tod}" for day in DAY_TYPES for tod in TOD_NAMES)
# Calendar share of each day type. Dataset independent.
DAY_TYPE_WEIGHTS = {"Weekday": 5.0 / 7.0, "Weekend": 2.0 / 7.0}
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def as_datetime(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        micros = value.astype("datetime64[us]").astype(np.int64)
        return datetime.fromtimestamp(
            float(micros) / 1_000_000.0,
            tz=timezone.utc,
        ).replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def timestamp_text(value):
    return as_datetime(value).isoformat(sep=" ")


def day_type(value):
    return "Weekend" if as_datetime(value).weekday() >= 5 else "Weekday"


def tod_bin(value):
    hour = as_datetime(value).hour
    for name, start, end in TOD_BINS:
        if start <= hour < end:
            return name
    raise ValueError(f"hour {hour} falls outside the time-of-day bins")


def calendar_stratum(value):
    """The stratum label used for balancing, selection and aggregation."""
    return f"{day_type(value)}/{tod_bin(value)}"


def stratum_weights():
    """Equal over time-of-day inside a day type, calendar share across them."""
    return {
        f"{day}/{tod}": DAY_TYPE_WEIGHTS[day] / len(TOD_NAMES)
        for day in DAY_TYPES
        for tod in TOD_NAMES
    }


def weighted_stratum_mean(values_by_stratum):
    """Combine per-stratum values; absent strata are dropped and renormalised."""
    weights = stratum_weights()
    total_weight, total = 0.0, 0.0
    for name, value in values_by_stratum.items():
        weight = weights.get(str(name))
        if weight is None or value is None or not np.isfinite(value):
            continue
        total += weight * float(value)
        total_weight += weight
    return float(total / total_weight) if total_weight > 0 else float("nan")


def observed_fraction(values, start, stop):
    """Fraction of finite targets over a trajectory window."""
    if values is None:
        return 1.0
    window = np.asarray(values)[int(start):int(stop)]
    if window.size == 0:
        return 0.0
    return float(np.isfinite(window).mean())


def lce_episode_span(input_len, pred_len):
    """Steps an absolute-accuracy episode touches, over tau = 0..input_len-1.

    The earliest input step is ``origin - input_len + 1`` and the latest target
    is ``origin + input_len + pred_len``, so two origins this far apart share no
    data in the LCE window. The PLES tail (tau up to ``post_steps``) reaches
    further; spacing it fully would need six hours per origin, which cannot fit
    five time-of-day bins into a day, so the tail is allowed to overlap and the
    manifest records that.
    """
    return 2 * int(input_len) + int(pred_len)


def select_stratified_origins(
    timestamps,
    segment_length,
    input_len,
    pred_len,
    pre_steps=0,
    post_steps=0,
    per_stratum=1,
    values=None,
    min_observed_fraction=0.30,
    segment_offset=0,
    min_spacing=None,
):
    """Deterministically place event origins across calendar strata.

    Returns one record per accepted origin. ``values`` is an optional
    ``[time, node]`` (or ``[time, node, feature]``) target array used only for
    the availability guard; the guard is skipped when it is ``None``.
    """
    segment_length = int(segment_length)
    first = int(input_len) + int(pre_steps) - 1
    last = segment_length - int(post_steps) - int(pred_len) - 1
    if first > last:
        return []
    if int(per_stratum) < 1:
        raise ValueError("per_stratum must be at least one")
    candidates = list(range(first, last + 1))
    if timestamps is None:
        span = int(input_len) + int(pred_len) + int(pre_steps) + int(post_steps)
        return [
            {
                "origin_id": rank + 1,
                "event_time_index": int(index) + int(segment_offset),
                "local_index": int(index),
                "event_timestamp": None,
                "calendar_day": f"block-{rank + 1:04d}",
                "day_type": "Weekday",
                "tod_bin": TOD_NAMES[0],
                "stratum": STRATA[0],
                "observed_fraction": 1.0,
            }
            for rank, index in enumerate(candidates[:: max(1, span)])
        ]

    timestamps = np.asarray(timestamps)
    grouped = {}
    for index in candidates:
        stamp = timestamps[index]
        key = (
            as_datetime(stamp).date().isoformat(),
            day_type(stamp),
            tod_bin(stamp),
        )
        grouped.setdefault(key, []).append(index)

    # Origins are the bootstrap's resampling unit, so their LCE episodes must not
    # share data. The check is global rather than per stratum, because adjacent
    # time-of-day bins touch (Midday ends where PMPeak starts) and a per-stratum
    # check let neighbouring bins place overlapping episodes.
    minimum_spacing = int(min_spacing if min_spacing is not None
                          else lce_episode_span(input_len, pred_len))
    window = int(input_len) + int(pre_steps)
    # Chronological order, not the alphabetical order of the bin names. Greedy
    # placement gives earlier strata priority, and sorting by name let AMPeak and
    # Evening crowd out Midday and PMPeak.
    def _stratum_order(item):
        (day, day_kind, tod), _group = item
        return (day, day_kind, TOD_NAMES.index(tod))

    selected, rejected, placed = [], [], []
    for rank, (key, group) in enumerate(sorted(grouped.items(), key=_stratum_order)):
        for slot in range(int(per_stratum)):
            offset = ((rank * int(per_stratum) + slot) * GOLDEN_RATIO_CONJUGATE) % 1.0
            index = group[int(offset * len(group))]
            if any(abs(index - other) < minimum_spacing for other in placed):
                continue
            fraction = observed_fraction(
                values,
                index - window + 1,
                index + int(post_steps) + int(pred_len) + 1,
            )
            if fraction < float(min_observed_fraction):
                rejected.append({"stratum": f"{key[1]}/{key[2]}", "observed_fraction": fraction})
                continue
            placed.append(index)
            selected.append((key, index, fraction))

    selected.sort(key=lambda item: item[1])
    origins = [
        {
            "origin_id": rank + 1,
            "event_time_index": int(index) + int(segment_offset),
            "local_index": int(index),
            "event_timestamp": timestamp_text(timestamps[index]),
            "calendar_day": str(key[0]),
            "day_type": str(key[1]),
            "tod_bin": str(key[2]),
            "stratum": f"{key[1]}/{key[2]}",
            "observed_fraction": float(fraction),
        }
        for rank, (key, index, fraction) in enumerate(selected)
    ]
    # The rejection tally is the audit trail for the availability guard: without
    # it a silently dropped origin looks exactly like a stratum that never
    # existed, which is how origin 51 of the old manifest went unnoticed.
    for origin in origins:
        origin["_candidate_strata"] = len(grouped)
    _record_rejections(origins, rejected)
    return origins


def _record_rejections(origins, rejected):
    if origins:
        origins[0]["_rejected"] = list(rejected)


def summarize_origins(origins):
    counts = {}
    clocks = {}
    for origin in origins:
        stratum = str(origin["stratum"])
        counts[stratum] = counts.get(stratum, 0) + 1
        stamp = origin.get("event_timestamp")
        if stamp:
            clocks.setdefault(stratum, set()).add(str(stamp)[11:16])
    rejected = list(origins[0].get("_rejected", [])) if origins else []
    rejected_by_stratum = {}
    for entry in rejected:
        name = str(entry["stratum"])
        rejected_by_stratum[name] = rejected_by_stratum.get(name, 0) + 1
    return {
        "origin_count": int(len(origins)),
        "candidate_strata": int(origins[0].get("_candidate_strata", len(counts))) if origins else 0,
        "stratum_counts": counts,
        "distinct_clock_times": {name: len(values) for name, values in sorted(clocks.items())},
        "rejected_for_missing_data": int(len(rejected)),
        "rejected_by_stratum": rejected_by_stratum,
        "day_type_counts": {
            day: sum(1 for origin in origins if origin["day_type"] == day)
            for day in DAY_TYPES
        },
    }


def strip_diagnostics(origins):
    """Drop the private bookkeeping keys before an origin list is frozen."""
    return [
        {key: value for key, value in origin.items() if not key.startswith("_")}
        for origin in origins
    ]
