"""Three-seed crossed paired diagnostics, ALFNet against MoGERNN. Post-hoc.

``scripts/paired_model_bootstrap.py`` compares one seed against one seed. It
says so itself -- it asserts the two runs share a seed and stamps
``single_seed_diagnostic_only`` into its output -- but that leaves the whole
comparison resting on a single training run, and training randomness is a real
source of variance here: on METR-LA the seed ordering reverses on most cells.

This adds the seed level. Seeds, overlap blocks and cohorts are **crossed**, not
nested: every seed replays the identical frozen evaluation design, verified by a
row-design hash that must match across all seeds and both models or the script
refuses to run. So each factor is drawn once per replicate:

    for each bootstrap replicate
        draw a seed multiplicity over {32, 42, 52}
        draw one cohort multiplicity
        draw one stratified overlap-block sample
        apply that single block/cohort weighting to every drawn seed and to
        both models; a seed drawn twice is weighted twice, not resampled
        statistic = seed-weighted mean of (ALFNet - MoGERNN)

The point estimate is the mean over the three real seeds of the per-seed
difference, which equals the difference of the two three-seed means the
collector publishes -- the same number because the mean is linear, and a test
asserts it.

**This output is descriptive, not confirmatory.** Three reasons, all recorded in
the JSON as ``post_hoc_diagnostic_only`` and
``inferential_status = descriptive_not_formal``:

1. The seed factor cannot reach significance at n=3. The exact two-sided
   sign-flip null over three paired seeds has minimum p = 2/2^3 = 0.25.
2. The seed draw is coarse: three draws with replacement from three seeds give
   27 ordered sequences but only **10** distinct multiplicity vectors.
3. The practical margins were chosen *after* the v3 test split had been read, so
   they are post-hoc and cannot be presented as pre-registered thresholds.

**No p-like statistic is emitted.** A bootstrap tail area computed across three
training seeds is not a calibrated p-value, and a Holm transform of one is not a
multiplicity correction. Earlier revisions published both as "descriptive"; they
were removed instead, because a number that looks like a p-value will be read as
one. What remains is the point estimate, a nominal percentile interval and the
seed tally. The endpoint grouping is fixed in this file for reproducibility and
is **not** pre-registered -- v3 had no analysis plan before its test split was
read.

Non-finite values fail closed. A NaN per-seed delta used to satisfy the
lower-is-better comparison and count as an ALFNet win, and an all-NaN family
produced a Holm score of 0.0 out of a NaN tail area. Nothing here filters NaN
silently; it raises with the endpoint named, and the JSON is written with
``allow_nan=False``.

    python scripts/multiseed_paired_inference.py --dataset METRLA \\
        --output-dir ./outputs/reports/final/multiseed
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_lifecycle_results import _load_run  # noqa: E402
from scripts.final_runs import load_manifest, verify_eval  # noqa: E402
from scripts.paired_model_bootstrap import (  # noqa: E402
    RowTensor,
    _main_cohort_ids,
    _pair_keys,
    _read_rows,
    _trajectory_span,
    assert_paired_bundles,
    assert_pre_event_identity,
    compute_statistics,
)

MODEL_A, MODEL_B = "ALFNet", "MoGERNN"

# A 95% percentile interval from a handful of draws is not a 95% interval. Unit
# tests inject draws directly rather than weakening this production floor.
MIN_BOOTSTRAP_SAMPLES = 1000

# Endpoint groups, fixed in this file for reproducibility. NOT pre-registered
# and NOT multiple-testing families: v3 had no analysis plan before its test
# split was read, and no p-value or correction is emitted anywhere.
PRIMARY_LCE = tuple(("main", row, "MAE") for row in ("Unseen", "Steady", "Join", "Fault"))
PRIMARY_PLES = tuple(
    ("ples", effect, metric)
    for effect in ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
    for metric in ("MAPE", "MAE")
)
FAMILIES = {"lce_primary": PRIMARY_LCE, "ples": PRIMARY_PLES}

# JoinLift is a benefit; everything else here is an error or a penalty.
HIGHER_IS_BETTER = {("ples", "JoinLift", "MAPE"), ("ples", "JoinLift", "MAE")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["METRLA", "PEMSBAY", "PEMS08"])
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--post-hoc-margin-mae", type=float, default=0.05,
                        help="MAE difference below which a direction is called negligible. "
                             "POST-HOC: chosen after the v3 test split was read, so it is "
                             "a reporting aid, not an acceptance threshold")
    parser.add_argument("--post-hoc-margin-mape", type=float, default=0.25)
    args = parser.parse_args()
    if args.bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
        parser.error(
            f"--bootstrap-samples must be at least {MIN_BOOTSTRAP_SAMPLES} to report a "
            f"95% interval; got {args.bootstrap_samples}"
        )
    for name, value in (("--post-hoc-margin-mae", args.post_hoc_margin_mae),
                        ("--post-hoc-margin-mape", args.post_hoc_margin_mape)):
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and >= 0, got {value!r}")
    return args


def load_seed_bundles(dataset, output_root):
    """Exactly the manifest's runs for this dataset. Nothing found on disk."""
    payload = load_manifest()
    expected_seeds = sorted(payload["seeds"])
    wanted = defaultdict(dict)
    for run in payload["runs"]:
        if run["dataset"] != dataset:
            continue
        if run["seed"] in wanted[run["model"]]:
            raise SystemExit(f"manifest lists {run['model']} seed {run['seed']} twice")
        wanted[run["model"]][run["seed"]] = run

    problems = []
    for model in (MODEL_A, MODEL_B):
        seeds = sorted(wanted.get(model, {}))
        if seeds != expected_seeds:
            problems.append(f"{model}: seeds {seeds}, manifest declares {expected_seeds}")
    if problems:
        raise SystemExit("REFUSING:\n" + "\n".join(f"  - {p}" for p in problems))

    bundles = {}
    for seed in expected_seeds:
        per_model = {}
        for model in (MODEL_A, MODEL_B):
            run = wanted[model][seed]
            run_problems, directory = verify_eval(run, output_root)
            if run_problems:
                problems.append(f"{run['run_id']}: " + "; ".join(run_problems))
                continue
            per_model[model] = (run, Path(directory))
        bundles[seed] = per_model
    if problems:
        raise SystemExit("REFUSING:\n" + "\n".join(f"  - {p}" for p in problems))
    return expected_seeds, bundles


def build_tensors(seeds, bundles):
    """One RowTensor per (seed, model), with the pairing checks per seed."""
    tensors, provenance = {}, {}
    for seed in seeds:
        run_a, dir_a = bundles[seed][MODEL_A]
        run_b, dir_b = bundles[seed][MODEL_B]
        (rows_a, _path_a), (rows_b, _path_b) = _read_rows(dir_a), _read_rows(dir_b)
        loaded_a, loaded_b = _load_run(dir_a), _load_run(dir_b)

        # The manifest says which seed this directory should be; the result file
        # says which seed it is. Checking both is the point -- a directory moved
        # or relabelled by hand passes one and fails the other.
        for loaded, run in ((loaded_a, run_a), (loaded_b, run_b)):
            if int(loaded["seed"]) != int(run["seed"]):
                raise SystemExit(
                    f"{run['run_id']}: result says seed {loaded['seed']}, "
                    f"manifest says {run['seed']}"
                )
            if loaded["model"] != run["model"]:
                raise SystemExit(
                    f"{run['run_id']}: result says model {loaded['model']}, "
                    f"manifest says {run['model']}"
                )

        assert_paired_bundles(loaded_a, loaded_b, rows_a, rows_b)

        # Put B's rows in A's order. Every statistic is computed from a shared
        # weight vector indexed by row position, so the two tensors must agree
        # row for row or the pairing is silently broken.
        order = {key: index for index, key in enumerate(_pair_keys(rows_a))}
        rows_b = sorted(rows_b, key=lambda row: order[
            (int(row["origin_id"]), int(row["cohort_id"]),
             str(row["track"]), int(row["tau"]))
        ])
        assert_pre_event_identity(rows_a, MODEL_A)
        assert_pre_event_identity(rows_b, MODEL_B)

        manifest = json.loads((dir_a / "protocol_manifest.json").read_text("utf-8"))
        input_len = int(manifest["resolved"]["input_len"])
        span_a = _trajectory_span(dir_a, input_len)
        span_b = _trajectory_span(dir_b, input_len)
        if span_a != span_b:
            raise SystemExit(f"seed {seed}: PLES spans differ, {span_a} vs {span_b}")
        cohort_ids = _main_cohort_ids(dir_a)

        # Saturated cohorts are outside the pooled tables, so they are dropped
        # before the tensor is built rather than filtered afterwards.
        wanted = {int(value) for value in cohort_ids}
        main_a = [row for row in rows_a if int(row["cohort_id"]) in wanted]
        main_b = [row for row in rows_b if int(row["cohort_id"]) in wanted]
        main_keys = _pair_keys(main_a)
        tensors[seed] = {
            MODEL_A: RowTensor(main_a, main_keys, input_len, span_a),
            MODEL_B: RowTensor(main_b, main_keys, input_len, span_a),
        }
        span = span_a
        provenance[seed] = {
            "row_design_hash": _row_design_hash(main_a),
            "row_count": len(main_a),
            "protocol_hash": loaded_a["hashes"]["protocol_hash"],
            "training_protocol_hash": loaded_a["hashes"].get("training_protocol_hash"),
            "input_len": input_len,
            "trajectory_span": span,
            "main_cohort_ids": cohort_ids,
            MODEL_A: run_a["run_id"],
            MODEL_B: run_b["run_id"],
        }

    hashes = {entry["protocol_hash"] for entry in provenance.values()}
    if len(hashes) != 1:
        raise SystemExit(f"seeds disagree on the evaluation protocol hash: {sorted(hashes)}")

    # The crossed resampler builds one weight vector and applies it to every
    # seed's tensor, so the row order must be identical everywhere. Verified,
    # not assumed: if a seed ever carried a different design the weights would
    # silently address different rows and the pairing would be meaningless.
    designs = {seed: entry["row_design_hash"] for seed, entry in provenance.items()}
    if len(set(designs.values())) != 1:
        raise SystemExit(
            "seeds do not share one evaluation row design, so seed/block/cohort "
            f"are not crossed and this estimator does not apply: {designs}"
        )
    return tensors, provenance


DESIGN_FIELDS = ("origin_id", "cohort_id", "track", "tau", "phase", "regime",
                 "event_time_index", "calendar_day", "cohort_fraction",
                 "cohort_style", "node_count", "count")


def _row_design_hash(rows):
    """Digest of the evaluation design, in row order, excluding any model output.

    Only protocol-determined fields go in -- no treated/control sums -- so two
    models on one seed must produce the same digest, and so must two seeds. It
    covers row *order* as well as content because the resampler indexes rows by
    position.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update("|".join(str(row[field]) for field in DESIGN_FIELDS).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


PLES_EFFECTS = ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
ARM_METRICS = ("MAPE", "MAE")

# Effect definitions and directions, stated rather than inferred from a name.
#   JoinLift          = control(Unseen)  - treated(Join)     larger is better
#   FaultPenalty      = treated(Fault)   - control(Steady)   smaller is better
#   OutageFaultPenalty= the same, read at tau = input_len - 1
# The arms are absolute errors, so on an arm smaller is always better.
EFFECT_DEFINITION = {
    "JoinLift": {"track": "join", "orientation": "higher_is_better",
                 "formula": "control(Unseen) - treated(Join)"},
    "FaultPenalty": {"track": "fault", "orientation": "lower_is_better",
                     "formula": "treated(Fault) - control(Steady)"},
    "OutageFaultPenalty": {"track": "fault", "orientation": "lower_is_better",
                           "formula": "treated - control at tau = input_len - 1"},
}


def expected_endpoint_keys(input_len, include_ples_arms=True):
    """The endpoint inventory, derived from the protocol rather than observed.

    Taking "expected" from whatever the run produced makes the completeness
    check vacuous: if point, draws and every seed all lack the same endpoint,
    comparing them to each other still passes. This is generated from
    ``input_len`` alone, so a silently dropped endpoint is caught.

    At input_len=12 this is 114 keys, or 126 with the PLES arms.
    """
    keys = set()
    for metric in ("MAPE", "MAE", "RMSE"):
        for name in ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup"):
            keys.add(("main", name, metric))
        for phase in ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage"):
            keys.add(("phase", phase, metric))
        for label in ("Join", "Fault"):
            for k in range(1, int(input_len) + 1):
                keys.add(("trajectory", f"{label}_k{k}", metric))
    for metric in ARM_METRICS:
        for effect in PLES_EFFECTS:
            keys.add(("ples", effect, metric))
            if include_ples_arms:
                for arm in ("treated", "control"):
                    keys.add(("arm", f"{effect}/{arm}", metric))
    return keys


def assert_endpoint_inventory(observed, input_len, what, include_ples_arms=True):
    expected = expected_endpoint_keys(input_len, include_ples_arms)
    observed = set(observed)
    if observed != expected:
        raise NonFiniteValue(
            f"{what}: endpoint inventory mismatch against the protocol "
            f"(expected {len(expected)}, got {len(observed)}); "
            f"missing {sorted(expected - observed)[:5]}, "
            f"extra {sorted(observed - expected)[:5]}")


def compute_arms(tensor, weights):
    """Treated and control absolute error behind each PLES endpoint.

    Mirrors the endpoint selection in ``compute_statistics`` exactly and takes
    the *same* weight vector, so within one bootstrap replicate the contrast and
    its two arms are computed from one draw. Recomputing the arms from a second
    draw would let them describe a different resample than the contrast they are
    supposed to explain.
    """
    post_slots = [slot for value, slot in tensor.tau_slot.items() if value >= 0]
    outage_slot = tensor.tau_slot.get(tensor.input_len - 1)
    output = {}
    for metric in ARM_METRICS:
        treated = tensor.calendar_balanced("treated", metric, weights)
        control = tensor.calendar_balanced("control", metric, weights)
        for effect in PLES_EFFECTS:
            track = tensor.track_slot[EFFECT_DEFINITION[effect]["track"]]
            if effect == "OutageFaultPenalty":
                if outage_slot not in post_slots:
                    raise NonFiniteValue(
                        f"missing the settled outage cell tau={tensor.input_len - 1}")
                index = post_slots.index(outage_slot)
                arms = {"treated": float(treated[track, post_slots][index]),
                        "control": float(control[track, post_slots][index])}
            else:
                arms = {"treated": float(np.mean(treated[track, post_slots])),
                        "control": float(np.mean(control[track, post_slots]))}
            for arm, value in arms.items():
                output[("arm", f"{effect}/{arm}", metric)] = value
    return output


def statistics_and_arms(tensor, weights):
    """One call so a caller cannot accidentally use two different draws."""
    merged = dict(compute_statistics(tensor, weights))
    merged.update(compute_arms(tensor, weights))
    return merged


def _cohort_multiplicity(tensor, rng):
    multiplicity = np.zeros(int(max(tensor.cohort_ids)) + 1, dtype=np.float64)
    for cohort_id in rng.choice(tensor.cohort_ids, size=len(tensor.cohort_ids), replace=True):
        multiplicity[int(cohort_id)] += 1.0
    return multiplicity


def _weights(tensor, rng, cohort_multiplicity):
    """Stratified overlap-block draw crossed with a cohort draw made elsewhere.

    The cohort draw is passed in rather than made here because a bootstrap
    sample replays one global cohort multiplicity at every seed it drew; drawing
    cohorts independently per seed would treat the frozen cohort set as if it
    were resampled three times.
    """
    block_multiplicity = defaultdict(float)
    for _stratum, blocks in tensor.blocks_by_stratum.items():
        for block in rng.choice(blocks, size=len(blocks), replace=True):
            block_multiplicity[str(block)] += 1.0
    weights = np.zeros(tensor.n_rows, dtype=np.float64)
    for block, repeats in block_multiplicity.items():
        indices = tensor.block_rows[block]
        weights[indices] += repeats * cohort_multiplicity[tensor.row_cohort[indices]]
    return weights


def per_seed_deltas(tensors, seeds):
    """Observed ALFNet - MoGERNN at each seed, plus each model's own arm values."""
    out, absolute = {}, {}
    for seed in seeds:
        ones_a = np.ones(tensors[seed][MODEL_A].n_rows, dtype=np.float64)
        ones_b = np.ones(tensors[seed][MODEL_B].n_rows, dtype=np.float64)
        stats_a = statistics_and_arms(tensors[seed][MODEL_A], ones_a)
        stats_b = statistics_and_arms(tensors[seed][MODEL_B], ones_b)
        out[seed] = {key: stats_a[key] - stats_b[key] for key in stats_a}
        absolute[seed] = {MODEL_A: {k: v for k, v in stats_a.items() if k[0] == "arm"},
                          MODEL_B: {k: v for k, v in stats_b.items() if k[0] == "arm"}}
    return out, absolute


def crossed_bootstrap(tensors, seeds, samples, seed):
    """Seeds, blocks and cohorts are crossed factors, so each is drawn once.

    Every seed replays the *same* frozen evaluation design -- identical origins,
    cohorts, tracks, taus, regimes and event times, verified by a row-design
    hash before this runs. So the three seeds are three model fits measured on
    one set of test events, not three independent samples of events.

    An earlier version drew a fresh block sample inside each drawn seed. That
    treats one set of physical test events as if it had been sampled once per
    seed, which inflates the effective sample size and narrows the interval. Here
    one block draw and one cohort draw are made per replicate and reused across
    every drawn seed and both models; drawing a seed twice only doubles its
    weight in the average, it does not resample the events again.
    """
    if samples <= 0:
        raise NonFiniteValue(f"crossed_bootstrap needs samples > 0, got {samples}")
    rng = np.random.default_rng(seed)
    reference = tensors[seeds[0]][MODEL_A]
    keys, draws = None, None
    for index in range(samples):
        seed_multiplicity = defaultdict(float)
        for drawn in rng.choice(seeds, size=len(seeds), replace=True):
            seed_multiplicity[int(drawn)] += 1.0
        # one draw each, shared by every seed and both models in this replicate
        cohorts = _cohort_multiplicity(reference, rng)
        weights = _weights(reference, rng, cohorts)

        accumulated = defaultdict(float)
        total = sum(seed_multiplicity.values())
        for drawn, repeats in seed_multiplicity.items():
            stats_a = statistics_and_arms(tensors[drawn][MODEL_A], weights)
            stats_b = statistics_and_arms(tensors[drawn][MODEL_B], weights)
            for key in stats_a:
                accumulated[key] += repeats * (stats_a[key] - stats_b[key]) / total
        if draws is None:
            keys = sorted(accumulated)
            draws = {key: np.empty(samples, dtype=np.float64) for key in keys}
        for key in keys:
            draws[key][index] = accumulated[key]
    return draws


class NonFiniteValue(SystemExit):
    """Raised, never swallowed, when a non-finite value reaches a statistic.

    Filtering NaN silently is what produced the two worst defects here: a NaN
    per-seed delta satisfied the lower-is-better test and was counted as an
    ALFNet win, and an all-NaN family produced a Holm score of 0.0 -- the
    strongest possible ordering -- out of a NaN tail area.
    """


def _require_finite(value, what):
    array = np.asarray(value, dtype=np.float64)
    bad = ~np.isfinite(array)
    if bool(np.any(bad)):
        index = int(np.argmax(bad)) if array.ndim else None
        where = "" if index is None else f" at index {index}"
        raise NonFiniteValue(f"non-finite value in {what}{where}: {array.ravel()[:5]}")
    return value


def seed_tally(directions, higher_is_better):
    """Count seeds by outcome, keeping ties and non-finite values separate.

    ``seeds_favouring == 0`` used to imply unanimity for the other model. It
    does not: per-seed deltas of ``[+1, 0, 0]`` gave "favours MoGERNN, all 3
    seeds agree" when only one seed favoured MoGERNN and two were exact ties.
    A tie supports neither model, and a non-finite delta supports nothing at all.
    """
    favour_a = favour_b = tie = nonfinite = 0
    for value in directions:
        if not np.isfinite(value):
            nonfinite += 1
        elif value == 0:
            tie += 1
        elif (value > 0) == higher_is_better:
            favour_a += 1
        else:
            favour_b += 1
    return {
        "n_favour_alfnet": favour_a,
        "n_favour_baseline": favour_b,
        "n_tie": tie,
        "n_nonfinite": nonfinite,
        "n_evaluable": favour_a + favour_b + tie,
    }


def _unanimous(tally, seed_count):
    """Every seed evaluable, and all of them on the same side."""
    if tally["n_evaluable"] != seed_count:
        return False
    return seed_count in (tally["n_favour_alfnet"], tally["n_favour_baseline"])


def _descriptive_verdict(delta, low, high, higher_is_better, margin, seeds_favouring,
                         seed_count):
    """A described direction, never a significance claim.

    Deliberately not a test. With three training seeds the exact two-sided
    sign-flip null has minimum p = 2/2^3 = 0.25, so no arrangement of three
    seeds can reach 0.05 on the seed factor; a bootstrap tail area computed
    across three fits is not a calibrated p-value for training randomness, and
    Holm-adjusting an uncalibrated p does not repair it. The wording therefore
    describes the interval and the seed agreement rather than declaring
    significance, and the margins are post-hoc (see the module docstring), which
    is a second reason no verdict here may be read as confirmatory.
    """
    if not np.isfinite(delta) or not np.isfinite(low) or not np.isfinite(high):
        return "undetermined"
    if delta == 0:
        return "no difference"
    favours_a = delta > 0 if higher_is_better else delta < 0
    direction = MODEL_A if favours_a else MODEL_B
    if not (low > 0 or high < 0):
        return "interval covers zero"
    if margin is not None and abs(delta) < margin:
        return f"below the {margin:g} post-hoc margin"
    scale = "" if margin is not None else ", direction only (no margin defined)"
    return f"favours {direction}{scale}"


def _assert_key_sets_agree(point, draws, seed_deltas, seeds, input_len=None):
    """point, draws and every seed must describe exactly the same endpoints.

    A missing endpoint in one of them would otherwise be scored on a quietly
    reduced set, and an extra one would be scored with no counterpart.
    """
    expected = set(point)
    problems = []
    if set(draws) != expected:
        problems.append(f"draws: missing {sorted(expected - set(draws))}, "
                        f"extra {sorted(set(draws) - expected)}")
    for seed in seeds:
        if set(seed_deltas[seed]) != expected:
            problems.append(f"seed {seed}: missing {sorted(expected - set(seed_deltas[seed]))}, "
                            f"extra {sorted(set(seed_deltas[seed]) - expected)}")
    if problems:
        raise NonFiniteValue("endpoint key sets disagree:\n  - " + "\n  - ".join(problems))
    if input_len is not None:
        assert_endpoint_inventory(point, input_len, "point estimates")


def build_records(point, draws, seed_deltas, seeds, margins, input_len=None):
    _assert_key_sets_agree(point, draws, seed_deltas, seeds, input_len)
    family_of = {key: name for name, keys in FAMILIES.items() for key in keys}
    records = []
    # Arm keys are reported by build_ples_records beside their contrast; emitting
    # them here as well would double-count them as standalone endpoints.
    for key in sorted(k for k in draws if k[0] != "arm"):
        table, row, metric = key
        delta_draws = np.asarray(draws[key], dtype=np.float64)
        # Order matters and used to be wrong: the percentile was taken over
        # `delta_draws[finite]`, i.e. NaN was filtered out before the check that
        # claimed nothing was filtered. Validate first, then compute.
        if delta_draws.size == 0:
            raise NonFiniteValue(f"{table}/{row}/{metric}: no bootstrap draws")
        _require_finite(delta_draws, f"bootstrap draws {table}/{row}/{metric}")
        low = float(np.percentile(delta_draws, 2.5))
        high = float(np.percentile(delta_draws, 97.5))
        _require_finite([low, high], f"interval ends {table}/{row}/{metric}")
        higher_is_better = key in HIGHER_IS_BETTER
        directions = [seed_deltas[s][key] for s in seeds]
        tally = seed_tally(directions, higher_is_better)
        if tally["n_nonfinite"]:
            raise NonFiniteValue(
                f"{table}/{row}/{metric}: {tally['n_nonfinite']} of {len(seeds)} "
                "per-seed deltas are non-finite; refusing to score this endpoint"
            )
        _require_finite(point[key], f"point estimate {table}/{row}/{metric}")
        if low > high:
            raise NonFiniteValue(f"{table}/{row}/{metric}: interval low {low} > high {high}")
        records.append({
            "table": table, "row": row, "metric": metric,
            "family": family_of.get(key),
            "delta": float(point[key]),
            "ci_low": low, "ci_high": high,
            "higher_is_better": higher_is_better,
            "per_seed_delta": {str(s): float(seed_deltas[s][key]) for s in seeds},
            **tally,
            "seeds_favouring_alfnet": tally["n_favour_alfnet"],
            "seed_count": len(seeds),
            "seed_unanimous": _unanimous(tally, len(seeds)),
        })

    # No p-like statistic is emitted. A bootstrap tail area over three training
    # seeds is not a calibrated p-value and a Holm transform of one is not a
    # correction; publishing either invites exactly the reading v3 cannot
    # support. Point, percentile interval and the seed tally are what remain.
    for record in records:
        margin = margin_for(record["metric"], margins)
        record["post_hoc_margin"] = margin
        record["verdict"] = _descriptive_verdict(
            record["delta"], record["ci_low"], record["ci_high"],
            record["higher_is_better"], margin,
            record["seeds_favouring_alfnet"], record["seed_count"])
        if record["family"] is None:
            record["verdict"] += " (exploratory)"
    return records


def _interval_status(delta, low, high, margin, higher_is_better):
    """favourable / degraded / equivalent / uncertain, from the interval.

    ``margin`` may be None, in which case only the sign of the interval is used
    and nothing is ever called equivalent -- there is no scale to call it on.
    """
    if not all(np.isfinite(v) for v in (delta, low, high)):
        return "uncertain"
    better_low, better_high = (low, high) if higher_is_better else (-high, -low)
    threshold = 0.0 if margin is None else margin
    if better_low > threshold:
        return "favourable"
    if better_high < -threshold:
        return "degraded"
    if margin is not None and -margin <= low and high <= margin:
        return "equivalent"
    return "uncertain"


def _arm_status(delta, low, high, margin):
    """Arms are absolute errors: lower is better, always."""
    status = _interval_status(delta, low, high, margin, higher_is_better=False)
    return {"favourable": "improve", "degraded": "degrade"}.get(status, status)


NON_DEGRADED_ARMS = {"improve", "equivalent"}


def arms_pareto_status(treated, control):
    if treated == "improve" and control in NON_DEGRADED_ARMS:
        return "treated_improves_control_non_degraded"
    if control == "degrade":
        return "control_degraded"
    if treated == "degrade":
        return "treated_degraded"
    if treated in NON_DEGRADED_ARMS and control in NON_DEGRADED_ARMS:
        return "no_material_arm_change"
    return "uncertain"


def combined_status(contrast, treated, control):
    """All three conditions, or a statement of which one failed.

    Deliberately not ``accepted``/``rejected``: v3 has no pre-registered gate,
    so this describes which conditions held, it does not adjudicate anything.
    Nor is it a joint 95% statement -- three marginal intervals do not have
    joint coverage, and a v4 joint rule belongs in a pre-registered plan.
    """
    if contrast == "favourable" and treated == "improve" and control in NON_DEGRADED_ARMS:
        return "all_conditions_met"
    if contrast != "favourable":
        return f"contrast_{contrast}"
    if control == "degrade":
        return "contrast_favourable_but_control_degraded"
    if treated != "improve":
        return f"contrast_favourable_but_treated_{treated}"
    return "uncertain"


def build_ples_records(point, draws, seed_deltas, seed_absolute, seeds,
                       contrast_margins, arm_margins):
    """One record per PLES endpoint carrying the contrast and both arms.

    Contrast and arms come from the same draws by construction: they were
    computed together in ``statistics_and_arms`` from one weight vector.
    """
    def _interval(key):
        values = np.asarray(draws[key], dtype=np.float64)
        if values.size == 0:
            raise NonFiniteValue(f"{key}: no bootstrap draws")
        _require_finite(values, f"draws {key}")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    records = []
    for effect in PLES_EFFECTS:
        for metric in ARM_METRICS:
            contrast_key = ("ples", effect, metric)
            treated_key = ("arm", f"{effect}/treated", metric)
            control_key = ("arm", f"{effect}/control", metric)
            for key in (contrast_key, treated_key, control_key):
                if key not in draws or key not in point:
                    raise NonFiniteValue(f"missing endpoint {key}")

            higher_is_better = EFFECT_DEFINITION[effect]["orientation"] == "higher_is_better"
            contrast_margin = margin_for(metric, contrast_margins)
            arm_margin = margin_for(metric, arm_margins)

            contrast_low, contrast_high = _interval(contrast_key)
            treated_low, treated_high = _interval(treated_key)
            control_low, control_high = _interval(control_key)

            contrast = _interval_status(point[contrast_key], contrast_low, contrast_high,
                                        contrast_margin, higher_is_better)
            treated = _arm_status(point[treated_key], treated_low, treated_high, arm_margin)
            control = _arm_status(point[control_key], control_low, control_high, arm_margin)

            # oriented so positive always favours ALFNet, whatever the endpoint
            oriented = point[contrast_key] if higher_is_better else -point[contrast_key]
            tally = seed_tally([seed_deltas[s][contrast_key] for s in seeds], higher_is_better)
            if tally["n_nonfinite"]:
                raise NonFiniteValue(f"{contrast_key}: non-finite per-seed contrast")

            records.append({
                "effect": effect, "metric": metric,
                "definition": EFFECT_DEFINITION[effect]["formula"],
                "orientation": EFFECT_DEFINITION[effect]["orientation"],
                "contrast_delta": float(point[contrast_key]),
                "contrast_oriented_advantage": float(oriented),
                "contrast_ci": {"low": contrast_low, "high": contrast_high},
                "contrast_margin": contrast_margin,
                "arm_margin": arm_margin,
                "treated_delta": float(point[treated_key]),
                "treated_ci": {"low": treated_low, "high": treated_high},
                "control_delta": float(point[control_key]),
                "control_ci": {"low": control_low, "high": control_high},
                "absolute": {
                    model: {
                        arm: {str(s): float(seed_absolute[s][model][("arm", f"{effect}/{arm}", metric)])
                              for s in seeds}
                        for arm in ("treated", "control")
                    }
                    for model in (MODEL_A, MODEL_B)
                },
                "per_seed_contrast": {str(s): float(seed_deltas[s][contrast_key]) for s in seeds},
                "per_seed_treated_delta": {str(s): float(seed_deltas[s][treated_key]) for s in seeds},
                "per_seed_control_delta": {str(s): float(seed_deltas[s][control_key]) for s in seeds},
                **tally,
                "seed_unanimous": _unanimous(tally, len(seeds)),
                "contrast_status": contrast,
                "arms_pareto_status": arms_pareto_status(treated, control),
                "treated_status": treated,
                "control_status": control,
                "combined_descriptive_status": combined_status(contrast, treated, control),
            })
    return records


def margin_for(metric, margins):
    """Margin for a metric, or None when no basis for one exists.

    RMSE deliberately has no margin. Borrowing MAPE's 0.25 would apply a
    percentage-scaled threshold to a squared-error scale, which is not the same
    quantity; without an independent basis the honest value is None and the
    endpoint is described by direction only. An unknown metric fails closed
    rather than silently inheriting a number.
    """
    if metric in margins:
        value = margins[metric]
        if value is None:
            return None
        if not np.isfinite(value) or value < 0:
            raise NonFiniteValue(f"margin for {metric} must be finite and >= 0, got {value!r}")
        return float(value)
    raise NonFiniteValue(f"no margin policy declared for metric {metric!r}")


def main():
    args = parse_args()
    seeds, bundles = load_seed_bundles(args.dataset, args.output_root)
    tensors, provenance = build_tensors(seeds, bundles)

    seed_deltas, seed_absolute = per_seed_deltas(tensors, seeds)
    point = {
        key: float(np.mean([seed_deltas[s][key] for s in seeds]))
        for key in seed_deltas[seeds[0]]
    }
    draws = crossed_bootstrap(tensors, seeds, args.bootstrap_samples,
                                   args.bootstrap_seed)
    margins = {"MAE": args.post_hoc_margin_mae,
               "MAPE": args.post_hoc_margin_mape,
               # no independent basis for an RMSE threshold; describe direction only
               "RMSE": None}
    input_len = provenance[seeds[0]]["input_len"]
    records = build_records(point, draws, seed_deltas, seeds, margins, input_len)
    # Contrast margins and arm margins are separate configuration on purpose:
    # sharing a number because the units match is how one silently becomes the
    # other. Both are post-hoc for v3.
    ples_records = build_ples_records(point, draws, seed_deltas, seed_absolute,
                                      seeds, margins, margins)

    print(f"MultiSeedPairedInference: {MODEL_A} vs {MODEL_B} | {args.dataset}")
    print(f"  seeds={seeds} | samples={args.bootstrap_samples} | "
          f"protocol={provenance[seeds[0]]['protocol_hash'][:12]}")
    print(f"  {'family':<12}{'row':<19}{'metric':<6}{'delta':>10}{'95% CI':>22}"
          f"  ALF/base/tie/eval  verdict")
    for record in records:
        if record["family"] is None:
            continue
        interval = f"[{record['ci_low']:+.4f},{record['ci_high']:+.4f}]"
        print(f"  {record['family']:<12}{record['row']:<19}{record['metric']:<6}"
              f"{record['delta']:>+10.4f}{interval:>22}"
              f"   {record['n_favour_alfnet']}/{record['n_favour_baseline']}"
              f"/{record['n_tie']}/{record['n_evaluable']}"
              f"{'  U' if record['seed_unanimous'] else '   '}"
              f"  {record['verdict']}")

    print()
    print(f"  {'PLES endpoint':<28}{'contrast':>10}{'treatedD':>10}{'controlD':>10}"
          f"   combined")
    for record in ples_records:
        print(f"  {record['effect'] + '/' + record['metric']:<28}"
              f"{record['contrast_delta']:>+10.4f}{record['treated_delta']:>+10.4f}"
              f"{record['control_delta']:>+10.4f}   {record['combined_descriptive_status']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{args.dataset}_multiseed_paired.json"
    destination.write_text(json.dumps({
        "analysis": "three-seed crossed paired bootstrap, v3 post-hoc",
        "post_hoc_diagnostic_only": True,
        "inferential_status": "descriptive_not_formal",
        "dataset": args.dataset,
        "model_a": MODEL_A, "model_b": MODEL_B,
        "seeds": seeds,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "design": "seeds, overlap blocks and cohorts are crossed, not nested",
            "design_reason": "every seed replays one frozen evaluation design; the "
                             "row-design hash is asserted equal across seeds and models",
            "seed_draw": "multiplicity over the three seeds; a repeated seed is "
                         "weighted again, not resampled",
            "block_draw": "one stratified overlap-block sample per replicate, "
                          "shared by every drawn seed and both models",
            "cohort_draw": "one global multiplicity per replicate, likewise shared",
            "paired_across_models": True,
        },
        "point_estimate": "mean over seeds of (ALFNet - MoGERNN); equals the "
                          "difference of the two published three-seed means",
        "endpoint_groups": {name: [list(k) for k in keys] for name, keys in FAMILIES.items()},
        "endpoint_groups_status": "code-fixed post-hoc grouping, not pre-registered",
        "post_hoc_margins": margins,
        "margin_status": "chosen after the v3 test split was read; post-hoc, not "
                         "pre-registered, and not usable as an acceptance threshold",
        "power_note": "n_seed=3. The exact two-sided sign-flip null over three "
                      "paired seeds has minimum p=0.25, so the seed factor cannot "
                      "reach 0.05 by any arrangement. Three draws with replacement "
                      "from three seeds give 27 ordered sequences but only 10 "
                      "distinct multiplicity vectors. Read the sign and the seed "
                      "agreement, not the width, and never as a formal test.",
        "significance_reporting": "none. No p-value, tail area or multiplicity "
                                  "correction is emitted; three training seeds cannot "
                                  "support one and a number shaped like a p-value "
                                  "would be read as one.",
        "provenance": provenance,
        "records": records,
        "ples_arm_records": ples_records,
        "ples_status_definitions": {
            "contrast_status": "favourable | degraded | equivalent | uncertain, "
                               "from the contrast interval against contrast_margin",
            "arms_pareto_status": "how the two absolute-error arms moved",
            "combined_descriptive_status": "all_conditions_met requires contrast "
                                           "favourable AND treated improve AND control "
                                           "non-degraded; it describes which conditions "
                                           "held and adjudicates nothing",
            "joint_coverage": "none. Three marginal intervals do not have joint 95% "
                              "coverage; a joint rule belongs in a pre-registered v4 plan.",
        },
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
