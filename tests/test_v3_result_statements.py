"""The published v3 result statements, locked against a committed fixture.

Two claims about these results were written down wrongly and had to be
corrected, both by over-generalising from a partial check:

  * "METR-LA has no cell where ALFNet wins all three seeds" -- it has two,
    Cold/RMSE and Onset/MAPE. The check that produced the claim had only been
    run over the 18 main-table cells and was then stated of all 36.
  * "PEMS08 Join is 2/3 on every metric" -- Join/RMSE is 1/3.

So the counting is exercised here rather than trusted, and the numbers it
produces are pinned.

The fixture is committed under ``tests/fixtures`` on purpose. ``outputs/`` is
gitignored and is rebuilt or cleaned routinely, and tests in this repo may not
depend on a result directory outside the working tree -- seven earlier tests
did, and broke the moment that tree was cleaned. The fixture holds the
recomputation from ``ples_paired_rows.csv``, which reproduces the frozen
summaries to 0.0.

Not ``tests/data``: ``.gitignore`` carries a bare ``data/``, which matches a
directory of that name at any depth, so a fixture placed there would be ignored
and the test would pass here and fail on a fresh clone -- the same
non-self-contained failure in a new disguise.
"""

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "v3_final_18_per_seed_cells.json"
METRICS = ("MAPE", "MAE", "RMSE")
MAIN_ROWS = ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup")
PHASE_ROWS = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
SEEDS = ("32", "42", "52")


@pytest.fixture(scope="module")
def cells():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]


def _mean(values):
    return sum(values) / len(values)


def _seed_wins(cell):
    """Seeds where ALFNet's error is lower. Every LCE metric is lower-is-better."""
    return sum(1 for seed in SEEDS if cell["ALFNet"][seed] < cell["MoGERNN"][seed])


def _mean_win(cell):
    return _mean([cell["ALFNet"][s] for s in SEEDS]) < _mean([cell["MoGERNN"][s] for s in SEEDS])


def test_fixture_is_complete():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["seeds"] == [32, 42, 52]
    for dataset in ("METRLA", "PEMSBAY", "PEMS08"):
        for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS)):
            for row in rows:
                for metric in METRICS:
                    cell = data["cells"][dataset][table][row][metric]
                    for model in ("ALFNet", "MoGERNN"):
                        assert sorted(cell[model]) == sorted(SEEDS)


# --- descriptive mean win counts --------------------------------------------

EXPECTED_WIN_COUNTS = {
    "METRLA": {"main": 13, "phase": 14, "total": 27},
    "PEMSBAY": {"main": 14, "phase": 18, "total": 32},
    "PEMS08": {"main": 18, "phase": 18, "total": 36},
}


@pytest.mark.parametrize("dataset", sorted(EXPECTED_WIN_COUNTS))
def test_descriptive_mean_win_counts(cells, dataset):
    expected = EXPECTED_WIN_COUNTS[dataset]
    counts = {}
    for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS)):
        counts[table] = sum(
            1 for row in rows for metric in METRICS
            if _mean_win(cells[dataset][table][row][metric])
        )
    counts["total"] = counts["main"] + counts["phase"]
    assert counts == expected


def test_overall_win_count_is_95_of_108(cells):
    total = sum(
        1
        for dataset in EXPECTED_WIN_COUNTS
        for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS))
        for row in rows
        for metric in METRICS
        if _mean_win(cells[dataset][table][row][metric])
    )
    assert total == 95
    main_total = sum(v["main"] for v in EXPECTED_WIN_COUNTS.values())
    phase_total = sum(v["phase"] for v in EXPECTED_WIN_COUNTS.values())
    assert (main_total, phase_total) == (45, 50)


# --- the two corrected seed statements ---------------------------------------


def test_metrla_has_exactly_two_unanimous_cells(cells):
    """The first correction. The wrong claim was 'none', from a main-table-only check."""
    unanimous = sorted(
        f"{row}/{metric}"
        for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS))
        for row in rows
        for metric in METRICS
        if _seed_wins(cells["METRLA"][table][row][metric]) == 3
    )
    assert unanimous == ["Cold/RMSE", "Onset/MAPE"]


def test_pems08_join_rmse_is_one_of_three_seeds(cells):
    """The second correction. Join/MAPE and Join/MAE are 2/3; RMSE is 1/3."""
    join = cells["PEMS08"]["main"]["Join"]
    assert _seed_wins(join["MAPE"]) == 2
    assert _seed_wins(join["MAE"]) == 2
    assert _seed_wins(join["RMSE"]) == 1
    # the mean still favours ALFNet on all three, which is exactly why the
    # mean-win count overstates how settled PEMS08 is
    assert all(_mean_win(join[metric]) for metric in METRICS)


def test_metrla_mean_wins_are_not_backed_by_seed_agreement(cells):
    """METR-LA's 27/36 rests almost entirely on 2-of-3 seeds."""
    unanimous = sum(
        1
        for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS))
        for row in rows
        for metric in METRICS
        if _seed_wins(cells["METRLA"][table][row][metric]) == 3
    )
    assert unanimous == 2, "METR-LA is a near-tie, not a 27/36 win"


def test_pemsbay_evidence_bearing_groups_are_unanimous(cells):
    """Steady/Join/Fault/LifeAvg are 3/3 on every metric; Unseen is not."""
    for row in ("Steady", "Join", "Fault", "LifeAvg"):
        for metric in METRICS:
            assert _seed_wins(cells["PEMSBAY"]["main"][row][metric]) == 3, f"{row}/{metric}"
    unseen = cells["PEMSBAY"]["main"]["Unseen"]
    assert any(_seed_wins(unseen[metric]) < 3 for metric in METRICS)


# --- derived rows are not independent endpoints ------------------------------


def test_worstgroup_is_derived_from_the_four_primitive_groups(cells):
    """WorstGroup is a max over the four groups, so it adds no new measurement.

    It is counted as three of the 36 cells per dataset, which is why the 36
    figure is descriptive rather than 36 independent tests.
    """
    for dataset in EXPECTED_WIN_COUNTS:
        for metric in METRICS:
            for seed in SEEDS:
                groups = [
                    cells[dataset]["main"][row][metric]["ALFNet"][seed]
                    for row in ("Unseen", "Steady", "Join", "Fault")
                ]
                worst = cells[dataset]["main"]["WorstGroup"][metric]["ALFNet"][seed]
                assert worst == pytest.approx(max(groups), rel=1e-9)


def test_lifeavg_is_the_mean_of_the_four_primitive_groups(cells):
    for dataset in EXPECTED_WIN_COUNTS:
        for metric in METRICS:
            for seed in SEEDS:
                groups = [
                    cells[dataset]["main"][row][metric]["ALFNet"][seed]
                    for row in ("Unseen", "Steady", "Join", "Fault")
                ]
                life = cells[dataset]["main"]["LifeAvg"][metric]["ALFNet"][seed]
                assert life == pytest.approx(_mean(groups), rel=1e-9)
