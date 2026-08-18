# v4 minimal-change freeze

Status: **frozen scope**. This is the executable contract for the next round of
experiments. It deliberately does not redesign the models.

## Why this exists

v3 is a valid *topology-known* retrospective. Its statistics tooling has been
repaired, but the remaining report-engineering work (content-addressed
publication, an independent collector oracle, regenerating v3 diagnostics) does
not change any model, metric or experimental conclusion. Those are **paused**.
What follows is the minimum that has to be right before a v4 number means
anything.

## 1. Datasets

| dataset | role | reason |
|---|---|---|
| PEMS03 | **confirmation** | no formal test result has been read |
| PEMS04 | **confirmation** | no formal test result has been read |
| PEMS08 | development / exploratory | its v3 test split has been read |
| PEMS-BAY | development / exploratory | its v3 test split has been read |

METR-LA stays as a **retrospective/exploratory** record. Its v3 results are kept
in full and are not deleted; it simply does not enter this round's four-dataset
main table.

PEMS03 and PEMS04 test outputs must not be read — no performance, no
predictions, no model outcomes — until the model, hyper-parameters and epoch
rule are frozen. Both are reported together whatever the result; keeping only
the better one after the fact is outcome-dependent selection.

## 2. Target policy

Dataset-native targets, frozen before any v4 result:

| dataset | target | channel |
|---|---|---|
| PEMS03 | flow (`q`) | 0 — the only channel it has |
| PEMS04 | flow (`q`) | 0 |
| PEMS08 | flow (`q`) | 0 |
| PEMS-BAY | speed (`v`) | as in v3 |

Consequences, stated so they are not violated by habit:

* MAE/MAPE/RMSE are **not** comparable or averageable across datasets with
  different targets. Flow and speed are different quantities in different units.
* ALFNet and MoGERNN are compared only within one dataset and one target.
* The v3 PEMS08 result is **PEMS08-speed**. It cannot stand in for a
  PEMS08-flow result; PEMS08 must be retrained under v4.

## 3. Minimal-change constraints

* ALFNet's architecture is not rewritten this round.
* MoGERNN's architecture is not rewritten this round.
* No large new modules.
* Later model work is limited to **at most one or two** small changes, each with
  a stated diagnostic justification and a stated rollback condition.
* Any protocol or data fix applies to **both** models identically.
* The test split takes no part in tuning.
* 36/36 is a descriptive engineering goal only. It is never a reason to reopen a
  test split, swap datasets, or drop a failing result.

## 4. What is paused, and why it does not block this

| paused | reason |
|---|---|
| Stage 1R-D2-B content-addressed publication | report packaging; affects no metric |
| Collector independent oracle (C) | v3 tables already reproduce to 0.0 by the current checker |
| v3 diagnostic regeneration | descriptive post-hoc output only |

Existing Stage 1 code stays as it is and is not extended further. None of the
paused items gate v4 data or model work.
