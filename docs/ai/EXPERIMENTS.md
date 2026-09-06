# Experiment evidence index

This file indexes evidence; it does not duplicate full metric tables. Conclusions are
limited to the stated dataset, seeds, phase, code identity, and provenance.

## E-4D-11M-3S — Four-dataset, 11-model validation/test report

- **Evidence:** `outputs_v4/reports/four_dataset_11_models_final/7523ac4c11f14e222dddedc905c9b7d50bc7eece1fc4deebc83a8994c382ac84/summary.json`
- **Scope:** PEMS03, PEMS04, PEMS08, PEMS-BAY; 11 models; seeds 32/42/52;
  validation and test.
- **Status:** complete historical report tied to source plan hash beginning `c8d6b8`.
- **Limit:** it is not the latest nine-model/METR-LA experiment and is not evidence
  that current HEAD satisfies the old plan hash.

## E-4D-SEED32-MIXED — Mixed-source four-dataset diagnostic

- **Evidence:** `outputs_v4/reports/seed32_hybrid_validation/59d38810eaa142934cb603ebb73b7c03fa0d0a29a2f030fdacef503a32bcd083/summary.json`
- **Scope:** validation, seed32, four datasets; corrected ALFNet/GWNET/MoGERNN mixed
  with earlier baseline artifacts.
- **Status:** diagnostic only.
- **Limit:** mixed provenance forbids presenting it as one formal comparison run.

## E-METRLA-9M-SEED32 — METR-LA nine-model candidate

- **Plan:** `configs/paper/metrla_seed32_nine_models_plan.json`, plan hash
  `057073f745ba24710cc4849abe91cc65c00b13ce4136337eb67563f60da35ff1`.
- **Evidence root:** `outputs_v4/metrla_seed32_nine_models/057073f745ba24710cc4849abe91cc65c00b13ce4136337eb67563f60da35ff1/`.
- **Scope:** METR-LA, nine selected models, seed32, validation only, ALF patch 8.
- **Checked state:** all nine run directories contain `COMPLETE` and
  `plan_validation_eval.json`; the plan builder check passed at the checkpoint.
- **Re-evaluation:** eight models agreed exactly with training-time validation. STGCN
  recorded `agrees=false`, maximum absolute difference
  `7.081064183012131e-06` against tolerance `1e-6`; cause remains pending.
- **Limit:** no seed42/52 conclusion and no test conclusion.

## E-5D-9M-SEED32-MERGED — Five-dataset reporting workbook

- **Evidence:** `outputs_v4/reports/five_dataset_seed32_nine_models_validation/05df96f892b8ccb532f081c4ece8a389c035849af3b50a264de5669f220cd6d4/`.
- **Files:** `five_dataset_seed32_nine_models_validation_tables.xlsx`, `tables.json`,
  and `provenance.json`.
- **Scope:** nine selected models, five datasets, seed32, validation only.
- **Status:** exploratory mixed-source report, as declared by its provenance file.
- **Limit:** the four pre-existing datasets and METR-LA were not produced by one newly
  frozen nine-model plan; it is not a formal three-seed or test table.

## Current run-state checkpoint

At `2026-09-06 15:29:32 +08:00`, process/GPU inspection found no GPU process whose
working command belonged to this repository. Existing tmux sessions were not treated
as evidence of an active job. This observation expires immediately after the stated
time and must be rechecked before operational decisions.
