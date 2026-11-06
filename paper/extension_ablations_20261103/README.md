# Whole-mechanism ablations (2026-11-03)

w/o SAHC (no graph-recurrent completion, window interpolation or reference centering; S kept) and w/o OS (no learnable computation receives role, state progress or mask; static routing; constant refinement multiplier; objective unchanged) against the current final ALFNet (PEMS08 WD 1e-3), seed 32, paired initialization.

- Trained: w/o OS on all four datasets, w/o SAHC on PEMS04/PEMS08 (plan `5867446b…`); METR-LA/PEMS-BAY w/o SAHC reuse the 2026-10-06 `no_imputer` runs and 2026-10-10 test records (equivalent definition there; reuse basis recorded in the plan).
- `results_and_provenance.json`, `cells_33_long.csv`, `lifeavg_summary.csv`: Full and both variants, validation and test.
- Formatted workbook and analysis: `paper/latex_neurips_20261020/results_extension_20261103/ablation_20261103.xlsx`, `README_CN.md`.
- Implementation: `experiments/alfnet_extension_ablation_adapter.py` (study-only; production ALFNet unchanged).
