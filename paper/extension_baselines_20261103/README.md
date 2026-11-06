# Extension comparison methods (2026-11-03)

Merlin, HD-TTS (incomplete observations) and STONE, STOP (network evolution / fixed-parameter generalization), adapted under the common v4 protocol; seed 32, single run each, four datasets.

- Training plan `configs/paper/extension_20261103/plan.json` (`5867446b…`), test plan `configs/paper/extension_test_20261103/plan.json` (`4742c349…`); validation 33-cell LifeAvg MAE selection, one test pass per checkpoint, validation replay 16/16 exact.
- `results_and_provenance.json`: per-run config/checkpoint hashes, best epochs, test records; validation and test 33-cell tables for the four new methods merged with the existing eight baselines and ALFNet (read only from `paper/pems08_wd_test_20261012/selected_tables.json`); ALFNet first-place counts for every comparison set.
- `cells_33_long.csv`, `lifeavg_summary.csv`: long tables (validation and test).
- Formatted workbook, counts and Chinese analysis: `paper/latex_neurips_20261020/results_extension_20261103/` (`comparison_20261103.xlsx`, `README_CN.md`).
- Adaptation differences: `docs/comparison_model_audit.md`, section 2026-11-03.
