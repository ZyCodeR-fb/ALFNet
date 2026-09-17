# METR-LA paired observation-support evaluation

This directory contains checkpoint inference, not training or headline LifeAvg
evaluation. The split is validation. The old Full and ablation results are not
overwritten. `COMPLETE` and `verification.json` indicate execution and numerical
verification respectively; they do not establish statistical significance.

## Common matched population

128 source windows were selected before observing predictions: 64 from the
existing Join schedule (validation-held-out nodes) and 64 from Fault (known
nodes). A seeded random schedule order and random eligible focal cohort node are
used. There are 128 distinct history origins and 76 distinct focal nodes.
Eligibility requires all 12 historical readings and all 12 future targets to be
naturally available. This restriction makes every requested q feasible on an
identical population. It excludes naturally incomplete windows from this
controlled analysis; it is not a claim about their performance.

`origin` in the CSV follows the existing loader: it is the history-window start.
The prediction starts at `origin + 12`. For every focal node/window, the future
targets, calendar, graph, input/target scope, target cohort and all nonfocal input
columns are fixed. Only focal input availability and legally corresponding
metadata/decoder availability change. Reconstruction is recomputed by the normal
model forward; reconstructed intermediate states elsewhere may consequently
change through model propagation.

## Trajectory intervention

The six arms are full history, Join with 3 or 6 post-join steps, Fault after 3 or
6 interrupted steps, and full-window outage. Roles and age are generated with
the existing lifecycle loader. The same focal nodes receive every arm; source
track labels identify the original known/held-out population, not an immutable
intervention role. Join-6 and Fault-6 share q=6 but differ in both observation
position and legal event metadata. This contrast alone does not isolate timing
from metadata.

## q × temporal layout

q is the actual number of available observations in the 12-step input window.
Levels are 0, 3, 6, 9 and 12. Recent keeps the last q positions; early keeps the
first q; dispersed selects floor((j+0.5)*12/q), j=0,...,q-1; middle_gap removes
one centered block of length 12-q starting at floor(q/2).

This experiment represents communication-level deletion at a mature active
sensor: roles remain AAS and ages remain 1. Deleting an observation is not
relabeled as a new sensor or physical failure. Every requested mask intersects
natural availability. q=0 and q=12 have one unique arm each; their repeated cells
in the heatmap are common anchors, not independent replications.

## Files and arithmetic

- `pairs.json`: selection before inference; no error-based filtering.
- `manifest.json`, `sources.json`: protocol and reused checkpoint identities.
- `*_paired.csv`, `paired_predictions.csv`: each focal-node forecast, target,
  support mask and three error metrics; 12 horizons per paired forecast.
- `paired_contrasts.csv`: per-pair MAE difference from MoGERNN and change from the
  same model's full-history arm. Positive paired_MAE_gain favors that model.
- `metrics.csv`: all three metrics, both pooled and source-track-specific.
- `relative_gains.csv`: three-metric relative gains against MoGERNN.
- `verification.json`: checks source identity, matched target/input masks,
  recomputed metrics, and identical predictions for the two full-history anchors.

Main figures average the 128 matched windows equally. RMSE is the square root of
the pooled squared error over windows and horizons; this differs from the
condition-balanced LifeAvg RMSE. Relative gain is 100*(baseline error-model
error)/baseline error on the same paired sample population, not the mean of
per-sample ratios. No population-level confidence or multi-seed significance is
claimed. Full, MoGERNN and Mask-only use the same samples; no test evaluation is
performed.

## Reproduction

From the repository root, use the verified ReaCH Python environment:

```bash
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/analyze_observation_support.py --output outputs_v4/reports/observation_support_reproduction --per-track 64 --batch-size 16
/home/xd/anaconda3/envs/ReaCH/bin/python -m scripts.verify_observation_support outputs_v4/reports/observation_support_reproduction
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/plot_observation_support.py --data outputs_v4/reports/observation_support_reproduction
```

The evaluator refuses to overwrite an existing output directory. Figures in
`paper/figures/` are regenerated from CSV/case records. The existing condition CSV
and case records in `support_analysis_v1` remain the source for the condition and
appendix case figures.
