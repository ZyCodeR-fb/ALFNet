# ALFNet v4 repository contract

This repository has one active experiment: ALFNet versus MoGERNN on five
traffic datasets under the v4 full-node-axis lifecycle protocol. Do not add a
v3 compatibility path, silently infer a missing configuration value, or use a
test result to tune the model.

## Preserved result and active reproduction

The completed experiment is immutable evidence:

- plan: `configs/paper/archive/v4_result_946d/plan.json`
- plan id: `946d0b450bb10e3e0e5bc5ee066694bce13b3bdcc29c1c5e5a7e0702b2f4d0c8`
- outputs: `outputs_v4/five_dataset/946d0b450bb10e3e0e5bc5ee066694bce13b3bdcc29c1c5e5a7e0702b2f4d0c8`
- seeds: 32, 42, 52
- archived table shape: 36 cells, including the derived WorstGroup row

Those files are not rewritten. The active plan is
`configs/paper/v4_five_dataset_plan.json`: a clean 30-run final LF_SUP/SUP
experiment. Old checkpoints are never relabelled as runs of the new plan. The
active table removes WorstGroup and therefore has 33 cells.

## Data and prediction targets

| dataset | N | target | channel | unit | static layout | `spa_patchsize` | largest resolved part |
|---|---:|---|---:|---|---|---:|---:|
| PEMS03 | 358 | flow (`q`) | 0 | veh/5min | component-aware graph BFS | 12 | 12 |
| PEMS04 | 307 | flow (`q`) | 0 | veh/5min | component-aware graph BFS | 10 | 10 |
| PEMS08 | 170 | flow (`q`) | 0 | veh/5min | component-aware graph BFS | 6 | 6 |
| METR-LA | 207 | speed (`v`) | 2 | mph | KD-tree on verified WGS84 | 7 | 7 |
| PEMS-BAY | 325 | speed (`v`) | 2 | mph | KD-tree on verified WGS84 | 11 | 11 |

Channel is an array index, not the number of variables. Flow and speed are
different tasks and their absolute errors must never be averaged together.

`spa_patchsize` is a padding capacity, not a temporal patch count. The active
retraining rule is derived, not hand-written: it equals the largest resolved
part. Smaller parts are padded and no part is split. This changes four ALFNet
layouts relative to archived plan 946d, so those checkpoints remain historical
evidence and are not reused by a new plan.

The temporal encoder uses `tem_patchsize=12`, `tem_patchnum=1`: one 12-step
history token per spatial part.

Only `data_v4/<dataset>/{raw_data.pkl,unknown_nodes.npy}` is consumed at
runtime. `data/<dataset>/original/` is immutable source evidence and must not be
deleted. Dataset manifests under `configs/paper/v4_data_manifests/` bind source,
payload, target, graph, layout and node identities by hash.

## Protocol

- Time split: 60% train, 20% validation, 20% test.
- Input and horizon: 12 steps each (one hour at five-minute cadence).
- Sliding-window stride: **1**. YAML is authoritative and the loader consumes
  `slide_step`; the active v4 entry point also fails closed unless it equals
  the frozen protocol constant 1.
- This is **topology-known, signal-inductive lifecycle forecasting**, not a
  strict topology-unseen claim. The full static node axis and topology are
  visible to both models in every phase.
- Dynamic signal scope: train K; validation K union V; test K union T.
- K, V and T are disjoint where required; the scaler fits train time on K only.
- Four equally sampled scenarios: zero-history, steady, join and fault.
- Join/Fault evidence length k is balanced over 1..12.
- Run seed (32/42/52) controls the frozen direct node split and independently
  hashed sub-streams for model initialisation, training schedule, shuffle,
  teacher forcing, DataLoader workers and training-time dropout/router RNG.
  The complete named seed contract is stored in the plan and checkpoints.
  Validation schedule seed is fixed at 12026 so checkpoint comparisons use one
  stable evaluation design. `lcbt_protocol_seed` does not exist.
- Teacher forcing is sample-level, training-only and mask constrained; it is
  exactly zero on validation/test.
- Checkpoint selection is validation LifeAvg MAE with strict `<`; ties keep the
  earlier epoch. The 33-cell table and selection score share one validation
  forward pass.
- The test split is never used for model selection.

## Reported table

Main rows: Unseen, Steady, Join, Fault and LifeAvg. Phase rows: Cold, Warm,
Ready, Onset, Decay and Outage. Each has MAE, RMSE and MAPE: 11 x 3 = 33 cells.

WorstGroup was `max(Unseen, Steady, Join, Fault)`, so it was not an independent
measurement and is no longer computed by active code. The archived 36-cell
JSON remains untouched for auditability.

MAPE uses `abs(error) / clamp(abs(target), 1e-6)`. It is unstable at low flow;
interpret it beside MAE/RMSE and the target-scale distribution, never alone.
The optional low-flow diagnostic additionally reports fixed target bins,
WMAPE, signed bias, overprediction rate and horizon breakdown. These fields do
not enter early stopping or the 33-cell headline table; sMAPE is not reported.

## Model architecture

Both models receive the exact same full-N `base_visible_input`, availability
masks, time marks and adjacency. A shared GRIN-style imputer produces one
`filled` tensor before ALFNet branches; ALFGraph and ALFTemporal both consume
that same tensor.

ALFNet adds only its declared method-specific lifecycle context (role one-hot, state age and
observation mask). ALFGraph applies lifecycle-conditioned graph propagation.
ALFTemporal applies the resolved static spatial layout plus the one-token
12-step temporal encoder. Their features are fused by FiLM and passed through
the graph refiner. With the resolved support inventory, ALFNet has 1,037,529
parameters on the three flow graphs and 1,112,937 on the two directed speed
graphs; MoGERNN has 226,270 on every dataset.

Archived plan 946d exposed five labels but fewer numerically distinct
propagation operators:

- directed weighted METR-LA/PEMS-BAY: 4 effective supports because weighted
  mean and forward diffusion are the same row-normalised operator;
- binary/symmetric PEMS03/PEMS04/PEMS08: 3 effective supports because weighted
  mean equals forward diffusion and forward equals backward.

The final SUP design replaces that inventory with ordered candidates
`I, binary-mean, forward-1, backward-1, forward-2, backward-2`, then removes
numerically duplicated matrices per dataset before constructing the router.
The existing soft lifecycle router and its load-balance auxiliary remain; no
second evidence router exists.

LF adds a small safe-relative term in
original units, role-balanced exactly like the existing ALFNet objective, with
a denominator floor fitted as the positive-target q10 on train-time known
nodes only. The bounded PEMS08/seed-32 validation study selected LF_SUP without
opening test (candidate plan SHA
`8f8a48a949f442a2ee7fd8df3f0502077219325f75272648c4d085e7abe8b0cd`).
Therefore PEMS03/04/08 use SUP plus LF weight 0.05; METR-LA/PEMS-BAY use SUP
with LF weight 0 in the current final plan.

Before launching that 30-run plan, one bounded exploratory study tests whether
the same LF_SUP recipe should also be used on speed. It contains exactly five
METR-LA seed-32 validation-only runs: BASE, LF, SUP, LF_SUP and the unchanged
MoGERNN control. Its configs live only under
`configs/paper/v4_candidates/METRLA/`; its plan forbids test access and freezes
the adoption gates before training. LF on speed is permitted only by the two
explicit candidate flags, so no final speed config can enable it accidentally.
After the validation decision, regenerate the final five-dataset plan with the
selected single recipe before any 30-run training.

MoGERNN is the sole headline baseline. It receives the common inputs but not
ALFNet's method-specific lifecycle context.

## Mechanism diagnostics

`scripts/eval_v4_mechanisms.py` is evaluator-only:

- k-study reports Join and Fault absolute errors for every k=1..12;
- PLES pairs Join with Unseen and Fault with Steady at the same origin/cohort;
- Outage is Fault minus Steady at k=12;
- every contrast is reported with treated and control absolute errors;
- it is descriptive, non-causal and forbidden for checkpoint/model selection.

## Reproduction commands

Activate `ReaCH` first.

```bash
python scripts/build_v4_configs.py --check
python scripts/build_v4_five_dataset_plan.py --check
bash scripts/run_v4_five_dataset.sh --dry-run
ALLOW_GPU_SHARING=1 JOBS=5 DEVICES=0,0,0,0,0 \
  bash scripts/tmux_v4_five_dataset.sh start
bash scripts/tmux_v4_five_dataset.sh status
```

After all 30 final runs are complete, validation audit/evaluation/collection
and the explicit test evaluator are the only supported reporting path. The
bounded METR-LA candidate launcher is permitted only before this final freeze;
do not call legacy v3 or archived evaluation tools.

## Editing rules

- YAML and manifests are authorities; runtime code validates and consumes them.
- A config/runtime disagreement is an error, never an undocumented override.
- Preserve `references/`, every `data/<dataset>/original/`, `data_v4/`, and the
  archived 946d result artifacts.
- No architecture, loss, optimizer, graph, split, target or patch-capacity
  change is a cleanup. Such a change needs a new plan and full retraining.
- Use targeted tests during editing. Run the full CPU suite once at the final
  boundary, not repeatedly after every small patch.
