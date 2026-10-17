# Comparison-model source and v4 adaptation audit

## 2026-09-27: flow GinAR numerical recovery

Original flow transfer LR.006 fails on PEMS03 at training batch4 (zero-based):
loss1.907654016e9 and gradient normInf. A two-batch finite-value preflight missed
this instability. The independent recovery plan uses LR.001 on all three flow
datasets, retaining architecture, objective, optimizer/scheduler, seed32 and
max100/patience10. Each dataset passed a full train epoch plus full33-cell
validation, finite gradients and matching canonical counts, before freezing.
Speed recipes remain unchanged; disclose this dataset-family LR difference.
Evidence: paper/baseline_extension_20260927/RECOVERY.md and diagnosis JSONs;
authority: configs/paper/baseline_recovery_20260927/plan.json. No test tuning.

## 2026-09-27: Merlin model adapter and GenCast feasibility

Merlin source STID teacher/student, MLP and loss files copied byte-identically
from references/Merlin; see model/comparison_model/Merlin/SOURCE_MANIFEST.json.
Adapter retains source node embeddings and four-view KD/CL, with explicit
train-K supervision/projection masking and common scaler/calendar/input handling.
CPU synthetic source forward and additive-loss parity, staged gradients, leakage
refusal and target blindness tested. No ALFNet-specific modules/objectives used.
Dedicated staged training is now implemented in scripts/train_merlin_v4.py,
using common max100/patience10 per stage and source scheduler boundaries.
Each stage has its own optimizer, scheduler, best/last checkpoints and RNG;
stage/teacher SHA enter resume identity. The model is registered for evaluation,
but generic training still refuses to discard KD/CL. Formal run authority and
actual preflight/completion state are recorded in the independent
configs/paper/baseline_extension_20260927/plan.json and task handoff.

GenCast-L/H require genuine aligned weather/position assets and speed physical
constraints; flow datasets lack positions and current common protocol has no
weather/road-description inputs. Minimal faithful adaptation is not presently
feasible across the five datasets. No fake zero weather or replacement physics;
not registered/copied as an executable adapted model. Reason recorded in
model/comparison_model/GenCast/README.md; source remains in references/GenCast.

## Current authority — 2026-09-26 integrity recheck

The executable SLPF wrapper is `source_forecast_single_stage_v1`; it does NOT
execute the historical three-stage pipeline described below. Those paragraphs
are retained historical records, superseded by the 2026-09-17 decision.
STBP is a structural-pattern offline adaptation: its source per-node expandable
pattern bank is replaced by `StructuralNodeEncoder`, in addition to omitting
continual expansion/training. Thus neither current result represents the entire
original method. This distinction is more specific than “fixed-graph adapter”.

Local source comparison and synthetic CPU parity checks found exact forward
agreement for SLPF's three cores and STBP's backbone when supplied identical
node embeddings/patterns. This does not validate training-pipeline equivalence.
Restoring complete methods is not a small interface repair under the frozen
protocol. See [integrity recheck](../paper/SLPF_STBP_INTEGRITY_20260926.md).
No runtime code, checkpoints, experiments, or evaluation settings were changed.

## Historical final-code repair — 2026-09-16 (superseded for SLPF)

Final Focal is locked; no further ALFNet optimization. See
[final code audit](../paper/final_code_audit_20260915/README.md) for exact scope,
validation artifacts, source differences and retraining requirements.

SLPF is now a three-stage source-core adapter, not the historical forecast-only
wrapper. ADP uses deterministic holdout of currently visible history; forecast
uses legal sensed targets; aggregation narrows the common cohort mask to unsensed
locations and skips batches without such supervision. Intermediate stages restore
their own validation-best states and freeze before the next stage. Resume identities,
optimizers and RNG are persisted. Final selection remains canonical33 LifeAvg MAE.
These are necessary protocol adaptations, not numerical source equivalence, and
old SLPF results remain invalid as evidence for this repaired pipeline.

STID prepared configs restore source speed-calendar settings (288 bins, weekday
enabled), source Adam/lr.002/milestones[1,50,80]/gamma.5/clip5/raw-MAE; the old
minute/1440 fix remains. Unequal tid/diw embedding widths now size the encoder
correctly. STGCN restores source Adam/raw-MAE. These repairs require new results.
Other baselines retain disclosed graph/structural-node adaptations; STBP remains
an offline fixed-graph backbone adaptation, not a full online cross-year protocol.

Prepared configs: `configs/paper/final_models_v1/`; no new training plan is yet
registered. CPU checks do not substitute for the future authorized GPU preflight.
The earlier missing-SLPF statements below are historical implementation states.

## Latest correction audit — 2026-09-15

The final nine names are UIGNN/GinAR/MoGERNN/ST-FiT/SLPF/STGCN/STID/PatchSTG/STBP.
GWNET is excluded from that shortlist but its root implementation is now unified.
Detailed dispositions: [completion audit](../paper/final_completion_20260915/README.md).
All ten inspected implementations on both speed datasets passed CPU forward,
backward, legal input isolation and Full-matched exposure/scaler checks (20 cases).
**This does not establish complete source equivalence:** SLPF remains forecast-only
and formal execution is blocked; STBP remains an offline backbone adaptation.

STID's raw minute/hour calendar is now normalized by1440, not288, before source
embedding indexing. ST-FiT uses actual augmented graph size for batch offsets,
finite min/max normalization, legally supervised VAE donors, source raw MAE/Adam,
and an equivalent batched-matrix ChebConv path. Output and all gradient classes
were checked against PyG; there is no edge pruning or architecture replacement.
PatchSTG retains source AdamW and raw MAE; calendar conversion/layout were correct.
Both new baseline recipes still use the declared common constant LR and canonical33
LifeAvg-MAE selector, not the original repositories' full training schedules.
ST-FiT source Gumbel sampling remains stochastic; patience counts forecast epochs.

Root GinAR/UIGNN now use the previously trained frozen adapters and source objectives.
GinAR scheduler steps epochs, not metric values, and binds to the restored optimizer.
GWNET root uses one official core, Conv2d compatibility conversion, shared structural
factors, Adam/raw MAE/clip5 for both datasets. Historical results are not retroactively
changed by root fixes. The old root implementation is archived.

The text below describes historical audit scopes. In particular, claims that all
historical checkpoints use the same selector are superseded: BAY STGCN/STID/SLPF/
STBP used the older micro selector and need the requested future rerun.

This file records what was compared before the unified-BASE plan was frozen.
Every baseline receives the same v4 batch, graph, split, masks, seeds,
checkpoint rule and evaluation arithmetic. A wrapper may reshape tensors; it
may not load private data or consume ALFNet lifecycle features.

| registered model | local core | checked source | retained method / necessary adaptation |
|---|---|---|---|
| HA | `HA/ha.py` | `references/ST-FiT/src/model/hi/hi.py` | historical-inertia copy; zero-output parameter anchor only permits the common optimiser API |
| FC-LSTM | `FCLSTM/lstm.py` | `references/ST-FiT/src/model/lstm/lstm.py` | LSTM/conv/head equations retained; source-private graph/data trimmer and device literals are replaced by `RuntimeDataContext`/runtime tensors |
| STGCN | `stgcn_n/*` | `references/ST-FiT/src/model/stgcn_n/*` | convolution core retained; all node views use the common full-N graph and source augmentation flags are disabled |
| STGODE | `stgode/*` | `references/ST-FiT/src/model/stgode/*` | ODE/TCN blocks and head retained; spatial matrix comes from the common graph; semantic matrix is topology-row cosine because source DTW would read held-out signals |
| GWNET | `GWNet/source_model.py`, `source_adapter.py` | `references/GWNet/model.py` | 2026-09-15: one byte-identical source core and minimal adapter for both datasets; old duplicate implementation archived; shared structural factors replace node-ID factors |
| STID | `STID/stid.py` | `references/STID/stid/arch/stid_arch.py` | temporal embeddings, MLP encoder and head retained; trainable node-ID table is replaced by a shared structural encoder |
| ST-FiT | `STFiT/*`, `stgcn_n/*` | `references/ST-FiT/src/model`, `src/trainer/train.py` | learned graph, mixup/VAE node synthesis, alternating VAE/forecast epochs and source MAE retained; train targets are visible only during its train-time synthesis, never validation/test |
| MoGERNN | `MoGERNN/*` | `references/MoGERNN/model`, `layers` | MoGE/DCRNN architecture retained; receives the same full-N masked tensor and runtime graph, with no lifecycle context |
| SLPF | `SLPF/SLPF.py` | `references/SLPF/model/SLPF.py` | core contains all stages, but historical wrapper runs forecast only; complete ADP/forecast/aggregation adaptation is NOT verified, and new formal runs fail closed until fixed |
| PatchSTG | `PatchSTG/patchstg.py` | `references/PatchSTG/models/model.py` | patch attention retained; spatial membership comes from the manifest-resolved layout rather than private coordinate loading |
| STBP | `STBP/stbp.py` | `references/STBP/src/model/model.py` | DLGA/frequency/encoder/head computation retained; graph/positions are injected, node-ID capacity is structural, and source MAE is used |
| MCPST-FSL | `MCPST_FSL/model.py` | `references/MCPST-FSL/models/model.py` | multi-scale, oscillator, spectral and uncertainty branches retained; flow MSE is lifecycle-masked and target-free physics/consistency regularisers remain active |

## Deliberate fairness adaptations

The following are not architecture claims and are shared wherever applicable:

- disk loaders, splitters, scalers and metrics from source repositories are not
  used; the v4 manifest/loader/evaluator is the sole authority;
- every model runs on the full static node axis, while unavailable dynamic
  values are mean-filled and masked identically;
- per-node trainable embeddings that would give held-out nodes learned identity
  are replaced by shared functions of the same topology/verified positions;
- source optimisation loss is declared in YAML (MAE for HA/FC-LSTM/STGCN/
  STGODE/GWNET/STID/ST-FiT/SLPF/PatchSTG/STBP; MSE for ALFNet/MoGERNN/
  MCPST-FSL), while early stopping and all reported metrics use the same v4
  definitions;
- target tensors never enter a model on validation/test. ST-FiT's use of train
  targets for source-defined synthetic-node training is explicit and tested.

The adapters were accepted only after a real PEMS08 one-batch CPU
forward/backward/parameter-update check for every registered model. The final
plan hashes every Python core and adapter listed here.

## 2026-09-13 isolated METR-LA wave screen

This supplement applies only to `outputs_v4/wave_fusion_workspace_20260913/`.
The accepted root Full and historical wrappers are preserved.

- `model/comparison_model/GWNet/source_model.py` is a byte-identical copy of
  `references/GWNet/model.py`, with the source LICENSE copied alongside it.
- `source_adapter.py` executes that source forward directly. The source's
  Conv1d objects with 2D kernels are converted to Conv2d with identical weights
  and temporal geometry for modern torch's 4D inputs. Two independent shared
  structural encoders replace node-ID factors, as required by the existing
  inductive contract. The full canonical node axis is enforced.
- The native baseline has no lifecycle features, GRIN, FiLM or ALFNet objective.
  Traffic input width is the protocol's single speed channel, rather than the
  source CLI's default two channels. Supports are doubletransition plus the
  structural adaptive graph. The source 8 gated layers, 1/2 dilation reset,
  diffusion order2, residual/skip paths, BatchNorm and output head remain.
- The runner retains the project's AdamW, normalized masked MAE and v4
  checkpoint rules. The source engine uses Adam, raw-scale MAE and gradient
  clipping5; this run does not claim bitwise reproduction of that training
  recipe or the source paper's transductive results. Source data/split/test code
  is not executed. Common training batch64/100epoch budget is frozen separately.
- ALFWave is explicitly a method variant, not the GWNET baseline. Its optional
  FiLM hook conditions the source head input after skip ReLU. This is a declared
  difference from the earlier untrained prototype's pre-ReLU conditioning.
  Hence old prototype timings are not substituted for this round's preflight.
- The lightweight temporal residual is a new ALFGraph-side adapter inspired by
  the source tanh/sigmoid temporal gates and 1/2 dilation schedule. It is not
  represented as a verbatim source module or a replacement for ALFGraph.
- Synthetic source forward and start-convolution gradient parity are exact at
  matched weights/structural factors. Native baseline and hybrid have independent
  variant/config identities. No PEMS-BAY training or test is included in this stage.

## 2026-09-15 missing inductive baselines

The new six-slot seed32 plan is `725a80ed153317b77834823ee42b696e623f65848f59357cfa5d55427a79c8bd`.
See the [frozen source/adaptation audit](../workspaces/missing_inductive_baselines_frozen_725a80ed153317b7/source_audit/ADAPTATION_AUDIT.md)
and [execution handoff](ai/handoffs/T-20260908-support-ablations/MISSING_BASELINES_EXECUTION_20260915.md).
GinAR model1 and UIGNN full-graph quantile/reconstruction adapters passed four
CPU production-path probes plus nine targeted unit tests. GCN-M's two slots are
protocol-blocked by genuine external historical-memory access beyond the common
12-step input; no fake memory or GWNET substitute is registered. The remaining
four wait persistently for the old PEMS-BAY workers, then run GPU admission and
concurrent probes before formal training. This is not a claim that six trainings
have started or that GPU preflight has already passed. Executable source is
frozen under workspaces/, outside outputs_v4; original root source is preserved.
# 2026-09-17 当前适配补充

最终九基线为UIGNN、GinAR、MoGERNN、ST-FiT、SLPF、STGCN、STID、PatchSTG、STBP。SLPF按用户明确选择，根实现为源forecast主干的单阶段适配；历史三阶段描述不再代表当前实现。STBP为固定图离线适配。ST-FiT保留源Gumbel评估采样，整理前后通过固定随机种子、归档源码配对验证；历史best不含RNG，不能宣称精确重播训练中的验证抽样。最终基线不使用ALFNet独有模块或Focal目标；先前模块隔离证据随研究档案保留。
