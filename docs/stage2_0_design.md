# Stage 2-0 — read-only design specification for strict v4

Status: **specification only.** No loader, model, config, dataset or manifest is changed by this
document. Implementation begins at Stage 2-A and not before.

This exists because v3 cannot support the claim the paper wants to make. v3 is a valid
*topology-known* retrospective; it is not a strict node-unseen experiment, and no amount of rewording
makes it one. v4 is the experiment that can be, and this is its contract.

---

## 1. The K / V / T GraphView contract

Three disjoint node sets:

| set | meaning |
|---|---|
| **K** | training-known nodes |
| **V** | validation-unseen nodes |
| **T** | outer test-unseen nodes |

**Training may access** only K's training-window signal, the induced adjacency `A[K,K]`, and `pos[K]`.
The signal, ID slot, position and incident edges of V and T must not influence the training sample
draw, the schedule, the forward pass, the loss, any gradient, or optimizer state.

**Validation may access** only the `K ∪ V` induced view. T must not influence origin selection, the
availability guard, normalisation, any prediction, any metric, or the selected epoch.

**Final test** uses only the `K ∪ T` view. Global node IDs are used outside the model, for metric
attribution only.

**No model may reload the full graph.** A model receives a `GraphView` from the loader and has no path
to `data_path`. This is the rule that v3 breaks most visibly.

Every model gets the identical graph-view capability. A baseline that can see more of the graph than
ALFNet, or less, is not a controlled comparison.

## 2. v3 invalidation matrix

Each row is an independent reason a v3 artefact cannot carry into v4.

| # | defect | what it invalidates | why |
|---|---|---|---|
| 1 | ALFTemporal loads the full adjacency/positions and builds a full-N KD layout; known inputs are scattered to full N and gathered after the backbone | ALFNet checkpoints | held-out nodes carry calendar/structural embeddings and participate in attention during training |
| 2 | training schedule uses degree/hop statistics computed on the full topology | **both** models' checkpoints | the sample schedule is a function of held-out structure |
| 3 | validation `active_nodes` is full N, and the origin guard uses a full-node finite fraction | **both** models' checkpoints | T can enter validation context and therefore the selected epoch |
| 4 | `weighted_mean` and `forward_diffusion` are the same intended formula | ALFNet only | the operator set is smaller than its label count; router usage across duplicated labels is uninterpretable |
| 5 | v4 changes data payloads, target policy, graph policy and splits | **both** models, all datasets | different inputs and protocol |

**Causality, stated precisely.** De-duplicating the operator (#4) by itself requires only **ALFNet** to
be retrained. MoGERNN and every other baseline must be retrained because of #2, #3 and #5 — the shared
GraphView, schedule, split, target and protocol changes — **not** because ALFNet changed its expert set.

v3 artefacts are retained as a topology-known retrospective. Nothing is deleted.

## 3. Data status, as measured

| dataset | signal & calendar | graph | positions |
|---|---|---|---|
| METR-LA | valid: 34272×207, speed, 2012-03-01…2012-06-27 | DCRNN canonical **directed** Gaussian road-distance graph; must not be symmetrised | real WGS84 lat/lon |
| PEMS-BAY | valid: 52116×325, speed, 2017-01-01…2017-06-30 | as above | real WGS84 lat/lon |
| PEMS08 | valid: 17856×170, 2016-07-01…2016-08-31 | **derived** symmetric RBF graph, not source-native | spectral, not physical |
| PEMS03 | **wrong dates** in `raw_data.pkl` (2016-07-01…2016-09-29; should be 2018-09-01…2018-11-30); flow-only, single channel | derived, symmetrised | degenerate spectral (308 exact unique rows, 17 at 1e-6; 8 components) |
| PEMS04 | **wrong dates** (2016-07-01…2016-08-28; should be 2018-01-01…2018-02-28) | derived, symmetrised | degenerate spectral (294 exact unique rows, 6 at 1e-6; 12 components) |

The original `.npz`/`.csv` signal and edge files show **no numeric corruption**. The fault is confined
to the derived `raw_data.pkl` timestamps and pseudo-coordinates.

**Provenance limit, stated honestly.** Local file hashes have been recorded, but this repository cannot
by itself establish that these bytes are the authoritative Caltrans distribution. The v4 manifest records
what is verifiable and marks the rest as unverified provenance rather than asserting authority.

The wrong timestamps are not cosmetic: LCBT's calendar stratification and LCE's 5/7–2/7 weighting both
key on day-of-week, and PEMS03 is off by one weekday while PEMS04 is off by three.

## 4. v4 dataset roles and target policy — frozen before any v4 outer metric

| dataset | role | target |
|---|---|---|
| **PEMS03** | **untouched confirmation** | `q` (flow) — only channel available |
| **PEMS04** | **untouched confirmation** | `q` (flow) |
| PEMS08 | development/exploratory | `q` (flow) |
| PEMS-BAY | development/exploratory | `v` (speed) |
| METR-LA | development/exploratory | `v` (speed) |

PEMS03/04/08 form the **PeMS flow family**; PEMS-BAY and METR-LA form a separate **speed family**. Raw
MAE/RMSE must never be averaged across the two families — different units, different quantity.

PEMS08's v3 result is `PEMS08-speed / custom-lifecycle-v3` and must not be compared against published
PEMS08-flow numbers.

METR-LA's v3 results are retained in full. A dataset is never dropped for performing badly.

## 5. Graph, position and operator policy

Three distinct, separately hashed views:

1. **source graph** — immutable, directed, with original edge semantics. Never symmetrised silently.
2. **propagation view** — what the model diffuses over. PeMS default is source-native directed. If a
   directed RBF is used instead, the formula, sigma and duplicate/self-loop rules are frozen before any
   result is seen.
3. **partition view** — may be undirected, used only for partitioning/layout, never for propagation.

**Duplicated intended operators are removed or redefined** before v4 training. Where no real coordinates
exist, Laplacian eigenvectors are not called coordinates; a deterministic, component-aware graph-native
partition/positional encoding is used, versioned and hashed as a model component. Sign-fixing alone is
insufficient for repeated eigenvalues. `StructuralNodeEncoder` uses its topology-only path when positions
are absent.

## 6. Splits, folds and access control

- Chronological **60/20/20**. Every window lies wholly inside its segment; boundary embargo at least the
  full trajectory span.
- Scaler fitted on training time × K only.
- At least **three pre-frozen outer spatial folds**, K/V/T disjoint in each.
- Node strata from topology and training-time statistics only: component, degree, distance-to-K,
  centrality, and training-window missingness/scale/volatility.
- Timezone and cadence policy explicit. PEMS-BAY's 2017-03-12 01:55→03:00 is a real America/Los_Angeles
  DST transition and is continuous in UTC — do not insert NaN rows or delete the window. METR-LA ships a
  nominal naive grid and keeps its original time semantics.
- Exact node IDs, origins, cohorts, graph view and hashes for PEMS03/04 are frozen before any outer
  metric is computed, with a loader guard that fails on pre-freeze access.

## 7. v4 analysis plan

Written and hashed **before** any confirmation run: endpoint families and directions, alpha, margins and
their validation-only basis, seed count, resampling design, tie/NaN rules, and the PLES joint rule.

Two standing constraints carried from v3:
- PLES contrasts are gameable, so any PLES gate is joint over the contrast **and** both absolute arms.
- Three marginal intervals do not have joint 95% coverage; a joint statement requires an
  intersection-union or joint-bootstrap rule declared in advance.

36/36 is a **development engineering goal and a pre-registered descriptive acceptance criterion**. It is
never a reason to reopen the confirmation test, swap datasets, or drop a failing result.

## 8. Stage sequence

| stage | content | gate |
|---|---|---|
| **2-A** | GraphView / loader strict isolation + six CPU counterfactual tests | all pass |
| **2-B** | ALFTemporal scope-local refactor; remove full-N scatter and full-asset reload; nested expert node-ID identity | all pass |
| **2-C** | versioned PEMS03/04 v4 payloads; PEMS08/METR-LA/PEMS-BAY v4 manifests; **the actual data repair** | manifests hashed |
| **2-D** | 60/20/20, folds, test-access guard | guard proven |
| **2-E** | full CPU regression and invariance gates | all pass |
| **3** | development-only GPU optimisation on development folds | no outer test |
| **4** | freeze gate | immutable manifest |
| **5** | one-shot PEMS03+PEMS04 confirmation | explicit authorisation |
| **6** | final tables, PLES arms, statistics, efficiency, paper wording | — |

No GPU work before every Stage 2 gate passes. No confirmation access before Stage 4 completes and is
explicitly authorised.
