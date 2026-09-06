# Decision log

Only consequential, still-relevant decisions belong here. Evidence paths are links;
implementation status is stated separately from the accepted intent.

## D-20260906-01 — Shared AI context structure

- **Status:** accepted and implemented for repository guidance.
- **Decision:** keep thin `AGENTS.md` and `CLAUDE.md` entries; maintain shared workflow,
  stable context, decisions, experiments, and task handoffs under `docs/ai/`.
- **Reason:** prevent Codex and Claude from carrying divergent copies of project rules
  and from treating old chat as current authority.
- **Scope:** AI collaboration documentation only; no scientific code or result changed.

## D-20260906-02 — Common inductive contract, model-specific innovation

- **Status:** accepted design; implemented in the v4 data/runner paths at the checked
  HEAD, subject to plan re-freezing.
- **Decision:** all methods share dataset identity, exposure masks, split, scaler,
  origins, evaluation arithmetic, and checkpoint-selection rule. A baseline may keep
  architecture/loss behavior required by its source, but must not receive ALFNet-only
  lifecycle inputs or objectives.
- **Reason:** comparable evidence requires equal observable data, not identical model
  internals.
- **Source:** `lib/lcbt_v4.py`, `lib/runtime_context.py`,
  `experiments/v4_runner.py`, and `docs/comparison_model_audit.md`.

## D-20260906-03 — Current ALF choice is unified_BASE / AR_BASE

- **Status:** accepted as the current ALF baseline for the latest seed32 comparisons.
- **Decision:** LF, SUP, direct-head, masking/reconstruction, and other candidates are
  not selected merely because code or old outputs remain. Current follow-up starts
  from unified_BASE with the AR_BASE recipe unless a new validation-only decision
  replaces it.
- **Reason:** prior candidate work did not establish a uniformly superior replacement;
  preserving a reproducible baseline takes precedence over silent retuning.
- **Scope:** does not validate or authorize the stale 13-model final plan.

## D-20260906-04 — Nine-model reporting shortlist

- **Status:** accepted user requirement for the current reporting task; not yet encoded
  in a new formal multi-seed plan.
- **Decision:** report ALFNet, MoGERNN, HA, FCLSTM, STGCN, STGODE, STID, SLPF, and STBP.
- **Alternative replaced:** the older 13-model plan remains historical evidence but is
  not the current paper shortlist.

## D-20260906-05 — METR-LA patch 8 is exploratory

- **Status:** accepted only for the completed METR-LA seed32 validation experiment.
- **Decision:** allow patch capacity 8 for that experiment because it uses one more
  partition than the resolved maximum of 7; do not describe it as the final uniform
  five-dataset setting without a new decision and frozen plan.
- **Reason:** the choice was informed by validation history and therefore requires
  explicit provenance and further confirmation rather than retroactive generalization.
