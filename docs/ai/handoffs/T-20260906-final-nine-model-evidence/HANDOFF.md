# T-20260906-final-nine-model-evidence

- **Status:** paused; context migration complete and ready for a new session.
- **Objective:** turn the latest accepted nine-model/five-dataset comparison work into
  a reproducible evidence package and decide whether METR-LA remains in the final
  study, without treating exploratory validation as formal multi-seed/test evidence.
- **Repository:** `/media/xd/38162e9d-06bd-4953-8790-0bc340486a80/ZMY/MoGERNN`.
- **Checked repository version:** branch `paper/final`, HEAD
  `cf8f259ebc7fcc4e51d9ae362cf14443ca560191`.

## Facts

- The latest user-selected reporting set is ALFNet, MoGERNN, HA, FCLSTM, STGCN,
  STGODE, STID, SLPF, and STBP across PEMS03, PEMS04, PEMS08, METR-LA, PEMS-BAY.
- `scripts/build_final_configs.py --check` passed with 75 generated YAMLs.
- `scripts/build_final_plan.py --check` failed at this HEAD because seven hashed code
  files have changed. `configs/paper/final_plan.json` also names 13 models, so it is
  a historical run identity, not the current nine-model authority.
- `scripts/build_metrla_seed32_nine_models.py --check` passed for plan hash
  `057073f745ba24710cc4849abe91cc65c00b13ce4136337eb67563f60da35ff1`.
  Its nine METR-LA seed32 runs are complete and validation-only. Eight independent
  evaluations agree exactly; STGCN exceeds its declared `1e-6` agreement tolerance by
  a small amount (maximum absolute difference `7.081064183012131e-06`).
- The five-dataset workbook under experiment E-5D-9M-SEED32-MERGED explicitly declares
  mixed-source, validation-only provenance. It cannot be relabeled as a formal
  three-seed or test table.
- The METR-LA candidate used ALF patch capacity 8 while its manifest-resolved maximum
  partition capacity is 7. This is an exploratory validation choice, not a frozen
  cross-dataset policy.
- At 2026-09-06 15:29:32 +08:00, no GPU process from this repository was active.

## Hypotheses

- METR-LA seed32 performance may justify retaining the dataset, but persistence over
  seeds 42/52 and test is unknown. The next discriminating evidence is a frozen-plan
  validation run, not immediate test access.
- The STGCN re-evaluation discrepancy may be benign floating-point nondeterminism, but
  its cause has not been isolated. Changing tolerance without reproducing and tracing
  it would hide rather than resolve the issue.

## Completed

- METR-LA seed32 training and validation re-evaluation for the nine-model shortlist.
- A five-dataset seed32 validation workbook with explicit mixed provenance.
- Historical four-dataset, multi-seed validation/test evidence remains indexed rather
  than rewritten.
- Shared Codex/Claude context initialization and this checkpoint.

## Modified but unverified

- Current committed model/runner code differs from the hash set frozen by the old
  final plan. This checkpoint did not run training, independent evaluation, or the
  full test suite; only the three builder checks named below and documentation checks
  are in scope.
- Candidate paths and switches remain in current code. Their exact reachability from
  each generated BASE YAML has not been re-audited in this documentation task.

## Not started

- A new nine-model, five-dataset formal plan bound to current code/config/data hashes.
- A decision and, if approved, validation runs for METR-LA seeds 42 and 52 under the
  frozen patch policy.
- Any new test opening, final paper-table replacement, or paper edit.
- Canonical environment reconciliation between `environment.yml` (`moge`) and the
  installed `ReaCH` environment used by recent checks.

## Ruled out

- Do not use the stale 13-model `final_plan.json` to launch or reuse new formal runs.
- Do not call the mixed-source seed32 workbook a formal final table.
- Do not use test results to decide model, patch, dataset inclusion, or configuration.
- Do not resume candidate architecture optimization merely because its scripts or
  outputs remain.

## Recommended next action

First ask the user to confirm two linked decisions: whether METR-LA is retained and
whether patch 8 is the frozen METR-LA setting for confirmation. Then create/check a
new nine-model plan at the current intended HEAD. Before launching anything, resolve
the STGCN validation mismatch with a deterministic production-path rerun and verify
that BASE YAMLs disable rejected candidate switches. If the plan and audit pass, run
remaining validation seeds. Open test only after the validation decision is frozen
and the user explicitly authorizes it.

## Checkpoint commands and evidence

Actually run during initialization with
`/home/xd/anaconda3/envs/ReaCH/bin/python`:

```bash
python scripts/build_final_configs.py --check
python scripts/build_final_plan.py --check
python scripts/build_metrla_seed32_nine_models.py --check
```

The first and third passed; the second failed for code-hash drift. Git began clean
apart from untracked `_ai_memory_templates/`. After migration, the unstaged changes
are modified `CLAUDE.md` and new `AGENTS.md`, `docs/ai/`, and
`_ai_context_backups/`; the template directory remains untracked. The staging area is
empty. No scientific source, configuration, paper, dataset, or experiment artifact
was changed. No commit or push was authorized or performed.

Documentation verification actually run: `git diff --check`; exact SHA-256 comparison
of the old Claude entry backup; existence checks for every formal context file and
indexed evidence path; root-to-task routing checks; and placeholder scans for example
task IDs/template markers. All passed. No pytest, model execution, training,
independent evaluation, or test access was performed for this migration.

Relevant evidence paths and their limitations are catalogued in
`docs/ai/EXPERIMENTS.md`.
