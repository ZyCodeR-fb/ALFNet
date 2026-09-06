# Shared AI workflow

This is the common operating contract for Codex and Claude Code. Root entry files
must remain thin; shared rules live here, stable facts in `PROJECT_CONTEXT.md`,
decisions in `DECISIONS.md`, evidence indexes in `EXPERIMENTS.md`, and transient
state in the active task handoff.

## 1. Establish authority before work

1. Read the two shared context files and the active task handoff.
2. Inspect branch, HEAD, upstream relation, staged/unstaged changes, and untracked
   files. Existing changes belong to the user unless proven otherwise.
3. Verify important claims against code, generated configuration, manifests, logs,
   checkpoints, or collectors. Chat and reports are discovery clues, not sufficient
   evidence by themselves.
4. Distinguish an accepted design from the implementation at the checked HEAD. If
   they disagree, record the conflict; do not rewrite the design or silently bless
   the implementation.
5. Never infer that an old tmux session, run, or test is still active. Check it and
   timestamp the observation.

## 2. Scope and change discipline

- Make only changes required by the active task. Do not resume a paused research
  direction from historical context.
- Preserve `data/*/original/`, `data_v4/`, `references/`, checkpoints, reports, and
  archived evidence unless deletion is explicitly authorized and targets are
  verified.
- Prefer focused patches. Do not use destructive Git commands or overwrite a whole
  file to undo a failed edit. Review the diff after each meaningful change.
- Do not change model architecture, data split, target channel, loss, checkpoint
  criterion, seed policy, metric arithmetic, or test-access policy as a cleanup.
  Those are research/protocol changes and require a recorded decision, a new plan
  identity, and affected-run analysis.
- A code cleanup is result-preserving only when its relevant behavior is tested or
  otherwise demonstrated. State the actual validation scope.

## 3. Protocol and fairness checks

Before training or comparing models, verify from current authorities:

- identical dataset payload, split membership, origins, masks, scaler fitting scope,
  target channel, prediction horizon, and metric implementation;
- topology and dynamic-signal exposure match the declared inductive setting;
- comparison models receive only common inputs plus inputs required by their source
  architecture; ALFNet-only lifecycle features and objectives must not leak into a
  baseline;
- source-model adaptations are documented in `docs/comparison_model_audit.md`;
- validation selects configurations/checkpoints; test remains sealed until the
  selection is frozen and explicitly authorized;
- MAE/RMSE must not be pooled across flow and speed datasets; MAPE limitations near
  zero must be disclosed rather than hidden by post-hoc changes.

Generated YAML and plans are immutable run identities, not suggestions. Run their
builders in `--check` mode. A plan that fails its code/config/data hash checks is
stale and cannot authorize reuse or new runs until deliberately regenerated.

## 4. Validation proportional to risk

- Documentation-only: verify links, placeholders, file scope, and `git diff --check`.
- Local code change: targeted tests that exercise the changed production path.
- Shared data/training/evaluation change: targeted fault tests plus the appropriate
  audit/preflight; run the full suite before formal evidence collection when feasible.
- Formal run/evaluation: record the exact plan hash, config, code commit, seeds,
  output root, commands, audit result, and whether test was accessed.

Do not describe a skipped or unrun check as passed. Small numerical differences must
be reported with tolerance and provenance, not rounded away.

## 5. Experiments and resource use

- Training, independent re-evaluation, and test opening require explicit task scope.
- Use validation-only candidates for method choice. Any reuse of checkpoints must be
  justified by semantic equivalence and recorded.
- Keep launcher logs and failed-run evidence. Do not silently replace or delete them.
- GPU concurrency is a resource choice, not a protocol fact. Confirm memory and
  throughput before increasing jobs.
- Stop or pause jobs recoverably; report checkpoint/resume status from current files.

## 6. Git and handoff

No standing authorization to commit or push is assumed. Unless the active task says
otherwise, leave reviewable changes in the worktree and report them. Never rewrite
history or force-push.

At a checkpoint, update only the documents whose facts changed:

- `PROJECT_CONTEXT.md` for stable accepted context;
- `DECISIONS.md` for a consequential accepted choice;
- `EXPERIMENTS.md` for a concise evidence index;
- the active task handoff for current state and next action.

Every handoff must separate Facts, Hypotheses, Completed, Modified but unverified,
Not started, and Ruled out. Include checked HEAD/worktree, commands actually run,
outputs, blockers, and the next discriminating action. Do not invent work merely to
keep a completed task open.
