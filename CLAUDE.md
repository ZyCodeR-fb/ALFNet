# Claude Code repository entry

Read and follow, in order:

1. `docs/ai/WORKFLOW.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/ai/HANDOFF.md`
4. The active task file linked by `docs/ai/HANDOFF.md`

Use `docs/ai/DECISIONS.md` and `docs/ai/EXPERIMENTS.md` when the task depends on
historical choices or experimental evidence. Do not treat old chat, stale plans,
template files, or archived outputs as current authority without verification.

Claude-specific rules:

- Inspect relevant files and `git diff` before editing; use small, reviewable edits.
- Do not use `git checkout`, `git restore`, `git reset`, or whole-file redirection to
  erase worktree changes. Preserve user and other-agent changes.
- Do not start training, evaluation, test access, commit, or push unless the current
  task explicitly authorizes it.
