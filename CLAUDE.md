# Claude Code repository entry

Read and follow, in order:

1. `docs/ai/WORKFLOW.md`
2. `docs/ai/CLAUDE_CODE.md` (Claude skill routing, Codex handoff and review-response rules; imported below)
3. `docs/ai/PROJECT_CONTEXT.md`
4. `docs/ai/HANDOFF.md`
5. The active task file linked by `docs/ai/HANDOFF.md`, as relevant to the current request

The current user request selects the task; historical handoffs do not authorize
resuming research or experiments. Apply `docs/ai/CLAUDE_CODE.md` automatically; the
user does not need to name a skill.

@docs/ai/CLAUDE_CODE.md

Use `docs/ai/DECISIONS.md` and `docs/ai/EXPERIMENTS.md` when the task depends on
historical choices or experimental evidence. Do not treat old chat, stale plans,
template files, or archived outputs as current authority without verification.

Claude-specific rules:

- Inspect relevant files and `git diff` before editing; use small, reviewable edits.
- Do not use `git checkout`, `git restore`, `git reset`, or whole-file redirection to
  erase worktree changes. Preserve user and other-agent changes.
- Do not start training, evaluation, test access, commit, or push unless the current
  task explicitly authorizes it.
