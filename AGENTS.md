# Codex repository entry

Read and follow, in order:

1. `docs/ai/WORKFLOW.md`
2. `docs/ai/CODEX.md` (Codex skill routing, independent review, and figure preferences)
3. `docs/ai/PROJECT_CONTEXT.md`
4. `docs/ai/HANDOFF.md`
5. The active task file linked by `docs/ai/HANDOFF.md`, as relevant to the current request

The current user request selects the task; historical handoffs do not authorize
resuming research or experiments. Apply `docs/ai/CODEX.md` automatically; the user
does not need to name a skill.

Consult `docs/ai/DECISIONS.md` and `docs/ai/EXPERIMENTS.md` only as needed.
Archived entries and `_ai_memory_templates/` are not active instructions.

Codex-specific rules:

- Use `rg`/`rg --files` for discovery and `apply_patch` for text edits.
- Inspect `git status` and the relevant diff before editing; never discard unrelated
  worktree changes.
- Use the repository's verified Python environment for commands; the current
  `environment.yml` versus installed `ReaCH` environment mismatch is still pending
  verification.
- Do not start GPU work, open test data, commit, or push unless explicitly authorized
  by the current task.
