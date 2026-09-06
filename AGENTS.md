# Codex repository entry

Read and follow, in order:

1. `docs/ai/WORKFLOW.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/ai/HANDOFF.md`
4. The active task file linked by `docs/ai/HANDOFF.md`

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
