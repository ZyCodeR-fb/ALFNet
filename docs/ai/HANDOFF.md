# Active task routing

| Task ID | Status | Handoff | Summary |
|---|---|---|---|
| `T-20261103-extension-experiments` | RESULTS READY (2026-11-12): 3-seed study of ALFNet + 12 comparison methods complete and verified (plans d6f66d20 + recovery 694a8c24, test d7dda3a8); ALFNet test first cells mean-vs-mean 33/28/30/30 = 121/132 (all ≥27), LifeAvg first on all 4 datasets; per-seed and seeds-42/52-only counts fall below 27 on 2–3 datasets; results in paper/latex_neurips_20261020/results_seeds2_20261110/ | [task handoff](handoffs/T-20261103-extension-experiments/HANDOFF.md) | Server-side supplementary comparison and ablation experiments; training and one test per final checkpoint authorized for these runs only; max 5 concurrent (2026-11-10); no commit/push; manuscript unchanged. |
| `T-20261020-paper-writing` | ACTIVE: method chapter finalized; Figure 1 being redrawn by author; next = paper optimization per NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md (literature survey, baseline criteria, experiment gaps, then intro/related work); experiments pending advisor decision | [task handoff](handoffs/T-20261020-paper-writing/HANDOFF.md) | ARS paper completion (method/abstract/conclusion → NeurIPS LaTeX → simulated review → revision). Writing only; no training/evaluation/test access/commit. |
| `T-20260908-support-ablations` | COMPLETE: advisor-accepted experiments and repository cleanup | [task handoff](handoffs/T-20260908-support-ablations/HANDOFF.md) | Final architectures and587 checkpoints retained;222 obsolete/excluded weights removed with inventory. Sources/evidence archived, research lessons and current project index written;62 targeted tests passed. |

The research handoff is complete; the active task is paper writing only. The
shared-context initialization itself is a completed documentation operation, not a
research task.
