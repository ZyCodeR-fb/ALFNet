# Active task routing

| Task ID | Status | Handoff | Summary |
|---|---|---|---|
| `T-20261103-extension-experiments` | RUNNING (2026-11-06): batch 1 results ready (results_extension_20261103/; STOP beats ALFNet on 3 datasets, author decision pending); batch 2 MCPST/MetaDG trained, test running; then auto chain `alfnet_ext_seeds_chain`: 3-seed training (seeds 42/52, 80 runs, ≤4 concurrent) + one test each for the 10 comparison methods (ALFNet seeds not yet) | [task handoff](handoffs/T-20261103-extension-experiments/HANDOFF.md) | Server-side supplementary comparison and ablation experiments; training and one test per final checkpoint authorized 2026-11-03 for these runs only; max 3 concurrent; no commit/push; manuscript unchanged. |
| `T-20261020-paper-writing` | ACTIVE: method chapter finalized; Figure 1 being redrawn by author; next = paper optimization per NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md (literature survey, baseline criteria, experiment gaps, then intro/related work); experiments pending advisor decision | [task handoff](handoffs/T-20261020-paper-writing/HANDOFF.md) | ARS paper completion (method/abstract/conclusion → NeurIPS LaTeX → simulated review → revision). Writing only; no training/evaluation/test access/commit. |
| `T-20260908-support-ablations` | COMPLETE: advisor-accepted experiments and repository cleanup | [task handoff](handoffs/T-20260908-support-ablations/HANDOFF.md) | Final architectures and587 checkpoints retained;222 obsolete/excluded weights removed with inventory. Sources/evidence archived, research lessons and current project index written;62 targeted tests passed. |

The research handoff is complete; the active task is paper writing only. The
shared-context initialization itself is a completed documentation operation, not a
research task.
