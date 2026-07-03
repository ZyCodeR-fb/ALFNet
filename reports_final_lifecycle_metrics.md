# Final Lifecycle Metrics

Metric convention: lower is better. `RoleMacro` is the mean of `VS/AAS/NAS/FS` MAPE. The six sensor states are reported as `Cold-start`, `Warm-up`, `Adapted`, `Degrading`, `Outage`, and `Steady`.

## Current Selection

Current final model: `rf_sub_sd`.

Reproducibility entry after model-name isolation: `configs/ALFNet/rf_alf_sd.yaml`. It keeps the same final structure and hyperparameters as `rf_sub_sd`, but loads the frozen paper-facing expert copies `ALFGraph` and `ALFTemporal` instead of the mutable baseline names `MoGERNN` and `PatchSTG`. Use this entry for future clean-code reruns once `MoGERNN` and `PatchSTG` are restored for original-baseline comparison.

Rationale: among the subtractive experiments, `rf_sub_sd` gives the best RoleMacro MAPE and keeps the model story simple: MoGE expert slimming plus residual-delta decoder. It is also slightly better than the previous final `rf_s_plp2grintftop065r003vr` on RoleMacro.

| Model | VS MAPE | AAS MAPE | NAS MAPE | FS MAPE | RoleMacro MAPE |
|---|---:|---:|---:|---:|---:|
| MoGERNN baseline | 15.6900 | 10.5900 | 9.8300 | 13.2600 | 12.3400 |
| Previous final `rf_s_plp2grintftop065r003vr` | 15.9901 | 9.1792 | 10.1130 | 9.9785 | 11.3152 |
| Final subtractive `rf_sub_sd` | 15.7572 | 9.2546 | 10.1615 | 9.9384 | 11.2779 |

## Subtractive Ablation Results

| Variant | Main Change | VS MAPE | AAS MAPE | NAS MAPE | FS MAPE | RoleMacro MAPE | AvailabilityMacro MAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| `rf_sub_sd` | MoGE slimming + residual-delta decoder | 15.7572 | 9.2546 | 10.1615 | 9.9384 | 11.2779 | 11.4776 |
| `rf_sub_delta` | residual-delta decoder | 15.8805 | 9.3182 | 10.0152 | 10.0072 | 11.3053 | 11.5538 |
| `rf_sub_prune` | visibility token pruning | 15.7974 | 9.4148 | 10.2426 | 10.8114 | 11.5665 | 11.8189 |
| `rf_sub_topo` | topology-balanced patch layout | 16.1519 | 9.4542 | 10.1797 | 10.7070 | 11.6232 | 11.8783 |
| `rf_sub_slim` | MoGE expert slimming | 17.0602 | 9.2403 | 10.4169 | 10.0176 | 11.6838 | 11.8259 |
| `rf_sub_tp` | topology layout + pruning | 16.0729 | 9.7680 | 9.9997 | 11.7463 | 11.8967 | 12.1962 |
| `rf_sub_all` | all four subtractive changes | 15.8139 | 9.5782 | 11.1609 | 11.8740 | 12.1067 | 12.4810 |

## Final Six-Sensor Metrics

MoGERNN six-sensor baseline is not available in the current local records. Do not claim improvement over MoGERNN on the six-sensor table until MoGERNN is re-tested with the updated async output or its `async_lifecycle_summary.json` is provided.

| Sensor | Model | MAPE | MAE | RMSE |
|---|---|---:|---:|---:|
| Cold-start | MoGERNN baseline | TBD | TBD | TBD |
| Cold-start | Previous final `rf_s_plp2grintftop065r003vr` | 15.8776 | 6.1194 | 9.5369 |
| Cold-start | Final `rf_sub_sd` | 15.6434 | 6.0364 | 9.3561 |
| Warm-up | MoGERNN baseline | TBD | TBD | TBD |
| Warm-up | Previous final `rf_s_plp2grintftop065r003vr` | 11.3577 | 4.0602 | 6.9900 |
| Warm-up | Final `rf_sub_sd` | 12.3497 | 4.2674 | 7.4375 |
| Adapted | MoGERNN baseline | TBD | TBD | TBD |
| Adapted | Previous final `rf_s_plp2grintftop065r003vr` | 10.0339 | 3.7155 | 6.4025 |
| Adapted | Final `rf_sub_sd` | 10.0554 | 3.8091 | 6.4113 |
| Degrading | MoGERNN baseline | TBD | TBD | TBD |
| Degrading | Previous final `rf_s_plp2grintftop065r003vr` | 12.4378 | 4.2701 | 8.2717 |
| Degrading | Final `rf_sub_sd` | 11.6623 | 3.9446 | 7.4375 |
| Outage | MoGERNN baseline | TBD | TBD | TBD |
| Outage | Previous final `rf_s_plp2grintftop065r003vr` | 10.0186 | 3.5367 | 6.8663 |
| Outage | Final `rf_sub_sd` | 10.0030 | 3.5022 | 6.7243 |
| Steady | MoGERNN baseline | TBD | TBD | TBD |
| Steady | Previous final `rf_s_plp2grintftop065r003vr` | 9.0801 | 3.2874 | 6.1898 |
| Steady | Final `rf_sub_sd` | 9.1516 | 3.2884 | 6.1507 |

## Best Subtractive Run By Sensor Metric

This table is diagnostic only. It is not a single deployable model.

| Sensor | Best MAPE | Run | Best MAE | Run | Best RMSE | Run |
|---|---:|---|---:|---|---:|---|
| Cold-start | 15.6434 | `rf_sub_sd` | 6.0364 | `rf_sub_sd` | 9.2948 | `rf_sub_prune` |
| Warm-up | 11.5762 | `rf_sub_tp` | 4.0668 | `rf_sub_tp` | 7.1204 | `rf_sub_tp` |
| Adapted | 9.8779 | `rf_sub_delta` | 3.6905 | `rf_sub_topo` | 6.3062 | `rf_sub_delta` |
| Degrading | 11.6623 | `rf_sub_sd` | 3.9446 | `rf_sub_sd` | 7.4375 | `rf_sub_sd` |
| Outage | 10.0030 | `rf_sub_sd` | 3.5022 | `rf_sub_sd` | 6.7230 | `rf_sub_delta` |
| Steady | 9.1398 | `rf_sub_slim` | 3.2621 | `rf_sub_slim` | 6.1507 | `rf_sub_sd` |

## Notes For Next Baseline Table

To complete the paper-style table, re-test original MoGERNN with the same async lifecycle setting and collect the six sensor metrics. The target comparison table should use the six short sensor names above and report `MAPE/MAE/RMSE` for each model.

## Fair Async Lifecycle Baseline Protocol

New baseline reruns should use `METRLA_async_lifecycle_100.yaml` under each model config directory. These files set `max_epoch=100`, `patience=10`, `train_for_async_lifecycle=true`, `lifecycle_training_strategy=round3_event_only`, and the same async test setting `events=4/k=12/seed=0`.

The baseline configs intentionally keep `lifecycle_role_features=false` and `lifecycle_history_age_feature=false` so the original comparison backbones are not structurally changed by extra role/age channels. ALFNet keeps lifecycle role/age input as part of the proposed method.

Configs prepared:
`MoGERNN`, `PatchSTG`, `MCPST_FSL`, `FCLSTM`, `GWNET`, `HA`, `SLPF`, `STBP`, `STFiT`, `STGCN`, `STGODE`, and `STID`.
