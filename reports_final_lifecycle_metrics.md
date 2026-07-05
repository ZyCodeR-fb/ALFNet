# Final Lifecycle Metrics

Metric convention: lower is better. `RoleMacro` is the mean of `VS/AAS/NAS/FS` MAPE. The six sensor states are reported as `Cold-start`, `Warm-up`, `Adapted`, `Degrading`, `Outage`, and `Steady`.

## Current Selection

Current final model: `ALFNet/rf_alf_sd`.

Reproducibility entry: `configs/ALFNet/rf_alf_sd.yaml`. It keeps the final subtractive structure and loads the paper-facing expert copies `ALFGraph` and `ALFTemporal` instead of the baseline names `MoGERNN` and `PatchSTG`.

Rationale: in the latest fair async-lifecycle rerun, `ALFNet/rf_alf_sd` achieves the best RoleMacro MAPE among all compared models while keeping the final method fixed and reproducible.

| Model | VS MAPE | AAS MAPE | NAS MAPE | FS MAPE | RoleMacro MAPE | AvailabilityMacro MAPE |
|---|---:|---:|---:|---:|---:|---:|
| MoGERNN | 16.1936 | 10.7584 | 10.3674 | 13.3324 | 12.6629 | 13.1329 |
| PatchSTG | 18.9494 | 9.7702 | 12.2808 | 11.1202 | 13.0302 | 12.9066 |
| MCPST-FSL | 20.5510 | 11.6121 | 10.3523 | 17.2948 | 14.9526 | 15.5645 |
| STGODE | 18.1941 | 12.5081 | 11.0029 | 18.3536 | 15.0147 | 15.6567 |
| STID | 26.2253 | 10.3439 | 11.3681 | 18.8221 | 16.6899 | 16.5683 |
| STGCN | 21.8276 | 13.4656 | 13.1794 | 19.3856 | 16.9645 | 18.5237 |
| STBP | 32.9664 | 10.7639 | 15.7531 | 13.7169 | 18.3001 | 17.6048 |
| SLPF | 28.7382 | 11.6053 | 22.7713 | 11.7983 | 18.7283 | 18.7832 |
| STFiT | 25.6467 | 12.8769 | 10.9427 | 26.9178 | 19.0960 | 20.3511 |
| FCLSTM | 28.2361 | 12.6307 | 11.0028 | 31.1963 | 20.7665 | 21.8237 |
| GWNET | 19.8382 | 12.0414 | 12.1132 | 15.5388 | 14.8829 | 15.3844 |
| HA | 28.2764 | 14.8849 | 12.6929 | 31.2096 | 21.7659 | 22.8078 |
| ALFNet (ours) | 15.6285 | 9.2521 | 10.0330 | 9.9205 | 11.2085 | 11.5066 |

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

## Source-Inspired Replacement Ablation Results

These runs test lightweight replacements inspired by external backbones and local reference code. They are diagnostic and are not selected as the final model because none improves over `ALFNet/rf_alf_sd`.

| Variant | Main Change | Params | VS MAPE | AAS MAPE | NAS MAPE | FS MAPE | RoleMacro MAPE | AvailabilityMacro MAPE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rf_x_pb` | spectral layout + pattern-bank patch mixer | 4,831,519 | 16.1345 | 9.5442 | 10.2196 | 10.7776 | 11.6690 | 11.6568 |
| `rf_x_agset` | adaptive graph support + Set Transformer patch mixer | 5,351,631 | 16.6403 | 9.5561 | 10.1143 | 10.5995 | 11.7275 | 11.9293 |
| `rf_x_set` | spectral layout + Set Transformer patch mixer | 5,348,319 | 16.8075 | 9.5186 | 10.8399 | 10.6202 | 11.9466 | 12.0161 |
| `rf_x_tl` | spectral layout + TokenLearner patch mixer | 4,744,839 | 16.7268 | 9.6556 | 10.9591 | 10.5320 | 11.9684 | 12.2248 |
| `rf_x_ag` | adaptive graph support | 4,850,111 | 17.8358 | 9.2417 | 11.2418 | 10.1400 | 12.1148 | 12.1349 |
| `rf_x_agpb` | adaptive graph support + pattern-bank patch mixer | 4,834,831 | 16.1423 | 9.4651 | 12.3081 | 10.6888 | 12.1511 | 13.0762 |
| `rf_x_agtl` | adaptive graph support + TokenLearner patch mixer | 4,748,151 | 18.0456 | 9.9333 | 11.3921 | 12.4708 | 12.9604 | 14.3717 |

Diagnostic summary: `rf_x_pb` is the best source-inspired variant by RoleMacro and gives the cleanest pattern-bank story, but it is still worse than the selected final model. Adaptive graph support is unstable in combination runs: it improves some failed-sensor states in isolation but hurts cold-start/NAS and makes the graph expert error much larger for AAS/FS.

## Final Six-Sensor Metrics

| Sensor State | Models | MAPE | MAE | RMSE |
|---|---|---:|---:|---:|
| Cold-start | MoGERNN | 16.0564 | 6.4608 | 9.9416 |
|  | PatchSTG | 18.7481 | 7.1286 | 11.0736 |
|  | MCPST-FSL | 20.3519 | 7.9125 | 11.8637 |
|  | STGODE | 18.0159 | 6.6408 | 10.0260 |
|  | STID | 26.0818 | 9.9169 | 13.2962 |
|  | STGCN | 21.6616 | 7.7071 | 11.5875 |
|  | STBP | 32.8486 | 13.8470 | 15.4510 |
|  | SLPF | 28.4993 | 10.0148 | 14.0990 |
|  | STFiT | 25.4052 | 9.3925 | 12.3507 |
|  | FCLSTM | 28.0360 | 9.4019 | 13.1081 |
|  | GWNET | 19.6966 | 7.8963 | 11.5212 |
|  | HA | 28.0774 | 9.4503 | 13.1173 |
|  | ALFNet (ours) | 15.5032 | 5.9865 | 9.2806 |
| Warm-up | MoGERNN | 10.3680 | 3.8482 | 6.2759 |
|  | PatchSTG | 13.5171 | 5.5980 | 8.0032 |
|  | MCPST-FSL | 12.6832 | 4.3462 | 8.0488 |
|  | STGODE | 13.1691 | 4.4824 | 8.4414 |
|  | STID | 13.1975 | 4.5364 | 8.4576 |
|  | STGCN | 15.6051 | 4.9033 | 9.1522 |
|  | STBP | 17.0116 | 6.2717 | 9.2424 |
|  | SLPF | 26.0099 | 9.2037 | 12.9892 |
|  | STFiT | 13.6571 | 4.6100 | 8.7173 |
|  | FCLSTM | 13.4508 | 4.4662 | 8.6369 |
|  | GWNET | 13.0763 | 5.3812 | 8.7546 |
|  | HA | 15.4302 | 5.7126 | 10.5926 |
|  | ALFNet (ours) | 12.6511 | 4.2827 | 7.5711 |
| Adapted | MoGERNN | 10.3851 | 4.0796 | 6.9100 |
|  | PatchSTG | 12.2885 | 5.1180 | 7.7253 |
|  | MCPST-FSL | 10.2466 | 3.7599 | 6.7422 |
|  | STGODE | 10.9109 | 3.9293 | 6.9245 |
|  | STID | 11.3500 | 4.2796 | 7.2902 |
|  | STGCN | 13.1629 | 4.2416 | 7.6535 |
|  | STBP | 15.7255 | 6.0096 | 8.2698 |
|  | SLPF | 22.7530 | 8.1939 | 11.3537 |
|  | STFiT | 10.8224 | 3.9338 | 7.0148 |
|  | FCLSTM | 10.8733 | 3.8432 | 6.9236 |
|  | GWNET | 12.0218 | 5.0390 | 8.1584 |
|  | HA | 12.5015 | 4.8943 | 9.2632 |
|  | ALFNet (ours) | 9.9011 | 3.7028 | 6.3200 |
| Degrading | MoGERNN | 18.3006 | 4.9596 | 8.2645 |
|  | PatchSTG | 12.0316 | 4.3157 | 7.6741 |
|  | MCPST-FSL | 21.5692 | 6.1882 | 9.9529 |
|  | STGODE | 21.2970 | 6.7331 | 9.7622 |
|  | STID | 19.5866 | 4.9906 | 9.0671 |
|  | STGCN | 28.2235 | 8.2992 | 12.3147 |
|  | STBP | 15.7398 | 5.1740 | 9.5746 |
|  | SLPF | 12.0418 | 4.3091 | 7.0822 |
|  | STFiT | 33.0267 | 9.3378 | 13.1958 |
|  | FCLSTM | 35.2969 | 8.8610 | 13.5253 |
|  | GWNET | 20.2114 | 7.2103 | 10.2253 |
|  | HA | 35.2838 | 8.8851 | 13.5134 |
|  | ALFNet (ours) | 11.8737 | 4.0405 | 7.6093 |
| Outage | MoGERNN | 13.0339 | 4.8172 | 7.9816 |
|  | PatchSTG | 11.1961 | 4.0919 | 7.1189 |
|  | MCPST-FSL | 17.0617 | 6.1183 | 9.5830 |
|  | STGODE | 18.1911 | 6.6032 | 9.5948 |
|  | STID | 18.9664 | 6.2775 | 10.4050 |
|  | STGCN | 19.1665 | 7.4887 | 10.8666 |
|  | STBP | 13.6574 | 4.7196 | 8.5462 |
|  | SLPF | 11.9192 | 4.2485 | 7.0909 |
|  | STFiT | 26.4708 | 9.4499 | 12.4520 |
|  | FCLSTM | 30.8057 | 10.3165 | 14.0596 |
|  | GWNET | 15.3577 | 6.5882 | 9.5129 |
|  | HA | 30.8223 | 10.3529 | 14.0567 |
|  | ALFNet (ours) | 9.9623 | 3.4878 | 6.6911 |
| Steady | MoGERNN | 10.6537 | 3.7855 | 6.7324 |
|  | PatchSTG | 9.6582 | 3.6817 | 6.3541 |
|  | MCPST-FSL | 11.4746 | 3.9219 | 7.1014 |
|  | STGODE | 12.3561 | 4.0907 | 7.2921 |
|  | STID | 10.2277 | 3.6089 | 6.6132 |
|  | STGCN | 13.3227 | 4.3515 | 7.6562 |
|  | STBP | 10.6457 | 3.7593 | 6.6808 |
|  | SLPF | 11.4757 | 4.2732 | 7.0584 |
|  | STFiT | 12.7241 | 4.1780 | 7.5148 |
|  | FCLSTM | 12.4792 | 4.1117 | 7.5333 |
|  | GWNET | 11.9427 | 5.0827 | 8.2843 |
|  | HA | 14.7315 | 5.3717 | 10.1870 |
|  | ALFNet (ours) | 9.1483 | 3.2783 | 6.1551 |

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

## Fair Async Lifecycle Baseline Protocol

New baseline reruns should use `METRLA_async_lifecycle_100.yaml` under each model config directory. These files set `max_epoch=100`, `patience=10`, `train_for_async_lifecycle=true`, `lifecycle_training_strategy=round3_event_only`, and the same async test setting `events=4/k=12/seed=0`.

The baseline configs intentionally keep `lifecycle_role_features=false` and `lifecycle_history_age_feature=false` so the original comparison backbones are not structurally changed by extra role/age channels. ALFNet keeps lifecycle role/age input as part of the proposed method.

Configs prepared:
`MoGERNN`, `PatchSTG`, `MCPST_FSL`, `FCLSTM`, `GWNET`, `HA`, `SLPF`, `STBP`, `STFiT`, `STGCN`, `STGODE`, and `STID`.
