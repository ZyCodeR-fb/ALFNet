# 超参数与消融结果汇总（validation，seed32）

超参数14/14新训练成功，4项复用；消融24/24成功，4项Full复用。
独立补跑已完成，消融表完整。
未进行新test、未更换最终Full；新超参数结果不能替换既有配对消融的Full对照。

## 超参数：同一固定锚点下的结果

| 数据集/配置 | MAE↓ | RMSE↓ | MAPE↓ | 第一格/33 |
|---|---:|---:|---:|---:|
| 06_PEMS04_bridge_delta1 | 40.0215 | 54.3389 | 31.9149 | 27 |
| 06_PEMS04_bridge_delta1_lr2 | 39.5717 | 53.5329 | 33.0216 | 23 |
| 05_PEMS08_triple_halfcenter_lr1 | 35.6713 | 46.4882 | 31.7130 | 30 |
| 05_PEMS08_triple_halfcenter | 36.1340 | 46.8327 | 32.7171 | 29 |
| 00_PEMS04_capacity_14 | 39.7065 | 53.6357 | 34.2667 | 27 |
| 02_PEMS04_capacity_18 | 39.4966 | 53.3529 | 34.9389 | 27 |
| 04_PEMS04_lr_factor_0.5 | 40.0822 | 54.9550 | 33.1350 | 15 |
| 06_PEMS04_dropout_0.05 | 39.5813 | 53.1957 | 32.9189 | 22 |
| 08_PEMS04_dropout_0.2 | 40.2230 | 53.7505 | 33.2017 | 26 |
| 10_PEMS04_weight_decay_1e-05 | 39.8322 | 53.5790 | 31.5931 | 27 |
| 12_PEMS04_weight_decay_0.001 | 39.6464 | 53.8858 | 31.8125 | 27 |
| 01_PEMS08_capacity_8 | 35.0924 | 46.5515 | 32.2617 | 29 |
| 03_PEMS08_capacity_10 | 35.6720 | 47.4274 | 34.5078 | 29 |
| 05_PEMS08_lr_factor_0.5 | 34.5275 | 45.4515 | 27.7428 | 28 |
| 07_PEMS08_dropout_0.05 | 35.5853 | 46.3675 | 31.6488 | 27 |
| 09_PEMS08_dropout_0.2 | 34.0858 | 45.4560 | 31.0849 | 30 |
| 11_PEMS08_weight_decay_1e-05 | 36.3678 | 47.1114 | 32.0191 | 29 |
| 13_PEMS08_weight_decay_0.001 | 32.6025 | 42.9736 | 29.4949 | 32 |

PEMS04：WD1e-5和1e-3均改善三项LifeAvg且维持27格；1e-5三指标均优于1e-3。容量18偏向MAE/RMSE，但MAPE退化。
PEMS08：WD1e-3三项LifeAvg改善且30→32格；LR0.5的MAPE更低但覆盖降至28格；dropout0.2改善三项并保持30格。单因素收益不能假设可叠加；本轮不启动组合或自动选型。

## 消融：相对固定Full的LifeAvg变化

正值表示移除组件后误差上升（支持Full）；负值表示该消融更好。

| 数据集 | 消融 | MAE变化% | RMSE变化% | MAPE变化% | Full更好格/33 |
|---|---|---:|---:|---:|---:|
| METRLA | w/o State Age | +1.73 | +0.78 | +1.30 | 25 |
| METRLA | w/o Adaptive Routing | +4.02 | +2.68 | +2.52 | 31 |
| METRLA | w/o Spatial Context | +7.15 | +4.05 | +11.65 | 33 |
| METRLA | w/o ALFRecon | +5.30 | +3.91 | +3.78 | 29 |
| METRLA | w/o ALFTemporal–FiLM | +15.48 | +15.28 | +11.83 | 25 |
| METRLA | w/o Graph Refinement | +1.66 | +1.33 | +0.56 | 29 |
| PEMSBAY | w/o State Age | +0.27 | +1.62 | +2.96 | 31 |
| PEMSBAY | w/o Adaptive Routing | +1.51 | +1.87 | +4.68 | 32 |
| PEMSBAY | w/o Spatial Context | +5.74 | +5.29 | +5.89 | 29 |
| PEMSBAY | w/o ALFRecon | +0.29 | +0.78 | +1.98 | 27 |
| PEMSBAY | w/o ALFTemporal–FiLM | +2.05 | +4.30 | +4.70 | 19 |
| PEMSBAY | w/o Graph Refinement | +0.44 | +1.78 | +3.25 | 31 |
| PEMS04 | w/o State Age | -2.76 | -3.14 | +2.34 | 15 |
| PEMS04 | w/o Adaptive Routing | -0.13 | -0.69 | +1.41 | 26 |
| PEMS04 | w/o Spatial Context | +3.40 | +1.23 | +12.31 | 29 |
| PEMS04 | w/o ALFRecon | +1.77 | -0.12 | +17.46 | 26 |
| PEMS04 | w/o ALFTemporal–FiLM | +5.39 | +5.18 | +21.58 | 32 |
| PEMS04 | w/o Graph Refinement | -0.93 | -1.02 | -1.19 | 8 |
| PEMS08 | w/o State Age | -1.95 | -0.74 | -0.77 | 13 |
| PEMS08 | w/o Adaptive Routing | +1.96 | +1.43 | +8.12 | 30 |
| PEMS08 | w/o Spatial Context | -0.31 | -1.33 | +0.48 | 26 |
| PEMS08 | w/o ALFRecon | +9.27 | +6.21 | +9.11 | 33 |
| PEMS08 | w/o ALFTemporal–FiLM | +11.62 | +10.61 | -5.04 | 31 |
| PEMS08 | w/o Graph Refinement | -4.26 | -3.59 | -0.64 | 22 |

METR-LA六项消融的三项LifeAvg均退化；PEMS-BAY首批完成五项也均退化。空间上下文补跑的最新结果及完成状态见上表，不预先假设其方向。
flow结果呈现条件依赖：PEMS04/08移除Graph Refinement均改善三项LifeAvg；PEMS08移除State Age也改善三项。不能声称所有组件在所有数据集必不可少。
时间条件分支对MAE/RMSE的贡献较稳定，但PEMS08移除后MAPE降低；学习式重建对PEMS08三项均重要，对PEMS04主要改善MAE/MAPE。
论文应区分组件作用与配置交互，所有消融保留，单seed结果不作显著性声明。

## 文件使用

- 03_hyperparameters_validation.xlsx：四数据集单因素主表/阶段表，历史配置与组合附后；flow锚点固定最终方案，speed保留原实验锚点。
- 05_ablations_validation.xlsx：四数据集全部可用消融主表/阶段表、LifeAvg变化、缺失状态。
- 原01/02正式比较表及04超参数test表未改写：新消融与flow单因素尚无test，不能复制validation充当test。
- verification.json与ablation_results_and_provenance.json保存审计和来源。独立重跑结束后再次运行本脚本可补齐唯一空缺。
