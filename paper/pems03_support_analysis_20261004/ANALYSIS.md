# PEMS03 支持方案参数实验分析

六项新训练均完整结束并通过独立审计；复用原版，共七项231格validation。无新test。

| 配置 | 第一格/33 | LifeAvg MAE | RMSE | MAPE% |
|---|---:|---:|---:|---:|
| PEMS03_capacity_mid | 16 | 32.61197 | 42.76921 | 32.93310 |
| PEMS03_center_zero_lr1 | 20 | 33.25830 | 43.07503 | 31.93708 |
| PEMS03_center_zero_lr2 | 23 | 33.33028 | 43.45693 | 33.73319 |
| PEMS03_center_half_lr1 | 26 | 31.36754 | 41.24802 | 30.16344 |
| PEMS03_center_half_lr2 | 19 | 32.44154 | 42.50815 | 31.51093 |
| PEMS03_center_full_lr1 | 5 | 34.65294 | 45.53494 | 34.63051 |
| PEMS03_center_full_lr2 | 5 | 32.72093 | 43.23799 | 36.86427 |

最佳为中心化0.5/LR1，26/33=78.79%，距离至少27格差一格。它也是本轮三项LifeAvg最低配置，均领先八基线；相对原版三指标改善3.82%/3.56%/8.41%。未满足联合目标，不能把78.79%向上写成80%。

同一LR1，中心强度0/.5/1对应20/26/5格，表明部分中心化更合适；增至LR2后半中心化反而降至19格。增加中心化强度或学习率并非单调改善。单seed结果不能作为显著性或机制因果证明。

Unseen三项相对原版均改善，MAPE已胜过八基线，MAE/RMSE仍落后UIGNN；长期中断Outage三项落后PatchSTG，是另一独立瓶颈。总体均值收益不代表每阶段都占优。

| 未胜条件 | 指标 | 候选 | 最优基线 | 基线值 |
|---|---|---:|---|---:|
| Unseen | MAE | 65.02653 | UIGNN | 58.10435 |
| Unseen | RMSE | 82.01728 | UIGNN | 74.74872 |
| Cold | MAPE | 21.89120 | STID | 21.47591 |
| Ready | MAE | 18.76585 | STID | 18.59051 |
| Outage | MAE | 42.31641 | PatchSTG | 31.67858 |
| Outage | RMSE | 57.28664 | PatchSTG | 44.74244 |
| Outage | MAPE | 50.89439 | PatchSTG | 36.05935 |

结论：方案在PEMS03有效，但尚未达到事先目标。当前不自动test、不追加训练、不恢复PEMS03正文数据集、不覆盖历史正式表。保留所有配置、权重与结果。若后续明确授权将接近门槛的候选做探索性test，应另冻计划并披露未达validation门槛，不能事后改写本轮要求。

文件：validation_results.xlsx/csv为全部7项主表/阶段表；validation_comparison.xlsx/csv为最佳候选与8基线，第一加粗第二下划线；validation_audit.json为训练完整性审计；selection.json记录差距与未打开test的决定。
