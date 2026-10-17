# Speed支持方案迁移结果（2026-10-06）

16项正式训练均结束；独立校验checkpoint/config/plan、有限权重、方法契约、最优validation epoch、零跳过batch及完整样本计数。复用2个原配置，共18项594格validation。未打开新test。

| 数据集/配置 | 第一格/33 | LifeAvg MAE | RMSE | MAPE% | 相对原版逐格胜/平/负 |
|---|---:|---:|---:|---:|---|
| METRLA_patch7 | 29 | 4.526930 | 6.884965 | 13.642090 | 0/33/0 |
| METRLA_center_zero_lr0.5_wd0.0001 | 15 | 4.912569 | 7.324645 | 14.727696 | 10/0/23 |
| METRLA_center_zero_lr1_wd0.0001 | 23 | 4.723625 | 7.171668 | 14.220917 | 11/0/22 |
| METRLA_center_zero_lr2_wd0.0001 | 26 | 4.551238 | 6.979807 | 13.710257 | 9/0/24 |
| METRLA_center_half_lr0.5_wd0.0001 | 24 | 4.819858 | 7.169087 | 13.893957 | 14/0/19 |
| METRLA_center_half_lr1_wd0.0001 | 22 | 4.740212 | 7.082068 | 14.017502 | 9/0/24 |
| METRLA_center_half_lr2_wd0.0001 | 26 | 4.698145 | 7.184336 | 14.004800 | 5/0/28 |
| METRLA_center_full_lr1_wd0.0001 | 16 | 4.887799 | 7.401075 | 14.114337 | 10/0/23 |
| 05_PEMSBAY_combo2 | 24 | 2.956078 | 4.604514 | 7.603029 | 0/33/0 |
| PEMSBAY_center_zero_lr0.5_wd1e-05 | 12 | 2.994603 | 4.733199 | 7.856334 | 11/0/22 |
| PEMSBAY_center_zero_lr1_wd1e-05 | 14 | 3.007463 | 4.736925 | 7.921374 | 8/0/25 |
| PEMSBAY_center_zero_lr2_wd1e-05 | 17 | 2.948064 | 4.619927 | 7.697104 | 10/0/23 |
| PEMSBAY_center_half_lr0.5_wd1e-05 | 11 | 3.027948 | 4.769713 | 8.058853 | 11/0/22 |
| PEMSBAY_center_half_lr1_wd1e-05 | 15 | 2.978481 | 4.707803 | 7.728822 | 13/0/20 |
| PEMSBAY_center_half_lr2_wd1e-05 | 24 | 2.946833 | 4.604688 | 7.673179 | 20/0/13 |
| PEMSBAY_center_full_lr1_wd1e-05 | 17 | 3.017699 | 4.797382 | 7.746464 | 11/0/22 |
| PEMSBAY_center_zero_lr2_wd0.0001 | 24 | 2.935454 | 4.592032 | 7.613820 | 25/0/8 |
| PEMSBAY_center_half_lr2_wd0.0001 | 23 | 2.956509 | 4.645939 | 7.665844 | 11/0/22 |

## 结论与取舍

METR-LA七个新候选三项LifeAvg均比原选配置差，最高26格，低于原29格。最接近原版的是alpha0/LR2；原方案保留。PEMS-BAY最高24格，与原24格持平，但没有同时改善原版三项LifeAvg的候选。没有新候选达到27格，不满足迁移成功条件。

PEMS-BAY alpha0/LR2/WD1e-4为最值得记录的取舍：MAE/RMSE略优、MAPE略差，且较alpha.5/LR2/WD1e-5三项LifeAvg均更低、胜格同为24。其具体幅度见下表，不把很小的MAPE退化当成新增硬阈值；不test的主因是尚无满足覆盖目标且明确胜过原版的方案。

| 代表新候选 | MAE变化% | RMSE变化% | MAPE变化% |
|---|---:|---:|---:|
| METRLA_center_zero_lr2_wd0.0001 | +0.5370 | +1.3775 | +0.4997 |
| PEMSBAY_center_zero_lr2_wd0.0001 | -0.6977 | -0.2711 | +0.1419 |

## 参数作用与解释边界

较小LR.5没有带来迁移收益；LR2往往改善新方案但仍未越过原配置。完整中心化在两个speed上均不占优。PEMS-BAY提高WD到1e-4对alpha0有益（17→24格），对alpha.5却略差（24→23格），说明正则化与中心化交互，不能用一个统一WD解释全部差异。dropout本轮固定.1，不声称已排除其他dropout。

已知实现改变同时包含可见历史插值、decoder残差scale从.3到1及中心化。因此本轮只能判断组合迁移效果，不足以把退化唯一归因到插值或倍率。speed与flow的量纲、节点变异和原版误差结构不同，属于合理解释假设而非本轮因果实验证明。不存在为统一叙事必须更换已有好结果的依据。

建议按用户允许的方式保留speed原ALFNet，flow采用支持增强方案（04alpha0/LR1，08alpha.5/LR1）。披露两类使用的处理开关、中心强度与decoder倍率；不声称同一固定增强设置已在四数据集验证成功。旧test中METRLA33/33、PEMSBAY29/33与这里validation29/24是不同split，不能混用。

本轮不自动进行未达门槛候选test，不追加训练，不覆盖正式主表/阶段表。历史test已影响研究过程，保留选型记录。若用户后续明确授权近门槛探索性test，可另冻候选范围，不能预先认定test会更好。

文件：validation_results.xlsx/csv全部18项主表/阶段表；validation_comparison.xlsx/csv原版、代表新候选与八基线，第一加粗第二下划线；validation_audit.json与selection.json为审计和逐格判定。
