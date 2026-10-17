# PEMS03近门槛候选test结果

用户明确授权在validation26/33未达门槛情况下测试最接近候选。固定半中心化/LR1、epoch57权重，完整33格validation复核最大差0，随后82个test batch完成，exit0。训练/评估数据划分、mask、指标和八基线不变。单新test，原版与旧reconstruction测试复用，共3配置99格。

| 配置 | test第一/33 | LifeAvg MAE | RMSE | MAPE% |
|---|---:|---:|---:|---:|
| PEMS03_center_half_lr1 | 20 | 34.80123 | 46.41778 | 39.83894 |
| 00_PEMS03_reconstruction | 13 | 36.62156 | 48.87057 | 47.66099 |
| PEMS03_capacity_mid | 8 | 35.26780 | 47.31011 | 50.43647 |

新候选20/33=60.61%，低于validation26/33，不满足27格目标。三项LifeAvg仍各自优于八基线最佳37.59699/50.62328/44.16910；相对原ALFNet改善1.32%/1.89%/21.01%，覆盖从8→20格。平均表现有实质改善，但没有满足用户同时要求的覆盖目标。

未胜13格：Unseen MAE/RMSE；Join MAE/RMSE；Cold MAE/MAPE；Warm MAE/RMSE；Ready MAE/RMSE；Outage三项。Unseen MAPE已领先，失败并非全部集中MAPE。

| 条件 | 指标 | 候选 | 最优基线 | 基线值 |
|---|---|---:|---|---:|
| Unseen | MAE | 78.63163 | UIGNN | 71.12075 |
| Unseen | RMSE | 101.97234 | UIGNN | 96.42113 |
| Join | MAE | 19.80354 | STID | 19.09329 |
| Join | RMSE | 27.69735 | STID | 27.24135 |
| Cold | MAE | 23.43038 | STID | 23.42178 |
| Cold | MAPE | 31.72718 | STID | 26.05930 |
| Warm | MAE | 19.47072 | STID | 18.77238 |
| Warm | RMSE | 27.27816 | STID | 26.82176 |
| Ready | MAE | 19.50488 | STID | 17.97394 |
| Ready | RMSE | 27.53401 | STID | 25.28092 |
| Outage | MAE | 46.13245 | PatchSTG | 35.39357 |
| Outage | RMSE | 61.74676 | PatchSTG | 48.13475 |
| Outage | MAPE | 67.68331 | PatchSTG | 48.57125 |

建议停止本轮PEMS03搜索，保持现有四数据集正文范围，将PEMS03完整结果保留为附录/泛化限制证据，不删除或隐瞒不利结果。不把按照test表现决定正文范围描述为事先固定数据集。没有自动启动其他候选test或训练。

文件：test_comparison.xlsx/csv为候选与八基线主表/阶段表（第一加粗、第二下划线）；test_results.xlsx/csv保留新旧三候选；validation_results.xlsx保留七项validation；selection.json记录目标判定。历史正式四表与论文正文未覆盖。
