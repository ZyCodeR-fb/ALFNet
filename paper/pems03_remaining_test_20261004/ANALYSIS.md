# PEMS03其余候选test与差异诊断

5个新增test全部完成且完整validation复核最大差0；复用已测半中心化LR1、原版及重建，共8项264格test。没有重训、重新选epoch或更改指标/划分。

| 配置 | validation第一/33 | test第一/33 | test LifeAvg MAE | RMSE | MAPE% |
|---|---:|---:|---:|---:|---:|
| PEMS03_capacity_mid | 16 | 8 | 35.26780 | 47.31011 | 50.43647 |
| PEMS03_center_zero_lr1 | 20 | 13 | 34.53128 | 46.45620 | 44.05491 |
| PEMS03_center_zero_lr2 | 23 | 14 | 34.17106 | 45.74895 | 42.96429 |
| PEMS03_center_half_lr1 | 26 | 20 | 34.80123 | 46.41778 | 39.83894 |
| PEMS03_center_half_lr2 | 19 | 16 | 34.82074 | 46.01926 | 43.62768 |
| PEMS03_center_full_lr1 | 5 | 4 | 36.94726 | 49.83707 | 48.20110 |
| PEMS03_center_full_lr2 | 5 | 7 | 35.53584 | 48.24727 | 50.36971 |

本轮test最多第一格配置：PEMS03_center_half_lr1，20/33。满足至少27格且三项LifeAvg均领先八基线的配置：无。test已用于广泛候选比较，只能视为探索性证据，不能称独立选型确认。

## 误差差距与排名差距是两件事

原validation最佳半中心化LR1的LifeAvg MAE增量约99.06%、RMSE增量96.50%、MAPE增量82.29%来自Unseen项（LifeAvg为四状态均值，可精确分解；不是因果贡献）。但26→20丢失的六个第一格全部是Join MAE/RMSE、Cold MAE、Warm MAE/RMSE、Ready RMSE。Cold test MAE仅比STID高约0.0086，validation Warm MAE仅领先约0.0314；计数把小幅排序变化也计作整格丢失，且主表与阶段表并非独立观测。不能把失去六格等同整体误差暴增。

## 分布与空间覆盖的证据

K/V/T分别268/45/45且互斥。validation与test既换时间段，也把未见节点由V换成T，目标群体不同。T没有直接K邻居者5/45（11.11%），V为1/45（2.22%）；该统计用公开邻接对称化的直接连接，不代表完全无法通过多跳获得信息。

已知K节点平均流量从validation181.60降至test174.35（约-3.99%）；同一validation时段V均值180.65、T176.67，说明节点组成也不同；test T均值170.17。节点均值CV：validation V .517，test T .538；正流量≤30占比11.73%→12.89%。这支持时间变化及节点异质性共同存在，但仅凭描述统计不能断言各自导致多少预测误差。

八基线Unseen三指标全部从validation到test上升，进一步说明不是ALFNet独有的评估异常：

| 基线 | Unseen MAE变化% | RMSE变化% | MAPE变化% |
|---|---:|---:|---:|
| MoGERNN | 22.33 | 24.80 | 54.45 |
| UIGNN | 22.40 | 28.99 | 62.14 |
| GinAR | 15.15 | 15.14 | 51.31 |
| SLPF | 1.44 | 5.34 | 26.14 |
| STGCN | 12.78 | 9.83 | 14.77 |
| STID | 5.10 | 7.76 | 34.68 |
| PatchSTG | 13.05 | 14.64 | 17.15 |
| STBP | 2.90 | 7.05 | 24.19 |

## 实现检查与处理决定

五候选逐一恢复训练时适配器并检查方法契约；checkpoint/scaler/seed/static topology/exposure hash一致，validation全33格精确复现；test关闭teacher forcing，协议计数与八基线一致。未发现可通过简单代码补丁纠正的错误。原始非正值仍按既有缺失定义处理，未修改MAPE分母或mask。

若无达标配置，继续改善需要解决无自身历史节点的空间尺度推断与长期中断泛化，属于新的模型/训练研究，不是简单调一个参数；本轮不启动。按用户约束，保持METR-LA、PEMS-BAY、PEMS04、PEMS08正文范围，PEMS03完整保留作附录/限制证据。不删除原始数据、权重或不利结果，不把test后范围取舍包装为预设benchmark。

输出：test_results.xlsx/csv为全部8项主表/阶段表；test_best_comparison.xlsx/csv为胜格最多候选与八基线；distribution_diagnostics.json为九个时间×节点组描述统计、误差分解和排名变化；decision.json为全部门槛判定。
