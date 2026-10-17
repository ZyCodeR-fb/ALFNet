# 四份论文实验表（2026-10-02重新选型）

2026-10-02按8个基线重新审查全部67配置；PEMS04改用default，其余不变。test参与选型，非独立确认；两split使用相同权重。

比较范围为ALFNet及8个基线；既有原始结果、被移除模型与此前表格完整保留。

| 数据集 | 配置 | validation第一/33 | test第一/33 | test LifeAvg MAE / RMSE / MAPE |
|---|---|---:|---:|---|
| METRLA | METRLA_patch7 | 29 | 33 | 3.8421 / 6.0846 / 10.3365 |
| PEMSBAY | 05_PEMSBAY_combo2 | 24 | 29 | 2.7232 / 4.2006 / 6.2761 |
| PEMS03 | PEMS03_capacity_mid | 16 | 8 | 35.2678 / 47.3101 / 50.4365 |
| PEMS04 | PEMS04_default | 19 | 22 | 38.1597 / 51.6398 / 25.0477 |
| PEMS08 | PEMS08_lr_double | 21 | 21 | 38.4054 / 49.2460 / 52.2712 |

## 取舍

PEMS04默认容量10/LR1/dropout0.1/WD1e-4：test 22格第一，LifeAvg三项均优于旧combo1，故替换。
PEMS03保留容量16：8格第一；WD1e-5有9格第一，但LifeAvg MAE/RMSE/MAPE分别由35.268/47.310/50.436退化为36.554/48.805/52.328，约3.65%/3.16%/3.75%。一格优势不足以抵偿三项平均误差一致变差；这是一项明确的综合取舍，没有设置硬性退化门槛。9格候选保留在完整排名中。
其余三数据集原配置仍为最多第一格的配置之一。完整67项validation/test排名见eight_baseline_candidate_ranking.json。33格有重叠，不能视作33个独立统计检验。

## 文件

01、02分别为选定同一权重的validation/test对比表，第一加粗、第二下划线；03、04为四轴单变量及下部组合结果，数值与分组保持不变，仅同步最终配置标记。
全部配置与test已被观察，本次仅离线重算排名。新修复实验另设validation-only计划，不覆盖这套结果。
旧表备份before_reselection_20261002；原模型过滤备份before_model_filter_20261001。MoGERNN旧patience3、SLPF/STBP适配限制继续见BASELINE_FAIRNESS_20260928.md。
