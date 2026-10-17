# 四份论文实验数据表

2026-10-01展示范围更新：用户要求从01、02对比表移除ST-FiT和Merlin，现为ALFNet＋8个基线。两份Excel及配套CSV/Markdown已按剩余模型重新标注第一、第二名；数值和选定权重不变，03/04未修改。下文原11模型排名属于筛选前历史分析，不代表当前9模型表的排名；当前核验见`model_filter_verification.json`。筛选前01/02保存在`before_model_filter_20261001/`，完整原始实验结果保留。复现时先执行`package_four_paper_tables.py`，再执行`filter_final_comparison_tables.py`。

117项test已全部完成并核验。按上一轮预先写定的test探索选型规则确定五个最终配置；validation和test使用相同配置/权重，不能分别挑选。因为test参与最终选择，本套表属于test探索选型后的结果，不可称为独立test确认。先前validation选定配置的test仍保留在../all_final_test_20261001/validation_selected_test.xlsx。

| 数据集 | 容量 | LR倍率 | Dropout | WD | validation第一格/33 | test主表第一/15 | test阶段第一/18 | test第一格/33 | test LifeAvg MAE / RMSE / MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| METRLA | 7 | 1 | 0.1 | 0.0001 | 21 | 13 | 17 | 30 | 3.8421 / 6.0846 / 10.3365 |
| PEMSBAY | 16 | 2 | 0.1 | 1e-05 | 18 | 12 | 10 | 22 | 2.7232 / 4.2006 / 6.2761 |
| PEMS03 | 16 | 1 | 0.1 | 0.0001 | 5 | 3 | 2 | 5 | 35.2678 / 47.3101 / 50.4365 |
| PEMS04 | 10 | 0.5 | 0.05 | 1e-05 | 8 | 5 | 4 | 9 | 39.9994 / 54.0097 / 25.2619 |
| PEMS08 | 6 | 2 | 0.1 | 0.0001 | 11 | 6 | 4 | 10 | 38.4054 / 49.2460 / 52.2712 |

## 使用方式

- 01_comparison_validation.xlsx：最终配置的validation对比主表/阶段表。
- 02_comparison_test.xlsx：同一最终配置的test对比主表/阶段表。
- 03_hyperparameters_validation.xlsx：论文四轴单变量validation主表/阶段表；其他组合在各sheet下方灰色区，不放正文。
- 04_hyperparameters_test.xlsx：同样分组/排序的test结果；其他组合同样置底。
每个工作簿包含五数据集的主表、阶段表；对比表第一加粗、第二下划线。CSV提供原精度数值，组合结果仅作记录。47个唯一参考/单变量配置＋20个其他组合覆盖全部67项，参考点为作图在四组重复列出，不是重复训练。

超参数固定参考点仍为METRLA容量8/LR1，BAY容量16/LR1，03容量20/LR1，04容量10/LR.5，08容量6/LR.5，drop.1/WD1e-4。这些不是新的最终配置；不能将已跑曲线重新标成围绕最终配置的单变量实验。完整配置与checkpoint来源见manifest和各workbook配置索引。

## 结果解读与取舍

METR-LA：容量7/LR1在test达到30/33第一，明显优于先前validation选择的25格；这也是最优配置跨split改变的实例。PEMS-BAY：LR2/WD1e-5在test保留22格第一，与测试前选择一致。
PEMS03：没有候选达到三项test LifeAvg均优于最强基线，按预先规则使用LifeAvg Pareto集合后选容量16/LR1，5格第一。该配置test LifeAvg MAE/RMSE第一，MAPE第三；不能继续声称五数据集15项平均指标全优。
PEMS04：低dropout/弱WD取得9格第一，比原配置8格多一格，但LifeAvg三项较原配置略差；这来自预先写定的覆盖优先规则，不是所有指标占优。原配置完整保留。
PEMS08：LR2在test为10格第一，并显著优于先前13格validation优胜组合的test结果（后者test只有4格第一）。这提示验证/测试时段差异和单种子选型波动，不能将其归因于某单一模块。
最终test合计76/165第一，METRLA超过80%，其余四个数据集未达80%。最终配置validation合计63/165第一；test LifeAvg14/15第一。单seed、重叠33格不能构成独立统计显著性证据。

基线预算/适配差异沿用既有披露；未修改模型、配置文件或权重，未重跑test，未修改论文正文。
