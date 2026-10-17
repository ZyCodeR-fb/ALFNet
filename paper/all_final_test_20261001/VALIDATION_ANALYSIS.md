# 最后一轮组合结果与test前选型

最后12项组合训练全部exit0，冻结训练计划校验通过。检查checkpoint参数有限性、配置原始/归档身份、报告最佳epoch与完整validation LifeAvg MAE最小值、无跳过batch。单变量55项＋组合12项合计67项。以下选择在打开本轮新test前写定；历史speed试验曾使用test反馈，不能把整个项目描述为从未观察test。

| 数据集 | 选择 | 容量 | LR倍率 | Dropout | WD | LifeAvg MAE | RMSE | MAPE (%) | 第一格/33 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| METR-LA | METRLA_lr_double | 8 | 2 | .1 | 1e-4 | 4.46337 | 6.77610 | 13.00883 | 25 |
| PEMS-BAY | 05_PEMSBAY_combo2 | 16 | 2 | .1 | 1e-5 | 2.95608 | 4.60451 | 7.60303 | 18 |
| PEMS03 | 16_PEMS03_lr_half | 20 | .5 | .1 | 1e-4 | 31.77512 | 41.05352 | 32.19644 | 6 |
| PEMS04 | PEMS04_lr_half | 10 | .5 | .1 | 1e-4 | 37.87326 | 51.41447 | 34.83448 | 11 |
| PEMS08 | 03_PEMS08_combo1 | 6 | .5 | .05 | 1e-5 | 34.10909 | 44.83209 | 26.74649 | 13 |

总第一格69→73/165，五个数据集LifeAvg三指标继续优于十个基线；各数据集仍未达到80%第一格。本轮没有证明所有参数的独立收益可以叠加。

- METR-LA：容量12/LR2的RMSE从6.77610略降到6.76836附近，但MAE/MAPE及覆盖不及容量8/LR2，后者仍25格第一。更强WD组合也未提供整体收益，保留原选定。
- PEMS-BAY：弱WD组合保持18格第一，相对上一版选定MAE+0.137%、RMSE−0.307%、MAPE−0.286%。以非常小的MAE代价换两项改善。容量16/LR1仍是候选中最低三项LifeAvg的备选，不能把覆盖优先选择称为平均误差全局最小。
- PEMS03：两个新组合不如原选定的三指标组合，保留容量20/LR.5。容量20/LR1仍有更低MAE，但较高RMSE/MAPE。
- PEMS04：两个新组合的第一格数和LifeAvg均未优于当前选定，维持容量10/LR.5。
- PEMS08：drop.05/WD1e-5提升覆盖9→13；相对原选定LifeAvg MAE+1.044%、RMSE+0.618%、MAPE−0.751%。这是覆盖优先的取舍。drop.05/WD1e-3的LifeAvg更好（33.1991/43.9916/26.5872），但只有8格第一，明确作为平均误差备选保留。两种结果不能混成一行。

本目录`validation_final_20261001.xlsx/.md`已更新完整主表/阶段表及配置；50个基线记录不变。所有67项原始validation数据继续保存在`../hparams_combinations_20260930/results.json`。

## 全量test与输出约定

2026-10-01用户明确授权全部对比算法与超参数test，并可依据test确定探索性最优组合。已冻结117checkpoint清单，23项已有test通过权重hash及源模型/数据代码一致性核验后复用，94项补测。对于普通确定性模型，完整validation一致才进入test；ST-FiT保留源Gumbel随机评估，固定seed重放核验单独披露，不伪称历史随机状态已恢复。

结束后分别输出：

- `validation_selected_test.xlsx/.md`：上表在本轮新test前确定的五个配置与全部基线。
- `exploratory_test_selected.xlsx/.md`：按照已写定覆盖优先规则选择的test探索表；不是独立泛化确认。
- `all_hyperparameter_tests.xlsx`：67项ALFNet的test主表/阶段表，按固定参考点四轴单变量和其他组合分组。
- `all_results_and_provenance.json`：全部117项test、配置、权重、validation选型及test探索选型。

不修改旧test表或旧checkpoint，不重训，不根据test只保留有利候选。不以单seed和重叠33格作显著性推断。现有MoGERNN早停预算和SLPF/STBP适配差异不因补test而自动消失。
