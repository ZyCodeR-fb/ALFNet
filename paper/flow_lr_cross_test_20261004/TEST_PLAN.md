# 单候选测试计划

当前用户授权：训练结束后分析结果，满足要求则test。唯一新增test固定为`05_PEMS08_triple_halfcenter_lr1`，不得根据test中途改配置或checkpoint。

该候选validation30/33第一（90.91%），LifeAvg MAE/RMSE/MAPE=35.671322/46.488155/31.712988，全部低于八基线相应最优38.981694/50.334806/36.571214，满足至少27格及三项LifeAvg领先。04新候选至多26格，不在本次test范围。04已有合格候选test直接复用。

独立审计4项新训练的计划/配置/权重hash、有限权重、最优epoch、零跳过batch、完整计数；导出新旧8项264格validation。使用原`flow_unified.adapter`恢复相同方法语义，先逐项校验checkpoint方法契约，再完整重放33格validation；通过后才测试一次。单GPU单评估，不训练其他模型。保留全部旧test及正式四表，生成独立版本结果。

训练计划c99b578d4f8f7ced5d82adfae6f9a103d6afb0c9d994b576c70e53d887264bec。新评估计划由freeze写入configs/paper/flow_lr_cross_test_20261004/plan.json。历史test曾参与研究方向选择，本次为探索性确认，不声称未经观察的独立holdout。PEMS08曾出现validation改善但test退化，不能据本轮validation提前宣布全部目标完成。
