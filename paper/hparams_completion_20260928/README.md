# 本轮运行与阅读入口

2026-09-30：24项新增训练已全部完成，断电续训正常结束。55项候选已核验并完成validation选型。
最新主表、阶段表、ALFNet五数据集超参数表及全部候选工作簿见
[最终结果目录](../validation_seed32_hparams_final_20260930/RESULTS_ANALYSIS.md)。
其中 `final_tables.xlsx` 包含全部表；`ALFNET_HYPERPARAMETERS.md` 为独立超参数表。
旧结果与test表保留。本目录原 `results.json` 仍是完整候选来源，不代表自动选择了最小MAE项。

冻结计划 `configs/paper/hparams_completion_20260928/plan.json`，身份
`5b7140ff0f835e34fa46feecb9c17da8b98dacd1c0df4d4e87cb9a9f5ed498c0`。

24项新训练、31条历史ALFNet配置复用；详见PLAN.md。运行脚本
`scripts/run_hparam_completion.py`，tmux `hparams-completion-20260928`。
实时状态在对应 `artifacts/experiments/hparams_completion_20260928/<sha>/`
的launcher.log、preflight/、queue_status.json；预检完成前没有正式worker。
正式阶段最多5worker，并根据真实峰值显存限制派发。

本轮未打开test，未改架构或损失；历史ALFNet源码完全一致，runner唯一新增Merlin注册。5项单变量配置测试通过，独立代码审查检查24配置的字段差异、31复用覆盖与队列dispatch。

完成后自动写results.json、cells.csv、results_summary.md，包含旧/新全部55个配置。
`first_cells`定义为**严格低于全部10个基线**，不包括精确并列第一；LifeAvg Pareto集合按MAE/RMSE/MAPE三个原始指标计算。不同数据集绝对误差不合并。
最低LifeAvg MAE与最多第一格可能不是同一配置，结果报告会同时保留。自动汇总不会覆盖已有主表/权重或将validation写入test表。

公平性审计见 `paper/BASELINE_FAIRNESS_20260928.md`，Fault机制解释见
`paper/MERLIN_FAULT_DIAGNOSIS_20260928.md`。这些审计指出现有baseline适配/预算差异，不代表已重训修复这些旧结果。
