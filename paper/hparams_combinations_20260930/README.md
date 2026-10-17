# 最后一轮组合实验入口

计划：`configs/paper/hparams_combinations_20260930/plan.json`。
SHA：`5412b3580499c4f4602df6cef17d663442b0f3c7d8cf956befe5e4d9fa71b4bd`。
具体12组合与理由见 [PLAN.md](PLAN.md)，去重来源为55项已核验配置。

启动：ReaCH Python `scripts/run_hparam_combinations.py queue`，tmux `hparams-combinations-20260930`。
过程：先构建5数据集对照身份，再逐项20训练＋4验证batch预检；全部12项通过后最多5worker并行。
预检中的CPU数据/模型构建阶段不代表GPU训练中断。输出根目录为
`artifacts/experiments/hparams_combinations_20260930/<上述SHA>/`。

- `launcher.log`、`preflight/`、`PREFLIGHT_COMPLETE.json`：预检证据。
- `queue_status.json`、每项log、`runs/`：正式训练、best/last权重与报告。
- `COMPLETE.json`：67项结果汇总；`TABLES_COMPLETE.json`：版本化表格生成完毕。
- 失败不会取消其他独立项；不自动重复失败项。进程/PID必须实时核验。

本目录完成后包含全部67项`results.json/cells.csv/results_summary.md`，以及本轮`final_tables.xlsx/final_tables.md`、选择审阅`selection_review.json`和实际配置`alfnet_hyperparameters.json`。旧单变量数据表保留参考点，在新工作簿末尾添加五张“最后组合”表。自动更新仅处理无取舍改善，其他候选不淘汰；最终综合分析可选择有取舍的候选，无旧硬退化门槛。

已完成启动前验证：3项针对性测试、12配置完整字段差异与55历史配置去重、冻结hash检查；临时目录中模拟67项汇总及5次预期替换，1815个Excel主表/阶段指标与排名回读通过。该模拟只测试收集逻辑，不是实际训练结果。

无模型/损失修改，无test、commit或push。新计划冻结了旧代码/配置/表格，运行期间不要改写这些源文件，否则后续worker会拒绝启动。
