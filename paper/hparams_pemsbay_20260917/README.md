# PEMS-BAY超参数实验索引

**2026-09-20已完成**：五项训练结束，只有patch16通过原门槛。候选validation独立复核最大差异0，另一个会话的test已完成；本轮修复离线表格汇总，未重复test。当前采用PEMS-BAY Focal容量16更新版，原容量11留作对照。详见[结果分析](RESULTS_ANALYSIS_20260920.md)及[审计](patch16_update_audit.json)。下述运行中描述为历史记录。

当前状态：五项正式训练均已启动。预检全部通过、参数量均1,037,529、初始参数SHA相同；前次现场GPU利用率约96%（瞬时观测，不是全程平均）。尚无正式完整epoch/最终精度结论，以queue_status和后续报告为准。tmux：`pemsbay-hparams-20260917`。

- [PLAN.md](PLAN.md)：本轮完整设计、公开论文依据、五项范围与validation门槛；冻结后不得编辑。
- [layout_diagnosis.json](layout_diagnosis.json)：train上下文静态布局核验。两数据集均32个分区；METRLA容量8、BAY容量11。BAY容量11/12/14/16分别带来27/59/123/187个额外token槽，全部是正邻接权重连接的外部分区节点。不能把槽数当作新增独立传感器数量。
- 配置/计划：`configs/paper/pemsbay_hparams_20260917/`。
- 当前计划SHA：`1ec318e96fdff89dc664ac66868a0d5a34d2d0e0c5ef75d158381a3f4d8b1bc4`。
- 运行输出：`artifacts/experiments/pemsbay_hparams_20260917/<plan_sha>/`，查看`PREFLIGHT_COMPLETE.json`、`queue_status.json`、各run日志与最终`COMPLETE.json`。
- 全五项完成后自动生成`results.json`和`sensitivity_cells.csv`，只含validation，不自动替换最终权重/test表。

训练已知节点的scaler也不同：METRLA均值58.133、标准差13.003；BAY均值62.926、标准差9.049。来自已有checkpoint身份，没有读取新test动态。相同损失/学习率不保证相同数据分布或梯度行为；Focal权重依赖原尺度误差。这是扫描优化步长的动机，不是已证明的排名差异原因。两个最终模型的训练配置与专家配置已逐字段比较，除容量及数据/身份标签外没有其他有效超参数差异。

原始release计划63b50760仍通过哈希检查；本轮没有修改模型/损失/训练器源文件。新实验由根目录新增脚本调用既有训练函数，独立计划逐字段约束候选，保留用户当前已提交的783cbf7版本。
