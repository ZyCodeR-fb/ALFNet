# 最新结果与论文超参数数据入口

`final_tables.xlsx` 包含：

- 五数据集×主表/阶段表：ALFNet与十个比较算法，数值及排名样式保持上一版不变。
- `ALFNet超参数`：最终validation选定配置。
- `超参数实验说明`：四轴单变量实验的共同参考点、口径和限制。
- `HP_数据集_主表/阶段表`：每个数据集四组单变量曲线的完整数值；后面附已完成组合和相对最早默认配置的改变轴。
- `55项配置索引`：原先紧凑汇总表保留，便于查找已跑配置。

单变量实验的参考点不是刚选定的最终配置：

| 数据集 | 固定参考容量 | 固定参考LR倍率 | Dropout | Weight decay |
|---|---:|---:|---:|---:|
| METR-LA | 8 | ×1 | 0.1 | 1e-4 |
| PEMS-BAY | 16 | ×1 | 0.1 | 1e-4 |
| PEMS03 | 20 | ×1 | 0.1 | 1e-4 |
| PEMS04 | 10 | ×0.5 | 0.1 | 1e-4 |
| PEMS08 | 6 | ×0.5 | 0.1 | 1e-4 |

各参数组均固定其余三轴，检查完整主配置与嵌套专家配置，除run标签和该参数组外无其他变化。学习率三个参数组同比例缩放。训练/验证数据身份、seed32、Focal和验证选型准则相同。47个独立参考/单变量配置＋8个其他组合覆盖全部55项；参考点在四组中重复列出，但不是重复训练。若以后要求围绕新最终配置重做四轴曲线，须重新核对覆盖，不能直接换参考标签。

- `PAPER_HYPARAMETER_EXPERIMENTS.md`：同工作簿的论文用完整数据。
- `paper_hparams_full_cells.csv`：可直接作图的长表，2310行含参考点重复展示。
- `paper_hparams_registry.json`：55项配置坐标、参考点、相对初始配置改变的参数、权重路径，后续组合实验去重依据。
- `paper_hparams_verification.json`：单变量差异、身份、覆盖和Excel回读核验。
- `MERLIN_FAULT_AND_FOCAL.md`：按最新选定配置重算的Fault诊断，区分事实与因果假设。

复现顺序：先运行 `scripts/collect_hparams_final.py` 生成结果表，再运行 `scripts/organize_hparam_ofat.py` 补全论文超参数分组；诊断由 `scripts/diagnose_selected_fault.py` 生成。全部为离线结果处理，不执行推理或test。`final_tables_before_ofat_20260930.xlsx` 是整理前工作簿备份。
