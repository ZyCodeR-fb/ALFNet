# PEMS-BAY超参数实验与patch16更新结论

五项训练均完整结束，训练/validation无跳过批次，checkpoint身份和完整历史中的最佳epoch已复核。只有patch16通过运行前固定的筛选门槛。采用 **PEMS-BAY Focal、空间容量16** 作为本次更新版；METR-LA保持原Focal容量8。架构、参数量、损失及三组学习率未变，变化只是固定32分区的空间token容量。原容量11的Focal及其表格保留作历史对照。

## Validation筛选

| 配方 | LifeAvg MAE | RMSE | MAPE (%) | MAE相对原Focal | 通过门槛 |
|---|---:|---:|---:|---:|---|
| 原Focal，容量11 | 2.964150 | 4.656433 | 7.733960 | — | 对照 |
| patch12 | 2.998468 | 4.704424 | 7.895783 | +1.16% | 否 |
| patch14 | 2.970974 | 4.678924 | 7.840531 | +0.23% | 否 |
| **patch16** | **2.912363** | **4.578590** | **7.536298** | **−1.75%** | **是** |
| lr×0.5 | 2.987420 | 4.706672 | 7.855707 | +0.79% | 否 |
| lr×2 | 2.968604 | 4.677559 | 7.931256 | +0.15% | 否 |

patch16最佳epoch77，共运行88 epochs；全部训练从同seed初始化开始。LifeAvg三指标改善1.75%/1.67%/2.56%，Unseen MAE改善1.68%，Cold/Warm/Ready MAE均值改善4.39%。33格中31格优于原Focal，只有Onset MAE/RMSE退化1.25%/1.23%。lr×0.5虽改善Unseen与早期接入，但整体退化，因此不能因局部收益替换主配置。

容量12并未改善总体性能，14也未通过；所以不能把结果概括为“最大分区+1必然更好”，也不能说容量单调越大越好。16只是本次已测范围内的最佳点，不是全局最优，也不足以因果解释两数据集排名差异。

## 已完成的test

候选在validation选定后，另一会话已经按用户授权完成一次全量validation复核和一次test。当前会话只审核保存结果并修复汇总，没有重跑推理。test LifeAvg：

| 指标 | 原Focal容量11 | Focal容量16 | 改善 |
|---|---:|---:|---:|
| MAE | 2.808320 | 2.757521 | 1.81% |
| RMSE | 4.359018 | 4.259742 | 2.28% |
| MAPE (%) | 6.646911 | 6.441370 | 3.09% |

test的33格中29格改善；Cold/Warm/Ready MAE均值改善0.99%。仍有退化：Onset MAE/RMSE +2.31%/+2.00%，Outage MAE/MAPE +0.64%/+0.20%。不能表述为全条件全面优于旧Focal。

## 与九个基线的排名

| PEMS-BAY表格 | 原Focal 第一/第二 | patch16 第一/第二 |
|---|---:|---:|
| validation主表（15格） | 5 / 7 | 7 / 7 |
| validation阶段表（18格） | 7 / 5 | 8 / 7 |
| test主表（15格） | 7 / 6 | 10 / 4 |
| test阶段表（18格） | 6 / 6 | 7 / 6 |

METR-LA全部记录与排名不变。两个数据集合计，validation第一从35/66增至38/66，test从45/66增至49/66。第一/第二按未舍入值、精确并列规则计算；格数不是独立重复或显著性证据。

## 表格和审计

- 新validation：`paper/validation_seed32_hparam_patch16_20260918/`。
- 新test：`paper/test_seed32_hparam_patch16_20260918/`。
- 各目录提供Excel、HTML、Markdown、660格CSV、结果身份及verification；第一加粗、第二下划线。
- 两个split共1320个数值的排名已用独立算法复核，80个Excel模型名称检查通过；每个split除BAY Focal外19条完整记录与原表一致。
- 原表`paper/validation_seed32/`及`paper/test_seed32/`未改写。旧test身份、scaler、有效计数与新候选一致；teacher forcing为0。
- 另一会话已完成评估，但汇总因`KeyError: UIGNN`中断，同时原发布函数未写Excel模型名。本轮新增`scripts/collect_patch16_update.py`修复离线汇总，冻结脚本/计划不改，原失败日志保留，queue_status已纠正为完成。
- 训练计划1ec318e9；候选评估计划eb8096d2。具体权重/config身份见候选计划与新表provenance；新表保留真实候选checkpoint路径，不冒用旧权重。
- 审计：`patch16_update_audit.json`。没有新增训练、test推理、commit或push。

这是seed32单种子、查看过原test之后进行的超参数补充探索。候选以validation选定，但后续使用的是同一test划分，不能包装成未受观察影响的首次独立确认；不据这次test进一步筛选候选。
