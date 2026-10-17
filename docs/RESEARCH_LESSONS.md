# ALFNet 定型过程与复用经验

本文件总结已完成探索，不是新实验计划。详细逐轮记录见
[任务记录](ai/handoffs/T-20260908-support-ablations/HANDOFF.md)；清理前源码、配置、报告及日志快照见
`archive/cleanup_20261017/sources_and_evidence.tar.gz`。旧权重删除清单和保留权重 SHA256 见同目录 `plan.json`。

| 阶段 | 尝试与观察 | 后续决定／可复用经验 |
|---|---|---|
| 早期 Full 与支持算子 | LF、SUP、组合配方及生命周期消融的收益随数据集变化；并非增加算子就稳定提高精度 | 不原样复用失败配方；区分具体配置失败和整个研究方向无效 |
| GWNet 与目标函数探索 | 探索大规模骨干替换、普通 MAE、尾部损失、频域损失和历史线性修正。MAE/MAPE 改善可能伴随 RMSE 代价 | 最终保留 ALFNet 架构和 MAE-Focal；没有接受 GWNet 作为最终骨干 |
| 基线适配与训练公平性 | 曾存在 MoGERNN patience=3、适配与协议冲突、GinAR 数值稳定性等问题 | 按最终来源核对训练预算、早停口径、节点划分；不能用目录名判断结果是否有效 |
| 容量与优化参数 | 单因素容量、学习率、dropout、权重衰减及组合搜索；单因素各自最佳不保证组合最佳 | 论文敏感性采用固定锚点，最终配置单独记录；容量是每分区 token 容量，不是分区数量 |
| flow 修复 | 归一化／相对误差目标、算子与低值修复未形成普遍优势；插值、解码残差倍率和中心化后取得更好结果 | 保留共同架构，按数据集设置输入处理及参数。流量尺度与低值 MAPE 敏感性是线索，不能当作已证明的唯一原因 |
| speed 迁移检查 | flow 的整套设置迁到 speed 并未稳定优于原配置 | METR-LA、PEMS-BAY 保留无插值、残差倍率0.3；flow 使用插值、倍率1；中心化强度按数据集设置 |
| PEMS03 | 多轮候选仍存在 validation/test 差异，未达到目标 | 正文采用四数据集；保留最终方案下 PEMS03 探索及诊断，不能把排除解释成该数据集无效 |
| PEMS08 最终更新 | WD=0.001 原为固定锚点 OFAT；后经授权 test 和新配对消融确认，更新最终配置 | 修改 Full 后，旧超参数消融不是新 Full 的配对证据；必须保留配置和权重身份 |
| 消融与机制图 | Routing、Spatial Context、Temporal-FiLM 在四数据集 test 三项 LifeAvg 均变差；Age 与 Refinement 有例外 | 论文展示子集与完整证据分开管理。删除展示行不等于删除权重或反向证据；图必须对应最终权重 |

## 当前冻结入口

- 最终参数和权重：`paper/pems08_wd_test_20261012/selected_configuration_manifest.json`。
- 比较结果：同目录 `selected_tables.json`、validation/test 主表和阶段表。
- 全量配对消融：`paper/pems08_wd_ablation_test_20261012/four_dataset_results.json`，更早十二类消融记录仍保留。
- 最终图和底层数值：`paper/final_figures_20261017/{validation,test}/`。
- 固定锚点超参数：`paper/final_supplements_20261008/`、`paper/final_four_tables_20261001/` 和 `paper/figure_hparams_test_20261017/`。

若配置保留曾使用已观察 test，论文不能将其称为未触碰 test 的独立选型。历史计划含旧源码哈希，清理后不可直接作为新运行授权；复核历史运行需在独立目录恢复快照，新实验另建计划。
