# 最终配置的METR-LA新版消融（2026-09-20）

Origin Skill: academic-research-suite / experiment-agent；plan→run。用户授权：两个指定候选test后选择最终配置，扩充并直接运行消融，最多5并行。正文除双数据集总体主表外使用METR-LA；PEMS-BAY已有结果完整保留供附录。

## 参数分析结论

现有容量、有效学习率、ALFTemporal dropout、AdamW decay四类敏感性足以覆盖上下文与优化；没有统一的论文参数数量配额。ALFGraph实际配置`moge_router_mode=soft`，在`LifecycleRouter.forward`走`_soft_gating`，混合全部5算子，`k`不被使用。单独扫top_k是无效实验。soft/static/hard routing属于机制选择，不把改变router模式伪装成当前模型top-k调参。本轮用static对照直接检验动态路由，不增加无效top-k训练。

## 文献依据（原文核验）

- [AGCRN，NeurIPS2020 §4.4](https://arxiv.org/pdf/2007.02842)：GCGRU、NAPL-GCGRU、DAGG-GCGRU及独立嵌入变体，逐项分离模块和共享策略。借鉴受控替换，不将其他论文的效果当作本模型结论。
- [Graph WaveNet，IJCAI2019 Table3](https://www.ijcai.org/Proceedings/2019/0264.pdf)：恒等、前向、自适应、双向、双向＋自适应五类图配置，验证空间关系来源。借鉴静态传播对照，不引入GWNet模块。
- [PatchSTG，KDD作者版本 §5.3](https://arxiv.org/html/2412.09972v1#S5.SS3)：移除depth/breadth attention、改变填充和分区方式等。原文起句称four variants但实际枚举更多，引用具体内容，不照搬该数量。借鉴局部上下文和整分支对照，不改变已冻结布局算法。
- [STID，CIKM2022 §4.4](https://finleywang.github.io/assets/pdf/2022_CIKM_STID.pdf)：分别去空间、日内、周内标识，并以单一数据集展示消融。借鉴将age与完整state分别检查，以及正文使用一个代表数据集。

## 参考配置与结果身份

先等待`evaluate_metrla_two_candidates.py`两个完整validation复核和test结束，再由其事先冻结的选择规则确定original/lr_double/patch12。新计划记录selection.json SHA及唯一reference checkpoint/config/table。

本次选型是用户明确要求的test参与配置选择，不能称独立test确认或仍遵循上轮validation门槛；要保留两候选完整test及旧表。消融本身只训练和评价validation，不再自动访问test，不根据消融结果调整最终模型。将来如果要求test消融，须明确授权对应全部固定变体，不能仅测试有利者。

所有新消融从seed32初始化，复用最终Full validation对照；不从其best微调。METR-LA、12→12、原时间/节点划分、scaler、观测支持协议、train/validation窗口与target mask不变；batch64、100轮上限、40轮起earlystop/patience10和TF线性40均不变。学习率、容量、dropout和decay继承最终选中配置，不针对消融单独调参。

## 九项消融与问题

| key | 修改 | 其余保留／解释 |
|---|---|---|
| mask_only | 显式role/age输入替换为完整mask重复，输出修正角色尺度取平均 | 旧消融重跑；真实标签仍用于损失和TF；参数量不变 |
| no_age | 仅age输入替换为mask，role保留 | 分离年龄与状态角色贡献；参数维度不变，不改真实事件标签 |
| static_router | ALFGraph改为每卷积共享的可学习静态5算子凸组合 | 旧消融重跑；算子、编码器、解码器不变，报告参数量差异 |
| local_context | ALFTemporal用已有逐节点local执行路径 | 旧消融重跑；保留时间表示及FiLM，删除该分支跨节点动态交互 |
| no_imputer | 移除学习式GRIN历史重建，缺失输入填标准化0，真实观测原样保留 | 0对应训练scaler均值，不是原速度0；保留完整mask，不注入缺失真值 |
| no_temporal_film | 移除ALFTemporal与FiLM条件分支，gamma=1、beta=0 | 这是整条件分支消融，不单独归因FiLM；保留ALFGraph、重建、输出修正 |
| no_refiner | 移除图输出修正，直接返回graph预测 | 定量区分解码预测与后置图修正 |
| plain_mae | Focal目标换为普通MAE | role平衡/模型aux保留，仅检验Focal尾部重加权；不是原MSE Full |
| no_role_balance | role平衡混合beta从0.1改为0 | Focal基础目标与路由aux保留，仅检验角色平衡监督 |

9项来自7个结构/输入控制与2个目标控制；不是逐步累加删模块，不能将收益相加，也不要求Full每格必须第一。初值先完整构建再删分支，所有保留的同名参数与Full完全相同；静态router新增logits以0初始化。移除分支是真的无参数替代，不用仍被统计的闲置分支伪装删减。原Full自身已有的结构不在本轮顺便重构。

## 实现与预检

原model/layers和共享train_v4均不改。新增`experiments/alfnet_ablation_adapter.py`只在本研究launcher显式安装；study checkpoint以后必须用同一adapter和study_ablation配置重建，不能误用通用最终模型评估器。三个已有开关写入独立nested YAML，其他替代在创建optimizer前应用。所有原数据/TF/指标路径复用。

冻结check绑定源码、配置、数据、reference与本计划。每项真实batch64的2train+2validation预检要求无skip、有限loss/梯度/输出，验证共享初值、数据身份、实际optimizer LR及参数量；单元测试检查缺失真值不泄漏、age不改其他字段、移除分支无参数、soft top_k无效。预检全通过才启动9项完整训练，45秒错峰、最多5worker，显存预留按预检峰值+安全余量；失败保留现场并阻止待运行项，不自动kill/retry，不动他人进程。

## 输出与解读

`artifacts/experiments/metrla_final_ablations_20260920/<sha>/`保留权重、报告、配置身份和日志。全部完成自动写`paper/ablations_metrla_20260920/results.json`、`ablation_cells.csv`和`summary.md`。展示LifeAvg三指标、Unseen、Cold/Warm/Ready、Fault分阶段及参数量；报告所有变体的正负收益。

正文历史Full/MSE validation消融不冒充新结果，待新训练结束后替换；旧表/机制干预保留历史身份。单seed32不声称显著性或全局最优，不把33格当独立重复。PEMS-BAY本轮不新增消融，已有数据保留。无commit/push。
