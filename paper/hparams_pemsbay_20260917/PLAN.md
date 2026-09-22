# PEMS-BAY：固定Focal架构的超参数敏感性实验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run（按用户2026-09-17授权）
- Origin Date: 2026-09-17
- Verification Status: 配置/源码核验已完成；训练结论待运行
- Version Label: pemsbay_hparams_round1_v1

## 问题与已核验事实

研究问题：固定Focal架构、目标函数及训练协议，PEMS-BAY的空间容量和优化步长是否限制了validation性能？一次仅改一个因素，检验性能曲线和阶段收益，而不是解释性地把test排名差异归因给容量。

当前seed32 test并非PEMS-BAY“几乎都不是第一”：Focal主表7/15第一、阶段6/18第一，两数据集LifeAvg的MAE/RMSE/MAPE均第一。METRLA在全部33格有32格第一，BAY为13格第一；BAY的短板集中于部分Unseen/Steady/Join和Cold/Warm/Ready条件。排名是相对九个基线的结果，不能据此识别因果。

当前Git HEAD为`783cbf7603a1a43962c7f4821fc35d201b3bfae6`（用户已提交的“最终版v2”），开始时工作区干净，上游差异0/0。旧handoff的9346dc0/dirty状态已过期；本轮不提交或推送。

| 有效配置 | METR-LA | PEMS-BAY |
|---|---:|---:|
| 节点数 | 207 | 325 |
| 原布局分区数 | 32 | 32 |
| 最大原分区节点数 | 7 | 11 |
| `spa_patchsize`容量 | 8 | 11 |
| 展开空间token数 | 256 | 352 |
| 相比原节点的额外token槽数 | 49 | 27 |
| Graph+共享imputer实际学习率 | 0.0005 | 0.0005 |
| Patch实际学习率 | 0.0001 | 0.0001 |
| Fusion/其余实际学习率 | 0.00005 | 0.00005 |
| AdamW weight decay | 0.0001 | 0.0001 |
| batch size / epochs上限 | 64 / 100 | 64 / 100 |
| early-stop起点 / patience | 40 / 10 | 40 / 10 |
| teacher-forcing线性衰减 | 40 epochs | 40 epochs |
| Patch层数 / embedding尺寸 | 1 / 64+32+32+32 | 同左 |
| attention dropout | 0.1 | 0.1 |
| 损失 | `mae_focal_raw02` | 相同 |

`runtime.model_recipe/model_variant`标签不同，但不是数值配方分支。Graph专家配置相同；Patch专家除数据路径/标签外的有效区别只有容量。顶层lr=5e-5被模型定义的三个optimizer group覆盖，因此只改顶层lr会形成无效扫描。

容量是每个固定分区容纳的节点token数，不是分区数。`model/ALFTemporal/ALFTemporal.py::_reorder_data`保留32个布局分区，按邻接权重选择实际节点补齐较小分区；不是补零，不重新划分。故最大分区+1会改变额外邻接上下文、重复token以及注意力计算，但它是否提高预测必须训练验证。PEMS-BAY容量不能直接设8（小于最大分区11）。增加容量并不增加本轮模型可训练参数；预检须验证形状、参数数目和相同seed初始化逐值一致。

数据集的节点规模、拓扑、训练动态与种子对应的未观测节点不同；METR-LA又经历过更多针对性开发。这些是潜在解释，不是已证实根因。首批不变更数据、布局、层数、宽度、模块、loss或生命周期比例。

## 文献依据（公开原文核验）

1. [PatchSTG, §5.6](https://arxiv.org/html/2412.09972v1#S5.SS6)：在代表数据集扫描层数1/3/5/7、输入维度32/64/128/256以及分区数量8至512，并报告不同数据集偏好的设置。借鉴其参数敏感性与规模分析方式；其“分区数量”不等同于我们的“固定分区容量”。本轮不扫描层数/宽度。
2. [AGCRN, §4及附录A.3](https://arxiv.org/pdf/2007.02842)：分析节点embedding维度与精度/参数量/训练时间关系，两个数据集最终embedding维度不同；明确学习率搜索范围和validation早停。说明同一模型允许数据集特定超参数，但不保证更大更好。
3. [Incrementally Improving Graph WaveNet Performance, §3及Table2](https://arxiv.org/pdf/1912.07390)：分别记录学习率衰减等训练设置的贡献，也报告许多未成功尝试。借鉴其单因素归因与完整负结果记录，不移植架构、不照搬其最优值。

这三篇不构成我们的最优性证明，其标准任务与本项目观测支持协议也不同。搜索仅发送公开论文关键词，没有上传本地数据/手稿。

## 首批明确运行范围：5个新run + 1个复用对照

全部PEMS-BAY、seed32，从头训练，不加载旧权重热启动。只复用现有Focal对照的已核验validation表作比较；对照已在release20项复算中通过。所有新run共用初始化、样本/教师强制随机种子、原12→12协议和scaler拟合范围。

| Run | 容量 | 三组lr倍率 | 唯一实验因素 | 空间token / 额外槽 |
|---|---:|---:|---|---:|
| baseline_reused | 11 | 1 | 已完成Focal，不重训 | 352 / 27 |
| patch12 | 12 | 1 | 最大分区+1，直接检验用户假设 | 384 / 59 |
| patch14 | 14 | 1 | 中等邻接上下文扩展 | 448 / 123 |
| patch16 | 16 | 1 | 更大上下文，检验收益是否饱和/退化 | 512 / 187 |
| lr_half | 11 | 0.5 | 三组lr同时减半，比例10:2:1保持 | 352 / 27 |
| lr_double | 11 | 2 | 三组lr同时加倍，比例10:2:1保持 | 352 / 27 |

其余配置逐字段等价；只允许标识/专家配置路径随新run变化。不开METR-LA训练，不自动追加未列出的候选。

## 选择与验证门槛（运行前固定）

- 每个run的best按完整validation 33格中LifeAvg MAE选，严格改善、平局保留更早epoch。训练预算上限100 epochs，早停起点40/patience10；不根据初期排名提前淘汰。
- 主要量：LifeAvg MAE及相对对照变化。报告全部33格MAE/RMSE/MAPE、Unseen和Cold/Warm/Ready均值；不只报告获胜格，也不把33个相关格当作33次独立重复。
- 可进入后续确认的强候选：LifeAvg MAE改善≥0.5%，LifeAvg RMSE及MAPE各退化≤0.5%，Unseen MAE退化≤1%，Cold/Warm/Ready的MAE算术均值不退化。
- 即使未通过门槛，全部参数点仍进入敏感性曲线；无候选通过则保留原配置。不自动替换已完成最终test表。
- 每run记录best epoch、epochs run、墙钟时间、峰值显存；并发墙钟包含争用，不能伪装为隔离单模型速度。训练skipped必须0，scaler/曝光/有效计数一致，正式运行不得截断train/validation batches。
- 这是单seed敏感性筛选，不能宣称全局最优或跨种子显著稳定。确认阶段应配对重跑基准与选出的候选；本轮不自动扩大种子范围。

## 后续实验顺序（设计草案，不自动运行）

1. 固定首批选定的参照，扫描attention dropout {0,0.05,0.1,0.2}与weight decay {0,1e-5,1e-4,1e-3}，各单因素；复用共同参照，不用巨型笛卡尔网格。
2. 如果容量和lr各有独立改善，再做一项组合与两项单因素的比较，检查交互作用；不能把组合提升全归因于容量。
3. 按既定配对规则确认稳定性，原配置和候选都用相同新seed。seed改变节点划分，须同seed成对比较，不与原seed32结果直接当作随机重复。
4. 后续转到METR-LA时迁移变量定义/相对容量规则和扫描方法，而不是照搬PEMS-BAY绝对容量；先固定范围再训练。

## Test边界与复现

本轮是在已查看seed32 test之后提出的补充探索，必须披露其时间顺序；即使训练选点只看validation，也不能称为完全不受此前test观察影响的预注册实验。现有`paper/test_seed32`和20份最终权重原样保留。新run不调用test、不会根据新test挑参数；若以后替换最终超参数，应单独确认评估方案，不能把重用的同一test包装成首次独立确认。

Python：`/home/xd/anaconda3/envs/ReaCH/bin/python`。工作目录为仓库根目录。新增配置在`configs/paper/pemsbay_hparams_20260917/`，独立plan冻结代码/配置/数据/基准权重/本方案SHA。仅新增实验脚本和配置，不修改原模型、训练器或已冻结release文件；继续从仓库根代码运行，不复制新的工作区。

执行入口：`python scripts/run_pemsbay_hparams.py freeze` → `check` → `preflight` → `queue`。preflight每项2个train+2个validation真实batch64，无保存权重；验证参数形状/初始化/scaler/曝光和实际三个lr一致。正式queue最多5项，错峰45秒，按探针显存预留30%及额外余量，不干预其他进程。每15秒检查进程存活；1小时无日志或48小时总时长只报advisory，不自动kill/retry。失败保留现场并阻止新的queued任务，已运行任务不随意中止。

输出：`artifacts/experiments/pemsbay_hparams_20260917/<plan_sha>/`，每项best/last/config身份/报告/log独立存放；全部完成后自动汇总`paper/hparams_pemsbay_20260917/results.json`与`sensitivity_cells.csv`。不会自动开启下一阶段或修改最终模型。
