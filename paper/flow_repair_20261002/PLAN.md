# Flow定向修复：2026-10-02

## 决策及范围

用户本轮授权多种flow修复、最多5个训练并发，不重训任何比较算法；必要时可重训speed。本计划先仅训练PEMS03/04/08的ALFNet，共9项，seed32，从头训练，不启动speed、baseline或test。既有模型、表格与权重保留。

原比较表按剩余8个baseline重新审查67配置：METRLA C7、BAY C16/LR2/WD1e-5、03 C16、04 C10/LR1、08 C6/LR2。其余dropout0.1/WD1e-4。表格两split固定相同权重。PEMS04更新；PEMS03综合取舍保留8格配置，9格配置的LifeAvg三项反而均变差约3%–4%。这次旧结果重选使用已观察test，明确属于探索选型。

## 已核验事实与因果边界

- PEMS04当前test Unseen三项已优于剩余基线；PEMS08在validation的最佳配置Unseen三项领先，test却落后；BAY也有validation Unseen不足。因此并非所有flow差、所有speed好。
- `v4_lifecycle_loader._scenario_pools`：train Unseen/Join在K模拟，validation在V，test在T；评估还跨时间。flow无历史节点的绝对流量/道路容量异质性与这种迁移相容，但尚未由受控实验分离原因。已有训练每窗隐藏节点，不能把“完全没训练过无历史输入”当原因。
- 当前Focal为z*tanh(0.1*sigma*z)，z是标准化绝对残差。sigma约144–154的flow比BAY约9更早饱和为近似MAE；不是误差权重无上限地放大。新方案为z*tanh(z)，等效标准化Focal系数2，原单位同比缩放时目标不变。该系数是本轮固定工程候选，不宣称理论最优。
- 二值图的mean/weighted_mean/forward_diffusion支持重复，但各专家Theta独立，不能说整个专家输出完全相同。重复会影响解释性，不自动证明预测受损；不复跑旧LF/SUP。
- BFS与KD-tree同时伴随图、目标量纲等差异，没有同数据集控制不能归因于布局。容量是每分区token容量，分区32不变。无真实经纬度不能伪造WGS84。
- dense soft routing确实计算所有专家；top-k参数不会在soft模式产生稀疏加速，但这是当前设计，不是flow独有故障。
- MAPE在低值目标敏感；不更改零值掩码、阈值或统计口径追分。Unseen MAE/RMSE也差时，不能只归因于MAPE。
- METRLA消融支持保留跨节点时间上下文/FiLM/填补；角色平衡、细化是有收益交换的模块，不能把speed消融直接当作flow结论。

## 受控候选

每个数据集固定其当前选定容量/LR/dropout/WD、相同初始化；旧raw-Focal结果复用为对照。三个新候选：

| 候选 | 相对原版/候选A改动 | 检验问题 |
|---|---|---|
| A unit_focal | 原版只改标准化Focal | 量纲敏感加权是否限制flow泛化 |
| B unit_no_role | A的角色混合beta由0.1改0 | 按角色再平衡是否产生不利优化交换 |
| C unit_no_refiner | A的输出图细化scale由0.05改0 | 后处理空间残差是否损害异质节点预测 |

B/C分别与A比较；不能把B/C相对旧模型的差异单独归因于某一个模块。C保留参数/初始化与checkpoint形状，scale0使细化输出为恒等，不删除源模块。主干、SUP库存、路由、GRIN、FiLM、输入/数据协议不变。

新模块`experiments/flow_repair_20261002.py`仅在独立运行器内临时安装新loss名称/函数和方法身份。原代码不修改，无大目录复制。日后评估必须通过同一适配入口，避免将新checkpoint误作为raw-Focal解释。

## 预检与训练

继承共同train/validation、max_epoch100/patience10/早停起点40/TF40/batch64/4CPU线程；checkpoint仍按完整validation LifeAvg MAE。每项验证曝光哈希、train-only scaler、全部初始参数、32分区/容量、LR/WD/dropout，并执行20训练batch+4验证batchGPU预检。禁止skipped batch。新plan冻结配置/源码/数据/参考checkpoint SHA256。

最多5并发，显存预检门控、间隔30秒启动；失败保留日志、不无穷自动重试。queue结束产出9项33格完整validation和时间信息，不自动替换旧表，不自动test。

## 选型与解释

以8个保留baseline为范围，列出33格第一数量及LifeAvg MAE/RMSE/MAPE，并保留逐条件结果、Pareto取舍。80%对应至少27/33，只是目标，非保证。无额外硬性退化阈值；优先覆盖与平均误差共同改善，冲突时明确给出取舍。33格彼此相关，不等于33个独立显著性检验。候选检查点选择准则不变。

只使用新validation筛选；旧test已参与本轮假设提出，不能再声称其完全独立。新test需候选冻结后另获明确授权。

## 数据集建议及论文依据

建议暂保留5套数据，不为追80%删除或更换flow。两套speed正文、flow作为明确报告的外部泛化扩展是可讨论的叙事范围，若模型只对speed可靠则需要收窄标题/结论。没有“至少N个数据集”能保证论文质量的规则。

- Graph WaveNet（IJCAI2019）评估METR-LA、PEMS-BAY两套speed：[原论文](https://www.ijcai.org/proceedings/2019/0264.pdf)。
- DCRNN（ICLR2018）也使用两套交通网络数据：[原论文](https://arxiv.org/abs/1707.01926)。
- TrendGCN（CIKM2023）作者仓库列6套，四flow+两speed：[官方代码与实验说明](https://github.com/juyongjiang/TrendGCN)。混合类型亦有高水平先例，不需要强制同为speed。
- GinAR（KDD2024）的作者论文报告5套数据：[作者论文](https://finleywang.github.io/files/2024_KDD_GinAR.pdf)；并非只有speed。

METR-LA/BAY是有传感器位置的交通检测器数据，不是车辆GPS轨迹。为得到KD-tree坐标而更换数据集，会引入新任务/新baseline训练成本，不符合当前优先复用比较算法的约束。修复若无明显收益，保留原模型与flow边界结论，不启动无尽搜索。
