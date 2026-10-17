# Merlin / ALFNet Fault差异：事实、解释与未验证假设

来源：paper/validation_seed32_baseline_extension_20260927/results_and_provenance.json，全部validation、seed32、共同33格协议。没有新增推理或test。

| 数据集 | ALFNet相对Merlin Fault MAE变化 | RMSE变化 | MAPE变化 |
|---|---:|---:|---:|
| METR-LA | +3.41% | −0.94% | −0.49% |
| PEMS-BAY | +1.70% | −2.04% | −7.06% |
| PEMS03 | +30.35% | +24.87% | +52.98% |
| PEMS04 | +36.48% | +28.75% | +64.45% |
| PEMS08 | +18.53% | +11.12% | +15.38% |

正值表示ALFNet误差更大。“speed的Fault全部更优”并不准确：speed的MAE也是Merlin领先，只是ALFNet在RMSE/MAPE上有优势。flow的差距明显扩大，是需要解释的实质结果。

## Merlin优势与输入边界

[Merlin原论文](https://arxiv.org/html/2506.12459v1)的核心是清洁观测教师指导缺失观测学生，并对同窗不同缺失率表示进行一致性约束；STID骨干保留节点和日历embedding。当前源文件字节一致，adapter中四种缺失率.25/.5/.75/.9、隐藏/输出蒸馏和对比目标没有引用ALFNet的ALFGraph、ALFTemporal、角色/年龄或Focal。见BASELINE_FAIRNESS_20260928.md逐模型审计。

这些机制与已知节点历史中断相匹配：已有训练记录可学习节点特征和日历模式，清洁教师为缺失视图提供额外约束。相反，Merlin在Unseen/Join上的误差较高，说明已知节点鲁棒性不等于新节点泛化。这是与结果相容的解释，不是已通过消融分离出的因果结论。

必须披露：Merlin教师额外读取共同train-K范围内的自然可用历史，学生看到共同masked输入的进一步删减视图；从未读取V/T动态历史。这是源方法所需训练特权输入，并非所有baseline训练时获得完全相同张量。每阶段100/10，两阶段总预算可能高于单阶段。Merlin保留node-ID，而多项旧baseline采用结构编码，不能隐去该适配政策差异。

## 为什么不能直接判定是flow代码或KD-tree问题

1. 已核验flow和speed共用同一ALFNet/Focal实现；flow用目标q、speed用v，scaler只拟合train K，12→12、32分区与曝光/计数核验正常。旧三轮ALFNet源码哈希一致；runner仅增加Merlin注册。未发现可解释该差异的已证实shape/目标切片错误。
2. flow的BFS和speed的KD-tree确有布局差异，但同时改变了节点、图、目标单位、分布和训练长度；跨数据集比较无法隔离布局效应。没有同数据集受控替换，不应声称KD-tree能解决问题或为缺少坐标构造假WGS84。容量实验只测固定分区的上下文容量，不是KD-tree/BFS对照。
3. 已知代码层面的尺度差异：Focal元素为归一化绝对误差乘tanh(0.1×原单位绝对误差)。约14.72原单位误差时权重达到0.9。同样数字系数不代表speed和flow具有同样有效加权；flow大误差更易进入近似MAE区间。因此损失尺度敏感性是有代码依据的假设，但尚无受控训练证明其导致Fault差距。本轮不擅自改变已确定损失。
4. ALFNet在LifeAvg保持领先，而Merlin在已知节点中断明显领先，说明存在泛化/条件优化权衡，不能仅依据Fault判定整个flow实现错误。先完整报告条件级误差，再用不改变架构的容量/LR/dropout/WD实验检查敏感性。

## 可借鉴但本轮不混入的方向

优先考虑同窗多缺失视图预测/表示一致性，或train-K清洁教师约束；其价值在训练策略，不需要把ALFGraph替换成Merlin。若后续授权，应单独分离“多视图”“蒸馏”“额外训练预算”，保持共同验证协议并报告新增成本。当前只执行四类超参数补齐，不修改架构、目标函数或新增蒸馏模块。

## 公平设置的文献依据

[TFB（ICML2024）](https://arxiv.org/abs/2403.20150)强调可复现、灵活且公平的比较管线，[BasicTS](https://github.com/GestaltCogTeam/BasicTS)提供配置驱动的统一训练/评价。实践中应统一数据划分、可访问信息、预测任务、指标、验证选型和预算口径；不等于要求不同方法同宽度、同loss、同优化器或同学习率。Merlin论文表4明确给出自己的batch16、embedding64、dropout.15、KD/CL权重等；这些属于源方法配方，应保留并披露必要适配。当前MoGERNN patience3、不同早停起点与结构替换属于必须清楚报告/后续修复的差异，不能仅凭max_epoch100宣称已经完全公平。
