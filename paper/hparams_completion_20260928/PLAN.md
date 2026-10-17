# 五数据集 ALFNet 超参数补齐（seed32，validation）

2026-09-28 用户授权：先完成容量、学习率、dropout、权重实验，既有有效结果复用，最多5个并发训练。权重沿用前轮定义为weight_decay，不改变损失权重/架构；test不在本轮范围。

## 单变量补齐与范围

METR-LA已完成容量7/8/10/12、学习率原始倍率0.5/1/2、dropout0.05/0.1/0.2、weight_decay1e-5/1e-4/1e-3，复用10条，不重复运行。
PEMS-BAY原容量11/12/14/16及容量11上的LR0.5/1/2已完成。当前容量16，补LR0.5/2、dropout0.05/0.2、WD1e-5/1e-3，共6项。
PEMS03原容量12/16/20及容量12上的LR0.5/1/2已完成。当前容量20，补LR0.5/2、dropout0.05/0.2、WD1e-5/1e-3，共6项。
PEMS04原容量10/14/18（LR1）及容量10上的LR0.5/1/2已完成。当前容量10、LR0.5，补容量14/18（LR0.5）、dropout0.05/0.2、WD1e-5/1e-3，共6项。
PEMS08原容量6/8/10（LR1）及容量6上的LR0.5/1/2已完成。当前容量6、LR0.5，补容量8/10（LR0.5）、dropout0.05/0.2、WD1e-5/1e-3，共6项。
合计31条历史ALFNet配置结果复用、24条新训练；不是额外全组合网格。曲线只连接其他设置相同的点，旧锚点系列另标，不混成单因素曲线。所有图/排名来自validation。

## 固定项与参数语义

架构、Focal、角色/年龄、SUP、输入12/输出12、seed32、节点分割、训练曝光K/验证K∪V、训练K scaler、validation设计seed12026不变。所有数据集32个分区；容量是每分区token数，不是分区数。
LR倍率同时作用于graph/patch/gate三组真实optimizer LR以及outer training.lr。dropout只改变ALFTemporal crossformer_dropout，与METR-LA前轮同义，不声称整个网络所有dropout均改变。WD应用于所有AdamW组。
max_epoch100、patience10，沿用最终ALFNet早停保护期40与teacher-forcing衰减40，以保持历史敏感性实验可比。基线不同early-stop/源模型改造问题单独审计，本轮不修改共享模型代码或将ALFNet保护期强加给基线。

## 复用与执行证据

逐条记录旧config/report/checkpoint哈希、最优epoch、33格计数、源代码差异。旧ALFNet/model/layers/lib/data_provider等代码哈希一致；runner差异仅新增Merlin注册（以Git源码移除此单行验证旧hash），不重新改写旧plan。旧plan全局hash过时不影响已经冻结、核验的历史证据。
新配置预检20个train batch＋4个validation batch，核对实际dropout/WD/三组LR、参数形状、曝光/scaler身份以及无跳过batch。每worker每epoch遇跳过batch或非法validation计数即失败，不静默产生缩减指标。
先按数据集轮转启动，最多5，按峰值显存加余量限制。单项失败保留日志，其他项继续；不静默重试。所有完成后自动汇总55个ALFNet配置的33格、三项LifeAvg、10基线相对第一格数、Fault与早期变化及Pareto候选。
不使用旧硬性退化门槛，也不以一个指标或首名格数隐去其他结果。自动报告最低LifeAvg MAE、最多第一格与三目标Pareto集合；取舍冲突留在报告，不自动覆盖既有主表/test或现有权重。

## 参考与披露

[TFB（ICML2024）](https://arxiv.org/abs/2403.20150)及[BasicTS](https://github.com/GestaltCogTeam/BasicTS)支持统一评估协议和透明、可复现的训练管线；不存在所有模型必须同LR/同宽度才公平的通用规定。源方法损失/优化器/学习率/结构参数应明确保留或披露例外。已有speed test曾用于探索性配置选择，本轮继续validation敏感性，不能宣称重新获得未触碰的独立test。
