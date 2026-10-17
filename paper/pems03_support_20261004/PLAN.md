# PEMS03统一支持方案的参数迁移

2026-10-04最新用户授权：接受插值＋倍率＋中心化为同一参数化方案，各数据集可用不同超参数；补跑PEMS03多个配置，若达到目标考虑保留数据集。最多5训练并发。本计划仅训练和validation，不自动test或正式纳入正文，不重训八个基线。

统一计算规则：在过去12步内按实际可见点插值，边缘就近延拓；全无历史仍用GRIN。中心为alpha×最后可见值，全无历史中心0；输入/teacher减中心，预测加回后进入原refiner。decoder residual scale固定1。中心强度alpha=0/.5/1为同一公式的参数，alpha0表示不中心化。插值本身是处理规则，不是连续数值参数；论文应区分规则和其强度/开关，不能把任意新模块都称为调参。

本轮6项：中心强度{0,.5,1}×学习率倍率{1,2}，完整矩阵。分别原样使用已核验bridge_delta1、triple_halfcenter、triple适配器，不改旧源文件、模型参数、损失、mask或数据划分。PEMS04已选alpha0/LR1，PEMS08已选alpha.5/LR1；同族配置合法，不要求相同固定参数。代码中不同variant名称映射到相同参数化公式，无数据集专属分支。

固定PEMS03容量16、32分区（不是16分区）、dropout.1、weight decay1e-4、batch64、seed32、maxepoch100、patience10、earlystart40、TF40。LR1图/时间/gate分别5e-4/1e-4/5e-5；LR2同时乘2。保留原PEMS03_capacity_mid作为复用控制。

历史核验：此前PEMS03已做unit-Focal/no-role/no-refiner/reconstruction/degree-SUP/relative20；支持迁移、统一中心化及LR交叉三轮仅有04/08，没有PEMS03上述六项。故本轮不重复旧配置，不带入失败损失方案。

先每项20训练+4验证batch的GPU预检，校验初始化/参数身份、学习率/dropout、32分区、完整协议、无跳过batch。正式六项采用显存门控和错峰调度，最多5并发。完整validation LifeAvg MAE选checkpoint，比较33格及三项LifeAvg。目标为至少27/33严格优于八基线且三项LifeAvg分别优于八基线最佳，不增加其他硬性退化门槛。并列不当严格胜，全部配置结果保留；若无满足者如实报告。

数据集属于同一研究，不能因结果好才把选择过程称为事先固定benchmark；PEMS03旧结果保留，正文是否恢复另行记录。新test仅在完成validation选择并冻结独立评估身份后进行。该轮为单seed、历史test反馈启发的探索，不声称统计显著性或完全独立确认。
