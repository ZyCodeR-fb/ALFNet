# 五数据集 baseline 补充实验（seed32，validation only）

授权：2026-09-27 用户要求 Merlin 五数据集，以及 PEMS03/04/08 当前最终代码缺少的对比实验；最多五个并发训练。测试集不在本次授权范围。

## 范围与复用

29 个新 run：Merlin × METRLA/PEMSBAY/PEMS03/PEMS04/PEMS08；UIGNN/GinAR/STFiT/SLPF/STGCN/STID/PatchSTG/STBP × PEMS03/04/08。
后24项有8项无历史seed32结果，其余16项属于旧微平均checkpoint选择，不能作为当前33格LifeAvg选择协议的结果。具体路径、旧/新指标和hash写入独立plan.audit。
复用26条：两speed数据集原20条，三flow数据集已确认ALFNet及复核通过MoGERNN共6条。原始表和权重保留。

## 公平性与配置

12→12；seed32；时间60/20/20；train K、validation K∪V；scaler只在train K拟合；固定validation设计seed12026。每个epoch完整33格validation，按LifeAvg MAE严格改善保存，平局保留早epoch。
各基线沿用已确认METRLA模型及训练recipe，只改flow数据目标/路径及必要布局尺寸。PatchSTG每个分区容量03/04/08=12/10/6，32个manifest BFS分区不变。UIGNN batch4，其余batch16，num_workers0，每进程4 CPU线程。
SLPF继续单阶段forecast适配；STBP继续离线结构编码适配，不声称完整原论文训练方法。不引入ALFNet角色/年龄条件、Focal损失或创新模块。

## Merlin 两阶段

教师/学生模型、MLP、CL源文件保持字节一致。按用户统一预算要求，两阶段各max_epoch100、patience10；Adam LR.0002、weight_decay.0001、batch16、clip5，保留源里程碑和.5衰减（超过100的节点不会执行）。每阶段独立best/last checkpoint和optimizer/scheduler/RNG，阶段及教师hash进入resume身份。教师最佳权重固定后才训练学生；合计最多200epoch，披露两阶段总计算预算，不声称与单阶段等算力。
教师训练只使用train K自然可用历史，监督为共同合法target mask；验证只使用共同masked输入。选定教师冻结后训练学生，四视图缺失率.25/.5/.75/.9，四项forecast及hidden/response KD求和、六对CL平均，保留源权重和epoch衰减。全节点CL投影之前屏蔽非K表示，杜绝heldout动态监督。
协议必要差异：train-K zscore代替源全数据minmax；规范有效mask代替零值sentinel；以共同动态支持及完整validation LifeAvg选择；源脚本自动test不执行。Merlin保留原node-ID参数；这不同于部分历史基线的结构编码适配，需披露。

## 执行与完成

独立脚本scripts/run_baseline_extension.py冻结完整源/配置/数据hash，所有29配置先做2batch真实训练+2batch validation预检；Merlin额外两阶段加载/冻结/保存预检。比较曝光、scaler、seed和静态图身份。
队列最多5个槽位，Merlin教师/学生顺序占同一槽位；按预检峰值显存加余量控制启动。任何worker失败停止新派发、保留日志，不静默重试。所有完成后自动检查best epoch、无跳过batch和33格计数，生成55条记录/1815格统一validation Markdown、Excel和CSV，第一名加粗、第二名下划线。
不启动test，不改ALFNet或手稿，不提交/推送。
