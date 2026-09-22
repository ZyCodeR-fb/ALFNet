# METR-LA固定Focal模型：四类超参数敏感性

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-20
- Verification Status: 设计冻结，正式结果待训练
- Version Label: metrla_sensitivity_v1

## 目的与边界

用户已经确定最终模型及test结果。本轮补充METR-LA超参数敏感性，最多5并发，模型架构和Focal损失不变；不自动重选最终模型、不运行新test、不改变已有PEMS-BAY容量16或METR-LA容量8主结果。不重训基线、消融或历史干预。

检验四个明确机制：空间邻接上下文范围（容量）、优化步长（三组学习率共同倍率）、随机正则化（ALFTemporal attention/residual dropout）、参数收缩（AdamW weight decay）。四因素不是顶会硬性数量要求，而是当前冻结架构下互补且确实生效的配置。

## 公开论文设计依据

- [PatchSTG §5.6，KDD论文作者版本](https://arxiv.org/html/2412.09972v1#S5.SS6)：扫描层数1/3/5/7、维度32/64/128/256、分区数量8至512，并展示不同数据集差异。借鉴分离参数曲线与资源比较，不把其分区数量误作本项目容量，也不跟随改变冻结网络深度/宽度。
- [AGCRN，NeurIPS 2020，§4与附录A.3](https://arxiv.org/pdf/2007.02842)：重点展示embedding维度敏感性，同时给出学习率搜索范围、不同数据集取值、参数量与成本。说明正文敏感性图与选参范围可以分开披露，不存在统一的参数数量配额。
- [STAEformer，CIKM 2023，Implementation](https://arxiv.org/pdf/2308.10425)：披露嵌入宽度、层数、头数、学习率、batch及validation早停设置。本轮借鉴完整训练设置披露，不将其“参数不敏感”的作者陈述直接转移给ALFNet。

上述为方法设计参考，不能证明本模型最优值或鲁棒性；四因素与范围属于本项目的预先固定选择。未将未发表数据或手稿上传外部服务。

## 运行矩阵

共同对照：`artifacts/models/main/METRLA/ALFNet/seed32/`原Focal，容量8、lr倍率1、dropout0.1、weight decay1e-4。此前全量validation复算已通过，复用checkpoint与其完整33格表。所有新run从seed32相同初始参数训练，不从best微调。

| 因素 | 全部分析值（含共同对照） | 新run | 其余参数 |
|---|---|---|---|
| 空间容量 | 7、8、10、12 | patch7、patch10、patch12 | 固定 |
| 实际学习率倍率 | 0.5、1、2 | lr_half、lr_double | 固定 |
| ALFTemporal dropout | 0.05、0.1、0.2 | drop05、drop20 | 固定 |
| AdamW weight decay | 1e-5、1e-4、1e-3 | wd_low、wd_high | 固定 |

共9个新训练而非108点笛卡尔积；每条曲线都与原配置直接比较，不在各曲线间逐步移动参照。9项队列最多5并行，空位后自动接续。PEMS-BAY本轮不新增训练，其已完成容量/lr五项结果作为独立验证集分析保留；两数据集的正则化曲线覆盖不完全，需明确报告。

32个原布局分区不变，最大原分区7；容量7/8/10/12对应224/256/320/384个空间token，额外槽17/49/113/177。补入邻接节点不等于增加独立节点，不改变布局成员关系。容量7是合法下界，不测试更小值。层数、宽度、router factor、模块、loss、观测支持场景配额不变。

三组有效基准lr依次为graph+shared imputer5e-4、patch1e-4、fusion及其余5e-5，倍率同步应用；仅改顶层lr会被模型group覆盖。dropout改变已有Dropout.p，不增加模块；weight decay应用于实际全部AdamW参数组。全部候选可训练参数数目、形状、初始化SHA必须与基准一致。

## 固定协议与分析

- seed32，validation设计seed12026，12→12，原60/20/20划分；只构造train和validation，训练已知节点scaler；同数据/拓扑/曝光/样本合同。
- batch64、max_epoch100、teacher forcing线性40轮、early stopping起点40、patience10，原Focal raw02与其他目标不变。所有run完整train/validation，未达原早停条件不因初期排名淘汰。
- best：完整validation33格LifeAvg MAE严格改善，平局取早epoch。报告全部三指标、主条件、阶段、best epoch、训练轮数、总墙钟/轮数、峰值显存。并发时间含竞争，不作隔离测速结论。
- 主敏感性图：四面板，每个参数值的LifeAvg MAE/RMSE/MAPE相对共同对照变化（%）；对照零线、原配置点标出，不画伪造误差条。容量同时列出token成本。另存全部33格CSV及JSON，失败也保留，禁止只挑好点作图。
- 辅助观察：Unseen、Cold/Warm/Ready平均MAE；沿用前轮筛选阈值只作描述标记（LifeAvg MAE≤−0.5%、RMSE/MAPE各≤+0.5%、Unseen MAE≤+1%、早期平均MAE≤0%），无论是否通过都不自动换最终模型。
- 单seed配对曲线不能宣称统计显著或全局最优。seed32固定了节点划分；未新增seed42/52，不把不同格当作独立重复。当前最终结果已被观察，因此本轮是事后敏感性补充，而不是盲态模型选择；不再访问test调参。

## 实施与运行身份

Python：`/home/xd/anaconda3/envs/ReaCH/bin/python`，仓库根代码运行，不复制工作区。
脚本：`scripts/run_metrla_hparams.py`（新增、复用既有训练函数，未改冻结训练器）。
顺序：freeze → check → preflight → queue；新计划位于`configs/paper/metrla_hparams_20260920/plan.json`。
每项预检2train+2validation真实batch64，检查0 skipped、同初值和参数形状、实际lr/dropout/weight decay、scaler与曝光身份。预检不保存训练权重。
最多5并发，45秒错峰，按实测显存留余量，每15秒记录进程存活和日志状态；1小时无日志/48小时总时长为advisory，不自动kill/retry。失败保留现场并阻止后续queued项。
输出：`artifacts/experiments/metrla_hparams_20260920/<plan_sha>/`。全部完成自动生成本目录results.json、sensitivity_cells.csv、四面板PNG/PDF、summary.md。不自动改写论文结论为未审核结果；实验稿先列明确设计与待完成状态。
