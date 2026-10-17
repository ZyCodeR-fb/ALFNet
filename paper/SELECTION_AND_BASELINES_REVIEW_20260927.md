# 选型规则与新增基线复核（2026-09-27）

本次为方法建议和公开代码核查；没有启动训练、打开test或改写既有选型表。

## 门槛的适用性

旧门槛：相对默认配置，LifeAvg MAE至少改善0.5%，RMSE/MAPE最多退化0.5%，
Unseen MAE最多退化1%，Cold/Warm/Ready平均MAE不得退化。最低MAE候选未通过则回到默认。
它是保护原有多指标收益的保守筛选规则，不是领域统一标准；这些百分比没有多种子方差、
显著性或业务代价校准。预先固定有助于约束选择自由度，但不能证明规则科学最优。
此前“最终最好候选”应理解为“旧约束下保留配置”，并非最低预测误差候选。

建议后续统一按 validation LifeAvg MAE选择checkpoint和配置；RMSE、MAPE、条件级误差
及胜格数完整报告，不设无依据的一票否决阈值。LifeAvg本身每个指标各有一个数，
不是已定义好的跨MAE/RMSE/MAPE综合分数。胜格数依赖所选基线且不反映胜负幅度；
主表与阶段表存在重复/相关信息，33格不是33个独立重复实验。
因此胜格数适合作为覆盖面说明，不作为唯一优化目标。

按最低validation LifeAvg MAE，建议候选：

| 数据集 | 配置 | MAE | RMSE | MAPE | 对MoGERNN胜格 |
|---|---|---:|---:|---:|---:|
| PEMS03 | 容量20/LR1 |31.5382|41.3182|33.4512|33/33|
| PEMS04 | 容量10/LR0.5 |37.8733|51.4145|34.8345|33/33|
| PEMS08 | 容量6/LR0.5 |34.2022|45.5565|26.0394|33/33|

三者是既有候选内、既定主指标下的最佳点估计，不是每个指标均优或统计显著性证明。
PEMS03容量20相对默认MAPE退化3.74%；PEMS08 LRhalf相对默认RMSE退化0.34%，
MAE收益仅0.20%。这些取舍继续披露。规则修订发生于看到validation后，须版本化记录，
不能回写旧冻结计划或称作原先预注册规则；后续基线使用一致主指标原则。
模型选择与性能确认应区分，参见[模型选择偏差研究](https://www.jmlr.org/beta/papers/v11/cawley10a.html)。

## 代码与发表核查

- Merlin，KDD2025：官方[代码](https://github.com/ChengqingYu/Merlin)有教师/学生训练入口、预测主干和损失实现；[KDD论文集](https://www.kdd.org/kdd2025/research-track-papers-2/)。第一类优先下载。需保留蒸馏/多视图训练，教师不能使用V/T动态信号；作者预训练权重不直接用于本协议。仓库可访问不等于已经通过适配审计。
- GinAR+，TKDE2025：[作者发表列表](https://finleywang.github.io/publications/)确认。此次未核验到明确对应+版本的独立公开实现；[GinAR仓库](https://github.com/GestaltCogTeam/GinAR)仍标KDD2024，不能将它当作GinAR+。按用户要求暂不新增+版本。
- SIML-MSIGN，TRC2026：[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0968090X26000525)及本地全文已查；未核验到官方公开代码，暂不新增。“未找到可核验代码”不等于证明作者没有公开。
- GenCast，AAAI2026：[正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38607)，[官方代码](https://github.com/suzy0223/Generalising-Traffic-Forecasting-to-Unobserved-Regions-GenCast)。第二类有更新且公开实现的候选，支持下载审查。但源方法包含天气信号、物理约束、空间分组；当前数据与输入条件未提供这些等价资产。若完整保留需要新增数据或改变比较范围，就保留现有第二类，不把删去核心组件的版本当成完整GenCast。
- DCDM：[IEEE论文](https://ieeexplore.ieee.org/document/11087699/)确认相关任务，本次未核验到对应交通方法的官方代码，不纳入下载建议；同名图像/医学仓库不相关。

建议第一类保留UIGNN/GinAR并新增审查Merlin；第二类保留MoGERNN/ST-FiT/SLPF，
GenCast只作为待适配性核查的更新候选；第三类维持STGCN/STID/PatchSTG/STBP。
SLPF和STBP保留既有适配标签，不将缺失的源训练过程描述成完整复现。
本次是定向检索，不能证明不存在其他新方法；论文可写“经典方法与近年来高水平发表的代表方法”，
不写“全部最新SOTA”，也不因未找到代码而说相关方向止于2025年。
