# 最终模型的观测支持分析

正文图3、图4与图6使用最终 ALFNet/Focal、UIGNN、MoGERNN、STBP 的固定 seed32 权重；不训练、不打开 test。

- [图3：观测轨迹](trajectory.png) / [PDF](trajectory.pdf)
- [图4：数量与时间布局](support_layout.png) / [PDF](support_layout.pdf)
- [图6：预测案例](cases.png) / [PDF](cases.pdf)
- [完整分组指标](metrics.csv)：MAE、RMSE、MAPE，以及已知节点／验证保留节点／等权总体。
- [图表来源](provenance.json)、[独立数值核验](verification.json)。

计划：[final_support_figures_20260922.json](../../configs/paper/final_support_figures_20260922.json)，SHA `1dcdf70400b7444e946b7df0aa3bfba740bde7551f659c01fddc5a28af0507c1`。原始逐窗口预测、选样和完成标记位于[独立评估目录](../../artifacts/evaluation/final_support_figures_20260922/1dcdf70400b7444e946b7df0aa3bfba740bde7551f659c01fddc5a28af0507c1/)。

三类代表均按已有完整 validation LifeAvg MAE 选定：UIGNN（5.371565）优于 GinAR；MoGERNN（5.059101）优于 ST-FiT、SLPF；STBP（5.062040）优于 STGCN、STID、PatchSTG。“最优”在这里特指这一预定指标，不是所有指标的共同最优，也不是图中每一条件最优。

每类节点64个固定随机窗口，共128个。六种轨迹与14种不同的数量—布局组合，共20臂、四模型、10,240条预测记录。主图3仅用保留节点；图4采用已知与保留节点各半的总体。四个案例为随机序列前四个保留节点窗口，不按模型误差筛选。预测阶段教师强制为零，目标与非焦点节点输入保持不变，缺失掩码只删减自然存在的读数。

主要发现：ALFNet 在六种轨迹的三种获得最低 MAE，但 UIGNN 在两个 Join 轨迹更优，MoGERNN 在 Outage 更优；RMSE 的排序并不一致。在14个数量—布局组合中，ALFNet 有12个最低 MAE；Early 的 q=3、q=6 分别由 MoGERNN 和 STBP 领先。案例同时展示 ALFNet 的局部优势、速度低估和所有方法共同未捕获的速度下降。此处采用窗口平均与 pooled RMSE，不等同主表的 LifeAvg。

## 论文修订依据

已阅读本地 [MoGERNN](../../references/paper/MoGERNN.md) 的实验设置、模块消融与参数实验（§5.1、§5.5、§5.6），以及 [SIML-MSIGN](../../references/paper/SIML-MSIGN.md) 的实验设置、结果和结构消融（§3、§4、§4.1.3）。采用其按研究问题组织实验、先给设置与图表再解释差异的写法，不移植其数值、数据划分或结论。

Sciverse 补充检索了交通预测消融与案例分析。检索返回的 GLAFF 论文片段同时展示成功及失败案例，为案例呈现提供参考；本次未据检索片段引入新的性能论断或新增正式引文。Academic Research Suite 用于证据约束的行文和图表检查；未调用外部模型上传稿件。

正文删除了工作日志式说明和写作教程。运行身份与探索性配置比较的限制移至附录 A.10，保留而非掩盖测试反馈的使用。九项消融全部保留：按是否有利于 Full 删除行会改变证据呈现，不能作为合理的实验设计修订。架构、损失、超参数、主表和既有检查点均未改变。
