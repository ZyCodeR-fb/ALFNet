# 按节点身份组织的机制图

此版本按用户接受的实验设计，离线重绘既有 plan1dcdf704 的固定权重验证集预测，没有训练、推理或新 test。原图及原预测完整保留。

- [图3](trajectory_cohorts.png)／[PDF](trajectory_cohorts.pdf)：保留节点接入、已知节点中断，分别展示绝对 MAE 与相对各自完整历史的 MAE 增量。完整历史是成熟活跃参照，不把它误称为 Join-12。
- [图4](layout_cohorts.png)／[PDF](layout_cohorts.pdf)：两类节点各64窗口，2×4面板、共同纵轴、保留全部布局与端点。已知节点13/14组合、保留节点4/14组合由ALFNet取得最低MAE。
- [图6](cases_truth_center.png)／[PDF](cases_truth_center.pdf)：按真实未来的均值、标准差和首末变化选择各组的中心窗口。规则与身份在[selection.json](selection.json)，全部曲线及误差在[case_predictions.csv](case_predictions.csv)。选择是事后描述性的，不是预注册；没有按模型输赢筛选。已知节点的三个面板为同一窗口，不是三个独立样本。
- [图3/4全部点值](plot_values.csv)，[独立核验](verification.json)。

原始完整三指标见[metrics.csv](../final_support_20260922/metrics.csv)，旧极端失败案例见[原图](../final_support_20260922/cases.png)。正文将普通MAE和去角色平衡按“训练目标”移至附录A.12，结构消融保留Mask-only等全部七项，所有数字未改，排名分别按所在表重算。

脚本：`scripts/plot_support_cohorts.py`。颜色与标记一致，ALFNet适度加粗；图3同一行共用纵轴，图4全部面板共用纵轴；案例显示完整预测曲线和每模型MAE。没有隐藏负增益、裁剪曲线或平滑预测。
