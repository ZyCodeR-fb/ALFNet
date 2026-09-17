# ALFNet 观测支持条件分析（seed32 validation）

## 结果

METR-LA LifeAvg MAE/RMSE/MAPE 为 4.638/6.807/14.626，相对最强基线分别降低 8.32%/7.32%/3.97%。PEMS-BAY 为 3.040/4.638/7.996，降低 5.50%/7.11%/4.26%。这两个数据集当前 LifeAvg 最强基线均为 MoGERNN。

METR-LA 的 Cold/Warm MAE 与 MAPE 较弱；PEMS-BAY 的 Unseen 三指标均落后最强基线。两数据集 Outage MAE 均非第一。完整排名和数值见 condition_results.csv，图中的最强基线逐单元确定，不代表同一个模型。

## 案例

每个数据集覆盖两个静态锚点与六个阶段。选择规则为每个条件的第一个合法 validation 调度样本及最小编号有效 cohort 节点，在查看误差前固定。灰色区域表示不可见历史；曲线展示未来真值、ALFNet 和 MoGERNN 预测。案例含 ALFNet 优势与失败情形，不能据此证明因果机制或总体排名。

case_records.json 保存原始图表数据、样本索引、节点、origin、mask 和检查点哈希。两模型共用相同有效输入、目标与 validation 调度；没有读取 test。

## 证据边界

原主表仍为 seed32、validation-only、mixed-source。METR-LA 附件 Warm/Onset 和 Decay/Outage 的列名错位；本报告采用仓库命名。没有生成干预实验的结论，也没有修改模型来美化图形。
