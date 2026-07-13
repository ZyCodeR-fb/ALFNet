# ALFNet 导师汇报稿

## 一、开场与本次进展

老师好，这次工作主要回应上次提出的四个问题：模型不能继续无节制堆叠；论文需要形成清楚的 Introduction 主线；实验不能只停留在常规消融；评价要覆盖真实 sensing network 的生命周期状态。

最终模型命名为 ALFNet，即 Asynchronous Lifecycle Fusion Network。论文使用的可复现入口是 `configs/ALFNet/rf_alf_sd.yaml`。它与早期实验名 `rf_sub_sd` 对应同一套最终结构，但改用了论文名称 `ALFNet / ALFGraph / ALFTemporal`，并在统一异步生命周期协议下重新训练和测试。旧 `rf_sub_sd` 的 RoleMacro MAPE 为 11.2779，当前最终公平重跑结果为 11.2085。

这版模型的原则不是再增加第三个或第四个大专家，而是在两个原有专家的失配位置做有边界的替换和减法：图分支负责可靠的空间锚点，时序分支只提供经过过滤且受角色预算约束的残差修正。

## 二、Introduction 主线

现有交通预测方法通常默认 sensing network 的节点集合和观测条件相对稳定，但真实系统中的传感器会异步经历接入、积累历史、稳定运行、退化和离线。相同节点在不同时间可能只有零历史、部分历史或被缺失污染的历史，因此静态的“已观测节点/未观测节点”二分法不能完整描述真实预测难度。

本工作把异步生命周期分成六个可解释状态：Cold-start、Warm-up、Adapted、Degrading、Outage 和 Steady。模型需要同时处理三个矛盾：低可见状态需要依赖图邻域恢复历史；有一定历史时需要利用局部时序模式；时序分支在缺失严重时又容易被噪声 patch 误导。

因此本文的核心问题是：如何在动态 sensing network 中，根据节点当前的生命周期阶段，自适应地协调空间迁移与时序证据，并阻止不可靠时序信息破坏图预测锚点？

ALFNet 的回答是“graph-anchored, temporally corrected”：先由 ALFGraph 给出稳定图预测，再由 ALFTemporal 产生经过结构化注意力和关系过滤的时序候选，最后只允许它在生命周期预算内修正图预测。

## 三、按架构图讲解模型

### 1. 输入与生命周期状态编码

图左侧输入包括历史交通序列 `X`、缺失 mask `M`，以及节点的生命周期信息。生命周期信息由角色 `R∈{VS,AAS,NAS,FS}`、历史年龄 `a` 和可见比例 `v` 组成。ALFNet 取当前窗口最后时刻的角色、年龄和可见比例，经过 MLP 得到节点状态向量 `z`。`z` 不直接生成预测，而是作为融合门的条件，告诉模型当前节点处于哪一种观测环境。

### 2. ALFGraph：图锚点专家

ALFGraph 是上方绿色分支，包含四个连续模块。

第一步是 GRIN Imputation Adapter。这里提取了 GRIN 中的 `GRIL / BiGRIL` 插补核心，而不是接入整个 GRIN 模型。项目中的 `SpatialConvOrderK`、`GCGRUCell`、`GRIL` 和 `BiGRIL` 按参考实现复制；本项目只增加张量布局、生命周期 mask、额外角色特征和输出拼接适配。

每个 GRIL 方向包含两级插补。第一级由图循环隐藏状态通过 `1×1 Conv` 产生 `x_hat_1`，先填补当前缺失值；第二级把第一次填补值、mask、隐藏状态和生命周期特征送入 `SpatialDecoder`，通过正向与反向图支持传播邻域信息，得到 `x_hat_2`。随后将第二级结果用于更新 GCGRU 隐状态。前向 GRIL 按时间正序执行，反向 GRIL 对反转序列执行相同流程；两侧的表示与 mask 经 MLP merge 得到最终插补 `X_imp`。最后使用 `torch.where` 保留所有真实观测，只替换缺失位置。

第二步是 Slim MoGE Block。原 MoGE 有多种统计和图聚合专家，最终模型保留 Mean Expert 与 Diffusion Expert，并使用 top-1 sparse router，温度为 0.8。Mean Expert 提供局部稳定平均，适合噪声较高或邻域变化较小时的保守恢复；Diffusion Expert 沿图随机游走支持传播信息，适合缺失节点从邻近传感器迁移空间模式。router 对每个节点选择一个专家，再将结果送入后续 STG 编码。架构图只展示最终启用的两个专家，不再绘制被删除模块。

第三步是原有 STG Encoder。它保留 DCRNN 图递归编码主干，将已恢复历史编码成图时序隐藏状态。这一段没有为了创新而重写，作用是维持原 MoGERNN 已验证的空间递归能力。

第四步是 Residual-Delta DCRNN Decoder，这是图中需要重点讲的结构变化。普通直接解码让线性头重构完整未来值；现在每个预测步先由 DCRNN 隐状态产生变化量 `Δ_p`，再通过显式跳连递推：

`x_(t+p) = x_(t+p-1) + 0.3 Δ_p`

第一个预测步以最后观测/插补值作为起点，后续步以上一预测值作为起点；训练期间仍保留 teacher forcing。这样解码器主要学习短期变化，而基准水平通过 skip connection 直接传递，降低零历史或不完整历史下的绝对值漂移。

### 3. ALFTemporal：受过滤的时序修正专家

ALFTemporal 是下方橙色分支。它保留 PatchSTG 的 ST Embedding、KDTree irregular spatial patching 和 Projection Decoder，但将原 depth/breadth dual attention 替换为 Crossformer-style Two-Stage Attention，并在其后接一层提取自 TimeFilter 的 patch filtration block。

ST Embedding 将交通值、节点身份和时间标记映射到 `D=64` 的 token。KDTree 按空间位置组织不规则传感器，把节点排列成 `P` 个 patch，每个 patch 含 `N` 个节点位置，形成 `[B,T,P,N,D]` 表示。

Crossformer TSA 的代码来自 Crossformer 的 `FullAttention`、`AttentionLayer` 和 `TwoStageAttentionLayer`，本项目只改类名，并把 PatchSTG token reshape 成 `[B×T,N,P,D]`。第一阶段对每个节点位置的 `P` 个 patch segment 做自注意力，用于学习 patch 之间的关系；第二阶段不再让所有节点两两直接通信，而是设置 `factor=4` 个可学习 router。`sender attention` 先把 `N` 个节点位置的信息汇总到 router buffer，`receiver attention` 再由这些 router 把跨节点信息分发回各 token。这个结构替换了原 dual attention，而不是在原 attention 外再并联一套大网络。

TimeFilter Patch Mixer 接收 TSA 输出并把 `N×P` 个 patch token 展平。它先通过两组线性投影学习 token affinity graph；`alpha=0.10` 用于裁去低分边。随后构造 temporal、spatial 和 spatio-temporal 三类结构 mask，Mask-MoE 根据当前 token 的关系分布选择这些 mask，`top-p=0.65` 控制保留的关系类型。筛选后的稀疏邻接进入多头 GCN，再经过 FFN 和 LayerNorm 得到 `Filter(X)`。最终只采用弱残差更新：

`X_out = X + 0.10 (Filter(X) - X)`

因此 TimeFilter 不是第三个预测专家，而是一层 patch 关系清洗器；即使过滤结果不稳定，它对主特征的最大直接改变量也受到 0.10 residual scale 限制。

### 4. 输出精炼与预算残差融合

ALFGraph 输出首先经过 Graph Diffusion Output Refiner。该模块对预测做一步正向和反向图扩散，将原预测、正向邻域预测和反向邻域预测拼接后映射为残差。残差经 `tanh`、角色缩放和 0.05 全局系数后加回图预测。最终角色系数为 AAS=0、NAS=1、VS=1、FS=0.25，因此它主要修正缺历史节点，而不扰动稳定 AAS。

ALFTemporal 输出经过 Temporal Conv Output Refiner：`Conv1D(k=3) → GELU → 1×1 Conv` 产生局部预测残差，再乘角色系数和 0.05。角色系数为 AAS=1、NAS=0.25、VS=0、FS=1；VS 完全关闭该修正，NAS 只保留四分之一强度。

最终融合使用 Budgeted Role-aware Residual Gate。角色对应的 MLP 读取 `[z,Y_g,Y_p,|Y_g-Y_p|]`，得到 patch gate：

`s = sigmoid(g_R([z,Y_g,Y_p,|Y_g-Y_p|]))`

`beta = b_R s`

`Y_hat = Y_g + beta (Y_p - Y_g)`

角色预算为 VS=0、NAS=0、AAS=0.20、FS=0.35。它与 `Y_hat=alpha Y_g+(1-alpha)Y_p` 完全等价，其中 `alpha=1-beta`。VS/NAS 的最终预测严格以图分支为锚点；AAS/FS 允许时序候选在上限内修正图结果。这里的关键不是手工固定最终融合权重，而是固定每个角色允许使用时序残差的最大预算，预算内部的 `s` 仍由样本条件动态学习。

### 5. 端到端目标与模型选择

最终训练目标的准确形式是：

`L = 0.90 L_f + 0.10 L_rb + 0.05 L_route + 0.03 L_regret`

`L_f` 是普通 MSE；`L_rb` 分别计算 VS、AAS、NAS 和 FS 的预测损失再平均，避免多数角色主导训练。`L_route` 使用 hard BCE 监督 `alpha`，但在 residual fusion 中比较的是图锚点 `Y_g` 与“使用完整角色预算后的时序修正候选”，而不是未经约束的原始 PatchSTG 输出。`L_regret` 在最终融合误差大于两个候选中的较优者时产生惩罚。

checkpoint 按验证集 `val_role` 选择，不使用 async test RoleMacro 做早停，因此测试集不会参与模型选择。

## 四、最终实验结果

统一协议为 `max_epoch=100`、`patience=10`、`train_for_async_lifecycle=true`、`lifecycle_training_strategy=round3_event_only`，异步测试统一为 `events=4 / k=12 / seed=0`。

ALFNet 最终 RoleMacro MAPE 为 11.2085，MoGERNN 为 12.6629，PatchSTG 为 13.0302；相对下降分别约 11.49% 和 13.98%。

| Sensor State | MAPE | MAE | RMSE |
|---|---:|---:|---:|
| Cold-start | 15.5032 | 5.9865 | 9.2806 |
| Warm-up | 12.6511 | 4.2827 | 7.5711 |
| Adapted | 9.9011 | 3.7028 | 6.3200 |
| Degrading | 11.8737 | 4.0405 | 7.6093 |
| Outage | 9.9623 | 3.4878 | 6.6911 |
| Steady | 9.1483 | 3.2783 | 6.1551 |

在 `6×3=18` 个 sensor-state 指标中，ALFNet 对 PatchSTG 为 18/18 全部更优，对 MoGERNN 为 15/18 更优；在全部对比模型中取得 14/18 项最优。需要如实说明的剩余问题是 Warm-up：原 MoGERNN 在该状态的 MAPE/MAE/RMSE 仍更低；此外 Degrading RMSE 的最优值来自 SLPF。这个结果不影响整体最优结论，但说明新传感器刚接入阶段仍是后续研究空间。

## 五、特色实验：Lifecycle Stress Map

Lifecycle Stress Map 不放入模型架构图，而作为独立实验图。为避免横纵轴来自同一个 mask 而产生数学共线，横轴定义为 lifecycle age，即事件发生后经过的 evaluation window 数；纵轴定义为当前输入窗口的 visibility ratio。

横轴分桶为 `0 / 1-3 / 4-6 / 7-9 / 10-12 / 13+`，纵轴分桶为 `0-25% / 25-50% / 50-75% / 75-100%`。每个格子显示 ALFNet 相对 MoGERNN 的聚合 MAPE 提升、样本窗口数 `n` 和平均 `beta=1-alpha`。提升率按格子内均值计算：

`Improve = (mean(MAPE_base) - mean(MAPE_ours)) / mean(MAPE_base) × 100%`

建议主文展示全状态 improvement map，再补充 Cold-start、Degrading 和 Outage 三张状态图。需要验证的不是“颜色是否好看”，而是两个可证伪命题：低 visibility 和事件初期区域是否有更高提升；模型在不可靠时序区域是否按设计保持更小的 `beta`。

更新后的诊断代码会在 `test_async` 结果中记录每个窗口、每个角色的 `alpha_mean`。因此需要重新执行测试推理，但不需要重新训练。旧结果仍可运行脚本，不过叠加的 beta 只能回退到角色级全局均值，不适合作为细粒度行为证据。

## 六、Lifecycle Stress Map 运行命令

先用已有 checkpoint 重新生成带分窗口路由统计的结果：

```bash
python main.py -c configs/ALFNet/rf_alf_sd.yaml --mode test_async
python main.py -c configs/MoGERNN/METRLA_async_lifecycle_100.yaml --mode test_async
```

然后指定两个 `async_lifecycle_summary.json`：

```bash
python scripts/lifecycle_stress_map.py \
  --ours "predictions/<ALFNet-exp>/async_lifecycle-base-events-4-k-12-seed-0/async_lifecycle_summary.json" \
  --baseline "predictions/<MoGERNN-exp>/async_lifecycle-base-events-4-k-12-seed-0/async_lifecycle_summary.json" \
  --out-dir "reports/lifecycle_stress_map/all" \
  --x-axis lifecycle_age \
  --age-bins "0,1,4,7,10,13,inf" \
  --visibility-bins "0,0.25,0.5,0.75,1.0" \
  --overlay patch_residual_weight
```

按状态分别生成：

```bash
for state in Cold-start Warm-up Adapted Degrading Outage Steady; do
  python scripts/lifecycle_stress_map.py \
    --ours "predictions/<ALFNet-exp>/async_lifecycle-base-events-4-k-12-seed-0/async_lifecycle_summary.json" \
    --baseline "predictions/<MoGERNN-exp>/async_lifecycle-base-events-4-k-12-seed-0/async_lifecycle_summary.json" \
    --out-dir "reports/lifecycle_stress_map/$state" \
    --x-axis lifecycle_age \
    --sensor-state "$state" \
    --overlay patch_residual_weight
done
```

输出包括 `lifecycle_stress_rows.csv`、`lifecycle_stress_grid.csv`、`lifecycle_stress_improvement.png` 和 `lifecycle_stress_mape.png`。
