# 协议汇报图

使用内置 imagegen 生成与定向编辑，未使用CLI回退。protocol_A.png 为时间与节点划分，protocol_B.png 为四场景及12步掩码，protocol_C.png 为训练与评价。图片供汇报，精确协议见上一级 experiment_protocol_for_advisor_cn.md。

## 初始提示词

### 图A

```text
Use case: scientific-educational. Create one polished 16:9 landscape Chinese academic presentation infographic, high resolution, white background, navy headings, teal/purple/blue accents, clear large Chinese typography, tasteful isometric road sensor illustrations and clean diagrams. Accuracy over decoration. No invented facts, no watermark. Title: 图A｜先分时间，再分节点. Top time bar exact proportions 60%,20%,20% labeled 训练 0–60%, 验证 60–80%, 测试 80–100%. Middle three distinct cards with miniature road networks: K 训练已知节点, V 验证保留节点, U 测试保留节点. Three separate non-overlapping sets. Small table text: METR-LA: K=157, V=25, U=25; PEMS-BAY: K=244, V=40, U=41. Bottom three permission cards: 训练：只读 K; 验证：只读 K∪V; 测试：只读 K∪U. Footer exact: 所有节点与路网拓扑始终存在；只有动态读数的权限改变. Side badge: 标准化仅使用训练时段的 K. Keep topology known not dynamically expanding. Must visually distinguish time split from node split.
```

### 图B

```text
Use case: scientific-educational. Create one polished 16:9 landscape Chinese academic presentation infographic, high resolution, white background, navy headings, teal/purple/blue accents, clear large Chinese typography, tasteful isometric road sensor illustrations and clean diagrams. Accuracy over decoration. No invented facts, no watermark. Title: 图B｜同一个预测任务，四种历史支持. Header: 输入过去12步 → 预测未来12步（5分钟/步）. Four large horizontal cards with sensor icons and exact binary strings in monospace, from oldest to newest: Unseen 零历史: 000000000000; Steady 持续观测: 111111111111; Join 接入 k=4: 000000001111; Fault 中断 k=4: 111111110000. Each string exactly 12 digits, digits1green digits0gray. Display a separate future target arrow to right, identical across cards labeled 未来12步目标仍需预测, not actual curves. Under cards two separate phase brackets, not joined sequentially: Join: Cold k=1 → Warm k=2…11 → Ready k=12; Fault: Onset k=1 → Decay k=2…11 → Outage k=12. Footer: 示例暂不含自然缺失；最终可见 = 自然有效 ∩ 阶段权限 ∩ 场景允许. Badge: k是事件进展步数，不是节点数. No Cold zero-history, no single six-stage chain.
```

### 图C

```text
Use case: scientific-educational. Create one polished 16:9 landscape Chinese academic presentation infographic, high resolution, white background, navy headings, teal/purple/blue accents, clear large Chinese typography, tasteful isometric road sensor illustrations and clean diagrams. Accuracy over decoration. No invented facts, no watermark. Title: 图C｜从场景训练到统一评价. Three columns with arrows. Left训练 with road sensor illustration and four equal wedges caption 四场景各25%; bullet labels 只用K构造代理任务, 每轮更新场景分配, 仅事件节点的有效未来参与损失. Center验证→测试 two separated boxes: 验证：固定场景，按LifeAvg MAE选权重; 测试：固定权重，不更新参数. Small caption: Unseen / Join：保留节点; Steady / Fault：已知节点. Right统计 ladder: 样本误差 → 日历平衡 → k平衡 → 四条件等权平均; formula LifeAvg = (Unseen + Steady + Join + Fault) / 4. Footer strip: MAE · RMSE · MAPE | 主表与配对干预分别回答整体表现与输入响应. Small warning badge: 每个滑窗分配一个场景；不是每个事件连续回放12次. No assertion test never previously seen; no fabricated statistical results. Clear visual information hierarchy.
```

## 定向修订

A：中间路网仅表示K/V/U身份，不表示读数权限；删除有／无动态读数图例。

B：改为本窗口无可见历史／不加事件遮挡；阶段副标题只描述步数，不表示稳定、恶化或在线适应。

图B二次修订：只将第四行Fault的副标题改为“本窗口末4步隐藏，前8步可见”，第三行Join与所有掩码保持不变。

C：k按事件进展步数平均；日历按五时段和5/7、2/7；删除气象、事故输入暗示；保留节点计分限定验证测试。
