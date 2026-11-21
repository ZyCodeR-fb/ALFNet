# 深度学习论文实验结果可视化图鉴：123 种图形与选型指南

> **用途**：科研写作、超参数敏感性分析、算法性能比较、时空预测与强化学习实验展示。  
> **覆盖范围**：12 大类、123 种图形（第 1–8 类 75 种为 2026-10-12 版；第 9–12 类 48 种为 2026-11-16 补充，覆盖三维/多维图、组合图、结构与注意力图、时间序列个例图）；**每种图形均包含 1 张实际绘制的示意图**，附有 PNG 和 SVG。  
> **重要说明**：所有示例图均采用**模拟/演示数据（Synthetic Data）**，不是任何论文或真实数据集的实验结论。部分流程图、SHAP、t-SNE、UMAP、CKA 和 CD 图仅示意视觉形式，不代表已计算真实方法输出或统计显著性。

## 快速导航

1. [性能对比与算法排名](#1-性能对比与算法排名)
2. [训练过程与动态趋势](#2-训练过程与动态趋势)
3. [统计分布与误差分析](#3-统计分布与误差分析)
4. [超参数敏感性与搜索](#4-超参数敏感性与搜索)
5. [计算成本与性能权衡](#5-计算成本与性能权衡)
6. [预测质量与诊断](#6-预测质量与诊断)
7. [表示学习与可解释性](#7-表示学习与可解释性)
8. [时空预测与动态决策](#8-时空预测与动态决策)
9. [三维与多维参数空间](#9-三维与多维参数空间)
10. [组合图与多面板](#10-组合图与多面板)
11. [结构、关系与注意力](#11-结构关系与注意力)
12. [时间序列与时空个例](#12-时间序列与时空个例)
13. [论文实验图快速选择表](#13-论文实验图快速选择表)
14. [科研作图原则与常见误区](#14-科研作图原则与常见误区)
15. [补充依据与外部资源](#15-补充依据与外部资源)

## 图形类型全局概览

| 主要类别 | 图形数量 | 在论文中回答的问题 |
|---|---:|---|
| 性能对比与算法排名 | 9 | 我们的模型相对基线是否更好？ |
| 训练过程与动态趋势 | 8 | 是否收敛？是否稳定？训练行为是什么？ |
| 统计分布与误差分析 | 10 | 提升是否普遍？尾部与异常误差怎样？ |
| 超参数敏感性与搜索 | 9 | 方法对参数敏感吗？参数之间有交互吗？ |
| 计算成本与性能权衡 | 8 | 精度提升是否值得增加的资源成本？ |
| 预测质量与诊断 | 11 | 错误从哪里来？预测有没有偏差或失准？ |
| 表示学习与可解释性 | 10 | 模型特征、归因与决策机制怎样？ |
| 时空预测与动态决策 | 10 | 何时、何处修正有效？如何受预算影响？ |
| 三维与多维参数空间 | 17 | 多个参数或多个维度如何共同影响结果？ |
| 组合图与多面板 | 13 | 如何在一张图里同时给出均值、波动、显著性或局部细节？ |
| 结构、关系与注意力 | 8 | 模型学到了怎样的节点关系、注意力或组成结构？ |
| 时间序列与时空个例 | 10 | 具体的时间段、节点或缺失模式下模型表现如何？ |
| **总计** | **123** |  |

第 9–12 类的预览：

![第 9–12 类图形预览](overview_ext.png)

### 与三种基础图的区别

折线图（Line Plot）、柱状图（Bar Chart）和热力图（Heatmap）仍然非常实用。本图鉴更关注**如何用其他图形表达差异、分布、统计不确定性、联合参数空间与成本权衡**。例如，单纯平均 MAE 适合表格或点图；想证明绝大多数样本获益，应进一步展示 ECDF 或收益分布；想证明效率，应画精度—成本 Pareto 图。

---
## 1. 性能对比与算法排名

本类共有 **9** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 1.1 | **Dot Plot**（点图） | 不同模型的单指标对比 | 指标方向要一致；低值更好时不要反向排序。 |
| 1.2 | **Lollipop Chart**（棒棒糖图） | 消融或模型排名 | 若从 0 截断轴，应说明坐标含义。 |
| 1.3 | **Dumbbell Plot**（哑铃图） | 同一数据集上改进前后的差值 | 仅对同一测试条件下的成对结果连接。 |
| 1.4 | **Slope Chart**（斜率图） | 两个条件下的性能变化 | 只有两个条件时表达最清晰；避免标签相互遮挡。 |
| 1.5 | **Forest Plot**（森林图） | 效应量及置信区间 | 明确效应量定义、区间计算方法和样本独立性。 |
| 1.6 | **Waterfall Chart**（瀑布图） | 模块累计增益与损失 | 只有顺序添加模块时才能解释为累计增量。 |
| 1.7 | **Radar Chart**（雷达图） | 归一化后的多目标概览 | 多个指标必须归一化，且“越大越好”语义一致。 |
| 1.8 | **Critical Difference Diagram**（临界差异图（CD 图）） | 跨数据集平均排名；显著性需另行检验 | CD 连线必须来自合法统计检验；示意图不代表显著性。 |
| 1.9 | **Bump Chart**（排名变化图） | 跨任务算法名次变化 | 排名依赖候选集合；注意同名次与并列情况。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Dot Plot · 点图</b><br>
    <img src="figures/dot_plot.png" alt="Dot Plot 示例图" width="430"><br>
    <small>不同模型的单指标对比</small><br>
    <a href="figures_svg/dot_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Lollipop Chart · 棒棒糖图</b><br>
    <img src="figures/lollipop_plot.png" alt="Lollipop Chart 示例图" width="430"><br>
    <small>消融或模型排名</small><br>
    <a href="figures_svg/lollipop_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Dumbbell Plot · 哑铃图</b><br>
    <img src="figures/dumbbell_plot.png" alt="Dumbbell Plot 示例图" width="430"><br>
    <small>同一数据集上改进前后的差值</small><br>
    <a href="figures_svg/dumbbell_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Slope Chart · 斜率图</b><br>
    <img src="figures/slope_chart.png" alt="Slope Chart 示例图" width="430"><br>
    <small>两个条件下的性能变化</small><br>
    <a href="figures_svg/slope_chart.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Forest Plot · 森林图</b><br>
    <img src="figures/forest_plot.png" alt="Forest Plot 示例图" width="430"><br>
    <small>效应量及置信区间</small><br>
    <a href="figures_svg/forest_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Waterfall Chart · 瀑布图</b><br>
    <img src="figures/waterfall_chart.png" alt="Waterfall Chart 示例图" width="430"><br>
    <small>模块累计增益与损失</small><br>
    <a href="figures_svg/waterfall_chart.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Radar Chart · 雷达图</b><br>
    <img src="figures/radar_chart.png" alt="Radar Chart 示例图" width="430"><br>
    <small>归一化后的多目标概览</small><br>
    <a href="figures_svg/radar_chart.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Critical Difference Diagram · 临界差异图（CD 图）</b><br>
    <img src="figures/critical_difference.png" alt="Critical Difference Diagram 示例图" width="430"><br>
    <small>跨数据集平均排名；显著性需另行检验</small><br>
    <a href="figures_svg/critical_difference.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Bump Chart · 排名变化图</b><br>
    <img src="figures/bump_chart.png" alt="Bump Chart 示例图" width="430"><br>
    <small>跨任务算法名次变化</small><br>
    <a href="figures_svg/bump_chart.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%"></td>
</tr>
</table>

---

## 2. 训练过程与动态趋势

本类共有 **8** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 2.1 | **Learning Curve**（学习曲线） | 训练损失和验证损失随 epoch 变化 | 区分训练与验证曲线，并标注平滑策略。 |
| 2.2 | **Ribbon Plot**（带状趋势图） | 均值与波动区间 | 阴影必须说明是 SD、SE 还是 CI。 |
| 2.3 | **Area Plot**（面积图） | 累计指标随训练推进变化 | 面积图适合累计量，非零基线会改变视觉含义。 |
| 2.4 | **Stacked Area Plot**（堆叠面积图） | 多个损失分量的组成 | 只有可加和分量才应使用堆叠图。 |
| 2.5 | **Step Plot**（阶梯图） | 阶梯学习率或阈值策略 | 使用 post/pre 标签明确阶梯跳变时刻。 |
| 2.6 | **Spaghetti Plot**（多运行轨迹图） | 多个随机种子训练轨迹 | 多种子轨迹不要选择性只显示最好的运行。 |
| 2.7 | **Learning Curve vs Data Size**（数据规模学习曲线） | 样本规模与模型性能 | 不同规模下保持其他训练预算/条件可比。 |
| 2.8 | **Event-Aligned Plot**（事件对齐图） | 事件发生前后的指标演变 | 清楚标注事件时刻和时间对齐方式。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Learning Curve · 学习曲线</b><br>
    <img src="figures/learning_curve.png" alt="Learning Curve 示例图" width="430"><br>
    <small>训练损失和验证损失随 epoch 变化</small><br>
    <a href="figures_svg/learning_curve.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Ribbon Plot · 带状趋势图</b><br>
    <img src="figures/ribbon_plot.png" alt="Ribbon Plot 示例图" width="430"><br>
    <small>均值与波动区间</small><br>
    <a href="figures_svg/ribbon_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Area Plot · 面积图</b><br>
    <img src="figures/area_plot.png" alt="Area Plot 示例图" width="430"><br>
    <small>累计指标随训练推进变化</small><br>
    <a href="figures_svg/area_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Stacked Area Plot · 堆叠面积图</b><br>
    <img src="figures/stacked_area.png" alt="Stacked Area Plot 示例图" width="430"><br>
    <small>多个损失分量的组成</small><br>
    <a href="figures_svg/stacked_area.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Step Plot · 阶梯图</b><br>
    <img src="figures/step_plot.png" alt="Step Plot 示例图" width="430"><br>
    <small>阶梯学习率或阈值策略</small><br>
    <a href="figures_svg/step_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Spaghetti Plot · 多运行轨迹图</b><br>
    <img src="figures/spaghetti_plot.png" alt="Spaghetti Plot 示例图" width="430"><br>
    <small>多个随机种子训练轨迹</small><br>
    <a href="figures_svg/spaghetti_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Learning Curve vs Data Size · 数据规模学习曲线</b><br>
    <img src="figures/data_size_learning_curve.png" alt="Learning Curve vs Data Size 示例图" width="430"><br>
    <small>样本规模与模型性能</small><br>
    <a href="figures_svg/data_size_learning_curve.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Event-Aligned Plot · 事件对齐图</b><br>
    <img src="figures/event_aligned_plot.png" alt="Event-Aligned Plot 示例图" width="430"><br>
    <small>事件发生前后的指标演变</small><br>
    <a href="figures_svg/event_aligned_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 3. 统计分布与误差分析

本类共有 **10** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 3.1 | **Box Plot**（箱线图） | 中位数、四分位数与离群点 | 五数概括之外最好报告样本规模。 |
| 3.2 | **Violin Plot**（小提琴图） | 不同方法的误差密度 | 小样本 KDE 容易失真；可叠加原始点。 |
| 3.3 | **Raincloud Plot**（雨云图） | 密度、样本点与摘要统计 | 密度、散点与箱线摘要的量纲应一致。 |
| 3.4 | **Beeswarm Plot**（蜂群图） | 尽量避免点遮挡的个体观测分布 | 点数过多时可能拥挤，必要时分层抽样。 |
| 3.5 | **Strip Plot**（条带散点图） | 单个样本或重复实验结果 | 抖动只能用于显示点，不可改变真实数据值。 |
| 3.6 | **Ridge Plot**（山峦图） | 多条件下的误差分布 | 多个山脊共享同一横轴尺度，谨慎比较曲线高度。 |
| 3.7 | **KDE Plot**（核密度图） | 连续误差密度比较 | 带宽选择直接影响峰值数量和视觉形状。 |
| 3.8 | **ECDF Plot**（经验累积分布图） | 误差低于阈值的比例 | 误差越小越好时，曲线越靠左上通常越优。 |
| 3.9 | **Histogram**（直方图） | 误差频次或尾部分布 | 统一直方图区间，避免 bin 不同导致误读。 |
| 3.10 | **Q-Q Plot**（Q-Q 分位数图） | 检验误差分布与参考分布的匹配程度 | Q-Q 图不能单凭接近对角线证明模型充分校准。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Box Plot · 箱线图</b><br>
    <img src="figures/box_plot.png" alt="Box Plot 示例图" width="430"><br>
    <small>中位数、四分位数与离群点</small><br>
    <a href="figures_svg/box_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Violin Plot · 小提琴图</b><br>
    <img src="figures/violin_plot.png" alt="Violin Plot 示例图" width="430"><br>
    <small>不同方法的误差密度</small><br>
    <a href="figures_svg/violin_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Raincloud Plot · 雨云图</b><br>
    <img src="figures/raincloud_plot.png" alt="Raincloud Plot 示例图" width="430"><br>
    <small>密度、样本点与摘要统计</small><br>
    <a href="figures_svg/raincloud_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Beeswarm Plot · 蜂群图</b><br>
    <img src="figures/beeswarm_plot.png" alt="Beeswarm Plot 示例图" width="430"><br>
    <small>尽量避免点遮挡的个体观测分布</small><br>
    <a href="figures_svg/beeswarm_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Strip Plot · 条带散点图</b><br>
    <img src="figures/strip_plot.png" alt="Strip Plot 示例图" width="430"><br>
    <small>单个样本或重复实验结果</small><br>
    <a href="figures_svg/strip_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Ridge Plot · 山峦图</b><br>
    <img src="figures/ridge_plot.png" alt="Ridge Plot 示例图" width="430"><br>
    <small>多条件下的误差分布</small><br>
    <a href="figures_svg/ridge_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>KDE Plot · 核密度图</b><br>
    <img src="figures/kde_plot.png" alt="KDE Plot 示例图" width="430"><br>
    <small>连续误差密度比较</small><br>
    <a href="figures_svg/kde_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>ECDF Plot · 经验累积分布图</b><br>
    <img src="figures/ecdf_plot.png" alt="ECDF Plot 示例图" width="430"><br>
    <small>误差低于阈值的比例</small><br>
    <a href="figures_svg/ecdf_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Histogram · 直方图</b><br>
    <img src="figures/histogram.png" alt="Histogram 示例图" width="430"><br>
    <small>误差频次或尾部分布</small><br>
    <a href="figures_svg/histogram.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Q-Q Plot · Q-Q 分位数图</b><br>
    <img src="figures/qq_plot.png" alt="Q-Q Plot 示例图" width="430"><br>
    <small>检验误差分布与参考分布的匹配程度</small><br>
    <a href="figures_svg/qq_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 4. 超参数敏感性与搜索

本类共有 **9** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 4.1 | **Contour Plot**（等高线图） | 两个超参数的联合性能曲面 | 格点较稀疏时不要通过插值暗示已观测的中间结果。 |
| 4.2 | **3D Surface Plot**（三维曲面图） | 二维参数与损失的三维映射 | 三维透视可能掩盖差异，最好配二维图。 |
| 4.3 | **Scatter Plot**（散点图） | 参数取值与性能关联 | 相关并不意味着因果；注意交互与混杂变量。 |
| 4.4 | **Bubble Chart**（气泡图） | 参数、得分与资源量三重编码 | 气泡面积而非直径应与数值对应。 |
| 4.5 | **Parallel Coordinates Plot**（平行坐标图） | 多超参数组合与最终得分 | 参数轴最好按可比范围归一化，突出表现最优的组合。 |
| 4.6 | **Scatter Matrix / Pair Plot**（散点矩阵图） | 参数之间的成对关系 | 散点矩阵很多子图，控制变量数量。 |
| 4.7 | **Response Surface Plot**（响应曲面图） | 拟合的响应曲面与采样点 | 明确区分真实观测点与拟合/插值区域。 |
| 4.8 | **Small Multiples / Facet Grid**（分面图） | 跨数据集参数响应对比 | 每个小图尽可能共享坐标轴范围。 |
| 4.9 | **Ternary Plot**（三元图） | 总和固定的三个权重 | 仅适合三权重非负、相加为固定值的情形。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Contour Plot · 等高线图</b><br>
    <img src="figures/contour_plot.png" alt="Contour Plot 示例图" width="430"><br>
    <small>两个超参数的联合性能曲面</small><br>
    <a href="figures_svg/contour_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>3D Surface Plot · 三维曲面图</b><br>
    <img src="figures/surface_3d.png" alt="3D Surface Plot 示例图" width="430"><br>
    <small>二维参数与损失的三维映射</small><br>
    <a href="figures_svg/surface_3d.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Scatter Plot · 散点图</b><br>
    <img src="figures/hyper_scatter.png" alt="Scatter Plot 示例图" width="430"><br>
    <small>参数取值与性能关联</small><br>
    <a href="figures_svg/hyper_scatter.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Bubble Chart · 气泡图</b><br>
    <img src="figures/hyper_bubble.png" alt="Bubble Chart 示例图" width="430"><br>
    <small>参数、得分与资源量三重编码</small><br>
    <a href="figures_svg/hyper_bubble.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Parallel Coordinates Plot · 平行坐标图</b><br>
    <img src="figures/parallel_coordinates.png" alt="Parallel Coordinates Plot 示例图" width="430"><br>
    <small>多超参数组合与最终得分</small><br>
    <a href="figures_svg/parallel_coordinates.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Scatter Matrix / Pair Plot · 散点矩阵图</b><br>
    <img src="figures/scatter_matrix.png" alt="Scatter Matrix / Pair Plot 示例图" width="430"><br>
    <small>参数之间的成对关系</small><br>
    <a href="figures_svg/scatter_matrix.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Response Surface Plot · 响应曲面图</b><br>
    <img src="figures/response_surface.png" alt="Response Surface Plot 示例图" width="430"><br>
    <small>拟合的响应曲面与采样点</small><br>
    <a href="figures_svg/response_surface.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Small Multiples / Facet Grid · 分面图</b><br>
    <img src="figures/small_multiples.png" alt="Small Multiples / Facet Grid 示例图" width="430"><br>
    <small>跨数据集参数响应对比</small><br>
    <a href="figures_svg/small_multiples.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Ternary Plot · 三元图</b><br>
    <img src="figures/ternary_plot.png" alt="Ternary Plot 示例图" width="430"><br>
    <small>总和固定的三个权重</small><br>
    <a href="figures_svg/ternary_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%"></td>
</tr>
</table>

---

## 5. 计算成本与性能权衡

本类共有 **8** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 5.1 | **Pareto Frontier Plot**（帕累托前沿图） | 性能与成本间的非支配关系（示意） | 前沿需由非支配解计算，不能随意连线。 |
| 5.2 | **Accuracy–Latency Scatter**（精度—延迟散点图） | 精度与推理延迟的取舍 | 需说明硬件、批量大小、精度模式和预热。 |
| 5.3 | **Cost Bubble Plot**（成本气泡图） | 延迟、性能与模型规模 | 保证所有点的气泡尺寸在同一尺度上。 |
| 5.4 | **Roofline Plot**（屋顶线图） | 硬件峰值与计算访存瓶颈 | 理论屋顶依赖硬件测量，并非通用固定数值。 |
| 5.5 | **Stacked Cost Breakdown**（成本组成图） | 多个模块的推理耗时 | 组成项必须可加，口径保持一致。 |
| 5.6 | **Sankey / Flow Diagram**（桑基/流量图） | 计算、数据或样本流向（示意） | 流量宽度原则上应反映数量；此图只是流程风格示意。 |
| 5.7 | **Gantt Chart**（甘特图） | 推理流水线耗时和时序 | 时序叠加表示可并行执行，别误当串行。 |
| 5.8 | **Speedup Plot**（加速比图） | 扩展资源量下的加速比 | 同环境下测量；展示理想线有助于解释瓶颈。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Pareto Frontier Plot · 帕累托前沿图</b><br>
    <img src="figures/pareto_frontier.png" alt="Pareto Frontier Plot 示例图" width="430"><br>
    <small>性能与成本间的非支配关系（示意）</small><br>
    <a href="figures_svg/pareto_frontier.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Accuracy–Latency Scatter · 精度—延迟散点图</b><br>
    <img src="figures/accuracy_latency.png" alt="Accuracy–Latency Scatter 示例图" width="430"><br>
    <small>精度与推理延迟的取舍</small><br>
    <a href="figures_svg/accuracy_latency.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Cost Bubble Plot · 成本气泡图</b><br>
    <img src="figures/efficiency_bubble.png" alt="Cost Bubble Plot 示例图" width="430"><br>
    <small>延迟、性能与模型规模</small><br>
    <a href="figures_svg/efficiency_bubble.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Roofline Plot · 屋顶线图</b><br>
    <img src="figures/roofline_plot.png" alt="Roofline Plot 示例图" width="430"><br>
    <small>硬件峰值与计算访存瓶颈</small><br>
    <a href="figures_svg/roofline_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Stacked Cost Breakdown · 成本组成图</b><br>
    <img src="figures/cost_breakdown.png" alt="Stacked Cost Breakdown 示例图" width="430"><br>
    <small>多个模块的推理耗时</small><br>
    <a href="figures_svg/cost_breakdown.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Sankey / Flow Diagram · 桑基/流量图</b><br>
    <img src="figures/sankey_cost.png" alt="Sankey / Flow Diagram 示例图" width="430"><br>
    <small>计算、数据或样本流向（示意）</small><br>
    <a href="figures_svg/sankey_cost.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Gantt Chart · 甘特图</b><br>
    <img src="figures/gantt_chart.png" alt="Gantt Chart 示例图" width="430"><br>
    <small>推理流水线耗时和时序</small><br>
    <a href="figures_svg/gantt_chart.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Speedup Plot · 加速比图</b><br>
    <img src="figures/speedup_plot.png" alt="Speedup Plot 示例图" width="430"><br>
    <small>扩展资源量下的加速比</small><br>
    <a href="figures_svg/speedup_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 6. 预测质量与诊断

本类共有 **11** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 6.1 | **Prediction vs Ground Truth Scatter**（预测—真实散点图） | 预测值与实际值的一致性 | 加 y=x 参考线；远离对角线的点值得检查。 |
| 6.2 | **Residual Plot**（残差图） | 是否存在系统性残差模式 | 残差应检查异方差、偏差和极端值。 |
| 6.3 | **Bland–Altman Plot**（Bland–Altman 一致性图） | 预测与真实值之间的一致性差异 | 一致性界限不能替代对预测误差阈值的任务判断。 |
| 6.4 | **Fan Chart**（扇形预测图） | 不同置信/预测区间的不确定性扩张 | 概率区间需经过有效的统计校准或评估。 |
| 6.5 | **Quantile Band Plot**（分位数带图） | 多个预测分位数及真值 | 不同覆盖概率的区间不能交叉；标注分位数。 |
| 6.6 | **Error Profile**（误差剖面图） | 预测步长与绝对误差 | 预测步长与单位应统一。 |
| 6.7 | **Taylor Diagram**（泰勒图） | 相关性、标准差与中心化 RMSE | 泰勒图的相关性、标准差和中心化 RMSE 有几何关联。 |
| 6.8 | **Coverage Plot**（覆盖率图） | 名义覆盖率与经验覆盖率 | 明确校准集与独立测试集的划分。 |
| 6.9 | **Reliability Diagram**（可靠性图） | 分类概率校准程度 | 概率校准不等于分类精度，建议配 Brier/ECE。 |
| 6.10 | **ROC / PR Curves**（ROC/PR 曲线） | 分类阈值下的精度—召回权衡 | 类别不平衡时 PR 通常更有信息量。 |
| 6.11 | **Confusion Matrix**（混淆矩阵） | 各分类之间的混淆情况 | 若类别分布不均衡，同时给出归一化矩阵。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Prediction vs Ground Truth Scatter · 预测—真实散点图</b><br>
    <img src="figures/prediction_scatter.png" alt="Prediction vs Ground Truth Scatter 示例图" width="430"><br>
    <small>预测值与实际值的一致性</small><br>
    <a href="figures_svg/prediction_scatter.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Residual Plot · 残差图</b><br>
    <img src="figures/residual_plot.png" alt="Residual Plot 示例图" width="430"><br>
    <small>是否存在系统性残差模式</small><br>
    <a href="figures_svg/residual_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Bland–Altman Plot · Bland–Altman 一致性图</b><br>
    <img src="figures/bland_altman.png" alt="Bland–Altman Plot 示例图" width="430"><br>
    <small>预测与真实值之间的一致性差异</small><br>
    <a href="figures_svg/bland_altman.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Fan Chart · 扇形预测图</b><br>
    <img src="figures/fan_chart.png" alt="Fan Chart 示例图" width="430"><br>
    <small>不同置信/预测区间的不确定性扩张</small><br>
    <a href="figures_svg/fan_chart.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Quantile Band Plot · 分位数带图</b><br>
    <img src="figures/quantile_band.png" alt="Quantile Band Plot 示例图" width="430"><br>
    <small>多个预测分位数及真值</small><br>
    <a href="figures_svg/quantile_band.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Error Profile · 误差剖面图</b><br>
    <img src="figures/error_profile.png" alt="Error Profile 示例图" width="430"><br>
    <small>预测步长与绝对误差</small><br>
    <a href="figures_svg/error_profile.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Taylor Diagram · 泰勒图</b><br>
    <img src="figures/taylor_diagram.png" alt="Taylor Diagram 示例图" width="430"><br>
    <small>相关性、标准差与中心化 RMSE</small><br>
    <a href="figures_svg/taylor_diagram.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Coverage Plot · 覆盖率图</b><br>
    <img src="figures/coverage_plot.png" alt="Coverage Plot 示例图" width="430"><br>
    <small>名义覆盖率与经验覆盖率</small><br>
    <a href="figures_svg/coverage_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Reliability Diagram · 可靠性图</b><br>
    <img src="figures/reliability_diagram.png" alt="Reliability Diagram 示例图" width="430"><br>
    <small>分类概率校准程度</small><br>
    <a href="figures_svg/reliability_diagram.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>ROC / PR Curves · ROC/PR 曲线</b><br>
    <img src="figures/roc_pr_curve.png" alt="ROC / PR Curves 示例图" width="430"><br>
    <small>分类阈值下的精度—召回权衡</small><br>
    <a href="figures_svg/roc_pr_curve.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Confusion Matrix · 混淆矩阵</b><br>
    <img src="figures/confusion_matrix.png" alt="Confusion Matrix 示例图" width="430"><br>
    <small>各分类之间的混淆情况</small><br>
    <a href="figures_svg/confusion_matrix.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%"></td>
</tr>
</table>

---

## 7. 表示学习与可解释性

本类共有 **10** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 7.1 | **t-SNE Projection**（特征 t-SNE 投影图） | 高维表示的二维可视化示意 | 此处是示意点云；实际应用必须对真实特征运行 t-SNE。 |
| 7.2 | **UMAP Projection**（特征 UMAP 投影图） | 嵌入样本的局部和全局结构展示 | 此处是示意点云；实际应用必须对真实特征运行 UMAP。 |
| 7.3 | **PCA Projection**（主成分 PCA 投影图） | 主成分空间的投影展示 | 真实 PCA 应基于拟合的主成分得分。 |
| 7.4 | **SHAP Beeswarm Plot**（SHAP 蜂群图） | 特征归因的方向和分布 | 示意归因图；实际 SHAP 值需由解释算法计算。 |
| 7.5 | **SHAP Waterfall Plot**（SHAP 瀑布图） | 单样本预测的特征贡献 | 示意累加贡献；真实图应由样本归因值构造。 |
| 7.6 | **Network Graph**（网络拓扑图） | 节点和边的拓扑结构 | 图布局改变视觉邻近，不一定改变拓扑。 |
| 7.7 | **Sankey / Alluvial Plot**（状态流转图） | 状态与路径转移 | 示意流带非严格面积守恒，论文用定量真实流量。 |
| 7.8 | **Activation Map**（激活图） | 模型的空间激活强度 | 激活强度高不等于因果重要性高。 |
| 7.9 | **Decision Boundary Plot**（决策边界图） | 二维特征空间中的分类边界 | 二维切片不代表完整高维决策边界。 |
| 7.10 | **CKA Similarity Matrix**（CKA 相似性矩阵） | 模型层间表示相似度 | CKA 值需用真实激活特征计算，并说明层对齐。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>t-SNE Projection · 特征 t-SNE 投影图</b><br>
    <img src="figures/tsne_projection.png" alt="t-SNE Projection 示例图" width="430"><br>
    <small>高维表示的二维可视化示意</small><br>
    <a href="figures_svg/tsne_projection.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>UMAP Projection · 特征 UMAP 投影图</b><br>
    <img src="figures/umap_projection.png" alt="UMAP Projection 示例图" width="430"><br>
    <small>嵌入样本的局部和全局结构展示</small><br>
    <a href="figures_svg/umap_projection.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>PCA Projection · 主成分 PCA 投影图</b><br>
    <img src="figures/pca_projection.png" alt="PCA Projection 示例图" width="430"><br>
    <small>主成分空间的投影展示</small><br>
    <a href="figures_svg/pca_projection.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>SHAP Beeswarm Plot · SHAP 蜂群图</b><br>
    <img src="figures/shap_beeswarm.png" alt="SHAP Beeswarm Plot 示例图" width="430"><br>
    <small>特征归因的方向和分布</small><br>
    <a href="figures_svg/shap_beeswarm.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>SHAP Waterfall Plot · SHAP 瀑布图</b><br>
    <img src="figures/shap_waterfall.png" alt="SHAP Waterfall Plot 示例图" width="430"><br>
    <small>单样本预测的特征贡献</small><br>
    <a href="figures_svg/shap_waterfall.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Network Graph · 网络拓扑图</b><br>
    <img src="figures/network_graph.png" alt="Network Graph 示例图" width="430"><br>
    <small>节点和边的拓扑结构</small><br>
    <a href="figures_svg/network_graph.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Sankey / Alluvial Plot · 状态流转图</b><br>
    <img src="figures/sankey_alluvial.png" alt="Sankey / Alluvial Plot 示例图" width="430"><br>
    <small>状态与路径转移</small><br>
    <a href="figures_svg/sankey_alluvial.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Activation Map · 激活图</b><br>
    <img src="figures/activation_map.png" alt="Activation Map 示例图" width="430"><br>
    <small>模型的空间激活强度</small><br>
    <a href="figures_svg/activation_map.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Decision Boundary Plot · 决策边界图</b><br>
    <img src="figures/decision_boundary.png" alt="Decision Boundary Plot 示例图" width="430"><br>
    <small>二维特征空间中的分类边界</small><br>
    <a href="figures_svg/decision_boundary.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>CKA Similarity Matrix · CKA 相似性矩阵</b><br>
    <img src="figures/cka_similarity.png" alt="CKA Similarity Matrix 示例图" width="430"><br>
    <small>模型层间表示相似度</small><br>
    <a href="figures_svg/cka_similarity.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 8. 时空预测与动态决策

本类共有 **10** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 8.1 | **Spatial Error Map**（空间误差图） | 节点/道路传感器的误差空间分布 | 节点位置应来自真实地理位置或明确标注抽象布局。 |
| 8.2 | **Node-Level Improvement Plot**（节点改善图） | 不同节点的修正收益 | 收益正负号与指标定义要写清楚。 |
| 8.3 | **Temporal Error Profile**（时间/预测步长误差图） | 不同预测 horizon 的误差曲线 | 不同步长应基于相同样本协议统计。 |
| 8.4 | **Spatio-Temporal Surface**（时空三维曲面图） | 时间、节点与误差的联合变化 | 三维面容易遮挡，必要时给二维切片。 |
| 8.5 | **Graph Overlay Plot**（图网络叠加图） | 在空间网络上展示关键决策 | 边、节点大小和颜色分别代表什么需说明。 |
| 8.6 | **Event-Aligned Trajectory**（事件对齐预测轨迹图） | 异常事件前后新旧预测比较 | 标明事件开始时间，排除由未来信息泄露产生的改善。 |
| 8.7 | **Revision Timeline**（修正决策时间轴） | 逐时刻记录节点修正动作 | 有动作与无动作都应编码，读者才能判断密度。 |
| 8.8 | **Gain Distribution Plot**（收益分布图） | 有益与有害修正的分布 | 要同时展示负收益比例，避免只报平均改善。 |
| 8.9 | **Budget–Performance Curve**（预算—性能曲线） | 有限修正预算下的误差变化 | 预算是调用次数、节点比例还是 FLOPs，必须说明。 |
| 8.10 | **State Transition Diagram**（状态转移图） | 策略中的状态和行动路径 | 状态转移边应来自可解释的动作/概率定义。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Spatial Error Map · 空间误差图</b><br>
    <img src="figures/spatial_error_map.png" alt="Spatial Error Map 示例图" width="430"><br>
    <small>节点/道路传感器的误差空间分布</small><br>
    <a href="figures_svg/spatial_error_map.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Node-Level Improvement Plot · 节点改善图</b><br>
    <img src="figures/node_improvement.png" alt="Node-Level Improvement Plot 示例图" width="430"><br>
    <small>不同节点的修正收益</small><br>
    <a href="figures_svg/node_improvement.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Temporal Error Profile · 时间/预测步长误差图</b><br>
    <img src="figures/temporal_error_profile.png" alt="Temporal Error Profile 示例图" width="430"><br>
    <small>不同预测 horizon 的误差曲线</small><br>
    <a href="figures_svg/temporal_error_profile.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Spatio-Temporal Surface · 时空三维曲面图</b><br>
    <img src="figures/spatiotemporal_surface.png" alt="Spatio-Temporal Surface 示例图" width="430"><br>
    <small>时间、节点与误差的联合变化</small><br>
    <a href="figures_svg/spatiotemporal_surface.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Graph Overlay Plot · 图网络叠加图</b><br>
    <img src="figures/graph_overlay.png" alt="Graph Overlay Plot 示例图" width="430"><br>
    <small>在空间网络上展示关键决策</small><br>
    <a href="figures_svg/graph_overlay.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Event-Aligned Trajectory · 事件对齐预测轨迹图</b><br>
    <img src="figures/event_trajectory.png" alt="Event-Aligned Trajectory 示例图" width="430"><br>
    <small>异常事件前后新旧预测比较</small><br>
    <a href="figures_svg/event_trajectory.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Revision Timeline · 修正决策时间轴</b><br>
    <img src="figures/revision_timeline.png" alt="Revision Timeline 示例图" width="430"><br>
    <small>逐时刻记录节点修正动作</small><br>
    <a href="figures_svg/revision_timeline.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Gain Distribution Plot · 收益分布图</b><br>
    <img src="figures/gain_distribution.png" alt="Gain Distribution Plot 示例图" width="430"><br>
    <small>有益与有害修正的分布</small><br>
    <a href="figures_svg/gain_distribution.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Budget–Performance Curve · 预算—性能曲线</b><br>
    <img src="figures/budget_performance.png" alt="Budget–Performance Curve 示例图" width="430"><br>
    <small>有限修正预算下的误差变化</small><br>
    <a href="figures_svg/budget_performance.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>State Transition Diagram · 状态转移图</b><br>
    <img src="figures/state_transition.png" alt="State Transition Diagram 示例图" width="430"><br>
    <small>策略中的状态和行动路径</small><br>
    <a href="figures_svg/state_transition.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 9. 三维与多维参数空间

本类共有 **17** 种展示形式，回答的问题是：多个参数或多个维度如何共同影响结果？

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 9.1 | **3D Scatter with Drop Lines**（三维散点图（带垂线）） | 三个损失权重或三个超参数的搜索空间，颜色表示得分 | 三维位置难以精确读数：加垂线或投影线，标出最优点，并在正文或附录给出数值表。 |
| 9.2 | **3D Bar Chart**（三维柱状图） | 两个离散超参数的网格结果 | 后排柱子会被遮挡；网格大于约 6×6 时改用带数值的热力图。 |
| 9.3 | **3D Surface with Contour Projection**（三维曲面＋底面等高线投影） | 两个连续参数的响应曲面，投影便于读出最优区域 | 曲面来自插值时须标出真实采样点；透视会夸大或缩小差异。 |
| 9.4 | **3D Wireframe Comparison**（三维线框对比图） | 两个模型在同一参数平面上的误差曲面对比 | 最多画两层曲面；曲面交叉时读者难以判断高低。 |
| 9.5 | **3D Waterfall / Stacked Curves**（三维瀑布图（层叠曲线）） | 多个条件（预测步长、层、时间）下的曲线或分布 | 条件多于 6 个时改用山峦图或热力图。 |
| 9.6 | **3D Triangulated Surface**（三角剖分曲面图） | 随机搜索等不规则采样点构成的响应曲面 | 三角面片是线性插值，不代表采样点之间的真实取值。 |
| 9.7 | **Loss Landscape with Trajectory**（损失地形与优化轨迹） | 沿两个方向的损失曲面及训练轨迹 | 方向的选取与归一化方式（如 filter normalization）必须说明；不同模型的地形不可直接比较尺度。 |
| 9.8 | **3D Embedding Scatter**（三维嵌入散点图） | 降维到三维的表示及其类别结构 | 视角会改变簇的可分程度；最好同时给二维投影。 |
| 9.9 | **Stacked Tensor Slices**（张量切片堆叠图） | 时间 × 节点 × 特征的三阶数据或注意力张量 | 切片之间有遮挡，只适合展示 3–5 个切片。 |
| 9.10 | **Contour with Optimization Paths**（等高线＋优化路径图） | 不同优化器或训练策略在损失平面上的路径 | 路径是高维轨迹的投影，投影方法须说明。 |
| 9.11 | **Ternary Contour Plot**（三元等值图） | 和为 1 的三个权重上的连续响应 | 只适用于三者非负且和固定的情形；三个独立权重应使用三维散点图。 |
| 9.12 | **Annotated Grid Heatmap**（带数值标注的网格热力图） | 两个超参数网格搜索的精确数值与最优格 | 颜色映射方向要与“越低越好”一致；标出最优格并说明选择依据（验证集）。 |
| 9.13 | **Bubble Matrix (Balloon Plot)**（气泡矩阵图） | 网格上同时展示均值（面积）与波动（颜色） | 面积与数值成正比；图例须给出面积含义。 |
| 9.14 | **Hexbin Density Plot**（六边形分箱密度图） | 样本量很大时的预测—真值或两变量联合分布 | 计数跨度大时用对数颜色并标明。 |
| 9.15 | **Dual-Axis Bar–Line Chart**（双轴柱线组合图） | 同一参数下的精度（柱）与开销（线） | 两个纵轴的范围选择会影响观感；两轴都要标注单位，避免暗示相关。 |
| 9.16 | **Hyperparameter Slice Plot**（超参数切片图） | 搜索试验在每个超参数上的边际表现 | 边际图掩盖参数之间的交互；颜色若表示试验顺序须说明。 |
| 9.17 | **Hyperparameter Importance Plot**（超参数重要性图） | 各超参数对结果方差的贡献（如 fANOVA） | 重要性依赖搜索范围与估计方法，不代表因果。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；绘制代码见 `generate_gallery_ext.py` 中的同名函数。

<table>
<tr>
  <td width="50%" valign="top">
    <b>3D Scatter with Drop Lines · 三维散点图（带垂线）</b><br>
    <img src="figures/scatter3d_stem.png" alt="3D Scatter with Drop Lines 示例图" width="430"><br>
    <small>三个损失权重或三个超参数的搜索空间，颜色表示得分（代码：<code>d_scatter3d_stem</code>）</small><br>
    <a href="figures_svg/scatter3d_stem.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>3D Bar Chart · 三维柱状图</b><br>
    <img src="figures/bar3d_grid.png" alt="3D Bar Chart 示例图" width="430"><br>
    <small>两个离散超参数的网格结果（代码：<code>d_bar3d</code>）</small><br>
    <a href="figures_svg/bar3d_grid.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>3D Surface with Contour Projection · 三维曲面＋底面等高线投影</b><br>
    <img src="figures/surface_contour_projection.png" alt="3D Surface with Contour Projection 示例图" width="430"><br>
    <small>两个连续参数的响应曲面，投影便于读出最优区域（代码：<code>d_surface_contour</code>）</small><br>
    <a href="figures_svg/surface_contour_projection.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>3D Wireframe Comparison · 三维线框对比图</b><br>
    <img src="figures/wireframe_3d.png" alt="3D Wireframe Comparison 示例图" width="430"><br>
    <small>两个模型在同一参数平面上的误差曲面对比（代码：<code>d_wireframe</code>）</small><br>
    <a href="figures_svg/wireframe_3d.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>3D Waterfall / Stacked Curves · 三维瀑布图（层叠曲线）</b><br>
    <img src="figures/waterfall_3d.png" alt="3D Waterfall / Stacked Curves 示例图" width="430"><br>
    <small>多个条件（预测步长、层、时间）下的曲线或分布（代码：<code>d_waterfall3d</code>）</small><br>
    <a href="figures_svg/waterfall_3d.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>3D Triangulated Surface · 三角剖分曲面图</b><br>
    <img src="figures/trisurf_3d.png" alt="3D Triangulated Surface 示例图" width="430"><br>
    <small>随机搜索等不规则采样点构成的响应曲面（代码：<code>d_trisurf</code>）</small><br>
    <a href="figures_svg/trisurf_3d.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Loss Landscape with Trajectory · 损失地形与优化轨迹</b><br>
    <img src="figures/loss_landscape_3d.png" alt="Loss Landscape with Trajectory 示例图" width="430"><br>
    <small>沿两个方向的损失曲面及训练轨迹（代码：<code>d_loss_landscape</code>）</small><br>
    <a href="figures_svg/loss_landscape_3d.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>3D Embedding Scatter · 三维嵌入散点图</b><br>
    <img src="figures/scatter3d_embedding.png" alt="3D Embedding Scatter 示例图" width="430"><br>
    <small>降维到三维的表示及其类别结构（代码：<code>d_scatter3d_embedding</code>）</small><br>
    <a href="figures_svg/scatter3d_embedding.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Stacked Tensor Slices · 张量切片堆叠图</b><br>
    <img src="figures/tensor_slices.png" alt="Stacked Tensor Slices 示例图" width="430"><br>
    <small>时间 × 节点 × 特征的三阶数据或注意力张量（代码：<code>d_tensor_slices</code>）</small><br>
    <a href="figures_svg/tensor_slices.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Contour with Optimization Paths · 等高线＋优化路径图</b><br>
    <img src="figures/contour_trajectory.png" alt="Contour with Optimization Paths 示例图" width="430"><br>
    <small>不同优化器或训练策略在损失平面上的路径（代码：<code>d_contour_trajectory</code>）</small><br>
    <a href="figures_svg/contour_trajectory.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Ternary Contour Plot · 三元等值图</b><br>
    <img src="figures/ternary_heatmap.png" alt="Ternary Contour Plot 示例图" width="430"><br>
    <small>和为 1 的三个权重上的连续响应（代码：<code>d_ternary_heatmap</code>）</small><br>
    <a href="figures_svg/ternary_heatmap.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Annotated Grid Heatmap · 带数值标注的网格热力图</b><br>
    <img src="figures/annotated_heatmap.png" alt="Annotated Grid Heatmap 示例图" width="430"><br>
    <small>两个超参数网格搜索的精确数值与最优格（代码：<code>d_annotated_heatmap</code>）</small><br>
    <a href="figures_svg/annotated_heatmap.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Bubble Matrix (Balloon Plot) · 气泡矩阵图</b><br>
    <img src="figures/bubble_matrix.png" alt="Bubble Matrix (Balloon Plot) 示例图" width="430"><br>
    <small>网格上同时展示均值（面积）与波动（颜色）（代码：<code>d_bubble_matrix</code>）</small><br>
    <a href="figures_svg/bubble_matrix.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Hexbin Density Plot · 六边形分箱密度图</b><br>
    <img src="figures/hexbin_density.png" alt="Hexbin Density Plot 示例图" width="430"><br>
    <small>样本量很大时的预测—真值或两变量联合分布（代码：<code>d_hexbin</code>）</small><br>
    <a href="figures_svg/hexbin_density.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Dual-Axis Bar–Line Chart · 双轴柱线组合图</b><br>
    <img src="figures/dual_axis_bar_line.png" alt="Dual-Axis Bar–Line Chart 示例图" width="430"><br>
    <small>同一参数下的精度（柱）与开销（线）（代码：<code>d_dual_axis</code>）</small><br>
    <a href="figures_svg/dual_axis_bar_line.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Hyperparameter Slice Plot · 超参数切片图</b><br>
    <img src="figures/slice_plot.png" alt="Hyperparameter Slice Plot 示例图" width="430"><br>
    <small>搜索试验在每个超参数上的边际表现（代码：<code>d_slice_plot</code>）</small><br>
    <a href="figures_svg/slice_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Hyperparameter Importance Plot · 超参数重要性图</b><br>
    <img src="figures/hyperparameter_importance.png" alt="Hyperparameter Importance Plot 示例图" width="430"><br>
    <small>各超参数对结果方差的贡献（如 fANOVA）（代码：<code>d_importance</code>）</small><br>
    <a href="figures_svg/hyperparameter_importance.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%"></td>
</tr>
</table>

---

## 10. 组合图与多面板

本类共有 **13** 种展示形式，回答的问题是：如何在一张图里同时给出均值、波动、显著性或局部细节？

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 10.1 | **Grouped Bar Chart with Error Bars**（分组柱状图（误差线＋填充纹理）） | 多个数据集上多个模型的均值与波动 | 误差线含义（SD、SE、CI）与重复次数写入图注；纹理保证灰度打印可分辨。 |
| 10.2 | **Bar Chart with Significance Brackets**（带显著性标记的柱状图） | 成对比较的统计检验结果 | 星号必须来自实际检验，并说明检验方法、样本单位与多重比较校正。 |
| 10.3 | **Broken-Axis Bar Chart**（断轴柱状图） | 个别方法数值远大于其他方法时保留细节 | 断轴处必须有明显标记，并标出被截断柱的数值。 |
| 10.4 | **Line Plot with Inset Zoom**（带局部放大的折线图） | 收敛后期或接近的曲线的细微差别 | 放大框与主图的对应关系要画出；放大区域不能只挑有利的一段。 |
| 10.5 | **Diverging Bar Chart**（发散条形图） | 消融或变体相对完整模型的增减百分比 | 基准与正负方向的含义写清楚（正值表示变差还是变好）。 |
| 10.6 | **100% Stacked Bar Chart**（百分比堆叠条形图） | 误差、耗时或样本在各类别中的占比 | 只比较占比，不能比较绝对量；必要时另给总量。 |
| 10.7 | **Heatmap with Marginal Bars**（带边际统计的热力图） | 二维结果及其按行、按列的平均 | 边际统计的聚合方式（均值或求和）需说明。 |
| 10.8 | **Clustered Heatmap with Dendrogram**（聚类热力图（带树状图）） | 按相似性重排后的特征、节点或模型矩阵 | 聚类距离与连接方式影响排序；标准化方式需说明。 |
| 10.9 | **Joint Plot with Marginal Histograms**（联合分布图（带边际直方图）） | 两个指标的联合分布与各自的分布 | 两组样本量不同时，边际直方图用密度而非计数。 |
| 10.10 | **Rank Heatmap**（排名热力图） | 多个模型在多个数据集或指标上的名次 | 排名隐藏差距大小，应与数值表配合使用。 |
| 10.11 | **Nightingale Rose Chart**（南丁格尔玫瑰图） | 少量类别的指标概览（展示性强） | 半径编码使面积随数值平方增长，容易夸大差异；正式比较用柱状图。 |
| 10.12 | **Paired Comparison Plot**（配对比较图） | 同一数据集或种子上两种方法的成对结果 | 只连接真正配对的结果；同时报告变好与变差的数量。 |
| 10.13 | **Back-to-Back Bar Chart**（背靠背条形图） | 两个场景在同一分组下的对照 | 两侧使用相同刻度，否则长度不可比较。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；绘制代码见 `generate_gallery_ext.py` 中的同名函数。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Grouped Bar Chart with Error Bars · 分组柱状图（误差线＋填充纹理）</b><br>
    <img src="figures/grouped_bar_error.png" alt="Grouped Bar Chart with Error Bars 示例图" width="430"><br>
    <small>多个数据集上多个模型的均值与波动（代码：<code>d_grouped_bar</code>）</small><br>
    <a href="figures_svg/grouped_bar_error.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Bar Chart with Significance Brackets · 带显著性标记的柱状图</b><br>
    <img src="figures/bar_significance.png" alt="Bar Chart with Significance Brackets 示例图" width="430"><br>
    <small>成对比较的统计检验结果（代码：<code>d_bar_significance</code>）</small><br>
    <a href="figures_svg/bar_significance.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Broken-Axis Bar Chart · 断轴柱状图</b><br>
    <img src="figures/broken_axis_bar.png" alt="Broken-Axis Bar Chart 示例图" width="430"><br>
    <small>个别方法数值远大于其他方法时保留细节（代码：<code>d_broken_axis</code>）</small><br>
    <a href="figures_svg/broken_axis_bar.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Line Plot with Inset Zoom · 带局部放大的折线图</b><br>
    <img src="figures/inset_zoom.png" alt="Line Plot with Inset Zoom 示例图" width="430"><br>
    <small>收敛后期或接近的曲线的细微差别（代码：<code>d_inset_zoom</code>）</small><br>
    <a href="figures_svg/inset_zoom.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Diverging Bar Chart · 发散条形图</b><br>
    <img src="figures/diverging_bar.png" alt="Diverging Bar Chart 示例图" width="430"><br>
    <small>消融或变体相对完整模型的增减百分比（代码：<code>d_diverging_bar</code>）</small><br>
    <a href="figures_svg/diverging_bar.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>100% Stacked Bar Chart · 百分比堆叠条形图</b><br>
    <img src="figures/stacked_100_bar.png" alt="100% Stacked Bar Chart 示例图" width="430"><br>
    <small>误差、耗时或样本在各类别中的占比（代码：<code>d_stacked_100</code>）</small><br>
    <a href="figures_svg/stacked_100_bar.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Heatmap with Marginal Bars · 带边际统计的热力图</b><br>
    <img src="figures/heatmap_marginal.png" alt="Heatmap with Marginal Bars 示例图" width="430"><br>
    <small>二维结果及其按行、按列的平均（代码：<code>d_heatmap_marginal</code>）</small><br>
    <a href="figures_svg/heatmap_marginal.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Clustered Heatmap with Dendrogram · 聚类热力图（带树状图）</b><br>
    <img src="figures/clustered_heatmap.png" alt="Clustered Heatmap with Dendrogram 示例图" width="430"><br>
    <small>按相似性重排后的特征、节点或模型矩阵（代码：<code>d_clustered_heatmap</code>）</small><br>
    <a href="figures_svg/clustered_heatmap.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Joint Plot with Marginal Histograms · 联合分布图（带边际直方图）</b><br>
    <img src="figures/joint_plot.png" alt="Joint Plot with Marginal Histograms 示例图" width="430"><br>
    <small>两个指标的联合分布与各自的分布（代码：<code>d_joint_plot</code>）</small><br>
    <a href="figures_svg/joint_plot.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Rank Heatmap · 排名热力图</b><br>
    <img src="figures/rank_heatmap.png" alt="Rank Heatmap 示例图" width="430"><br>
    <small>多个模型在多个数据集或指标上的名次（代码：<code>d_rank_heatmap</code>）</small><br>
    <a href="figures_svg/rank_heatmap.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Nightingale Rose Chart · 南丁格尔玫瑰图</b><br>
    <img src="figures/rose_chart.png" alt="Nightingale Rose Chart 示例图" width="430"><br>
    <small>少量类别的指标概览（展示性强）（代码：<code>d_rose</code>）</small><br>
    <a href="figures_svg/rose_chart.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Paired Comparison Plot · 配对比较图</b><br>
    <img src="figures/paired_plot.png" alt="Paired Comparison Plot 示例图" width="430"><br>
    <small>同一数据集或种子上两种方法的成对结果（代码：<code>d_paired_plot</code>）</small><br>
    <a href="figures_svg/paired_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Back-to-Back Bar Chart · 背靠背条形图</b><br>
    <img src="figures/pyramid_bar.png" alt="Back-to-Back Bar Chart 示例图" width="430"><br>
    <small>两个场景在同一分组下的对照（代码：<code>d_pyramid</code>）</small><br>
    <a href="figures_svg/pyramid_bar.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%"></td>
</tr>
</table>

---

## 11. 结构、关系与注意力

本类共有 **8** 种展示形式，回答的问题是：模型学到了怎样的节点关系、注意力或组成结构？

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 11.1 | **Adjacency Matrix Comparison**（邻接矩阵对比图） | 预定义图与学习到的图结构 | 节点排序决定视觉上的块结构；两图用同一排序与同一色标。 |
| 11.2 | **Attention Map**（注意力权重图） | 查询（预测步）对键（输入步或节点）的注意力 | 注意力权重不等同于重要性或因果；说明取自哪一层、哪个头或如何平均。 |
| 11.3 | **Arc Diagram**（弧线图） | 一维排列的节点之间的连接强度 | 节点顺序影响可读性；边多时只画权重最高的部分并说明阈值。 |
| 11.4 | **Chord Diagram**（弦图） | 区域或类别之间的相互流量或依赖 | 弦宽应与数量成正比；类别多于约 10 个时难以阅读。 |
| 11.5 | **Triangular Correlation Heatmap**（三角相关性热力图） | 变量或节点之间的成对相关 | 使用以 0 为中心的发散色标；说明相关系数类型。 |
| 11.6 | **UpSet Plot**（UpSet 集合交集图） | 多个模型预测正确或失败样本的重叠 | 集合多于 3 个时比韦恩图清晰；交集按规模排序。 |
| 11.7 | **Treemap**（矩形树图） | 参数量、耗时或数据量的层级组成 | 面积与数值成正比；小块标签容易看不清。 |
| 11.8 | **Graph Partition Map**（图分区示意图） | 节点的空间分区、聚类或社区划分 | 位置应来自真实坐标或明确标注为抽象布局；凸包只为便于辨认。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；绘制代码见 `generate_gallery_ext.py` 中的同名函数。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Adjacency Matrix Comparison · 邻接矩阵对比图</b><br>
    <img src="figures/adjacency_compare.png" alt="Adjacency Matrix Comparison 示例图" width="430"><br>
    <small>预定义图与学习到的图结构（代码：<code>d_adjacency_compare</code>）</small><br>
    <a href="figures_svg/adjacency_compare.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Attention Map · 注意力权重图</b><br>
    <img src="figures/attention_map.png" alt="Attention Map 示例图" width="430"><br>
    <small>查询（预测步）对键（输入步或节点）的注意力（代码：<code>d_attention_map</code>）</small><br>
    <a href="figures_svg/attention_map.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Arc Diagram · 弧线图</b><br>
    <img src="figures/arc_diagram.png" alt="Arc Diagram 示例图" width="430"><br>
    <small>一维排列的节点之间的连接强度（代码：<code>d_arc_diagram</code>）</small><br>
    <a href="figures_svg/arc_diagram.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Chord Diagram · 弦图</b><br>
    <img src="figures/chord_diagram.png" alt="Chord Diagram 示例图" width="430"><br>
    <small>区域或类别之间的相互流量或依赖（代码：<code>d_chord</code>）</small><br>
    <a href="figures_svg/chord_diagram.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Triangular Correlation Heatmap · 三角相关性热力图</b><br>
    <img src="figures/correlation_triangle.png" alt="Triangular Correlation Heatmap 示例图" width="430"><br>
    <small>变量或节点之间的成对相关（代码：<code>d_corr_triangle</code>）</small><br>
    <a href="figures_svg/correlation_triangle.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>UpSet Plot · UpSet 集合交集图</b><br>
    <img src="figures/upset_plot.png" alt="UpSet Plot 示例图" width="430"><br>
    <small>多个模型预测正确或失败样本的重叠（代码：<code>d_upset</code>）</small><br>
    <a href="figures_svg/upset_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Treemap · 矩形树图</b><br>
    <img src="figures/treemap.png" alt="Treemap 示例图" width="430"><br>
    <small>参数量、耗时或数据量的层级组成（代码：<code>d_treemap</code>）</small><br>
    <a href="figures_svg/treemap.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Graph Partition Map · 图分区示意图</b><br>
    <img src="figures/partition_map.png" alt="Graph Partition Map 示例图" width="430"><br>
    <small>节点的空间分区、聚类或社区划分（代码：<code>d_partition_map</code>）</small><br>
    <a href="figures_svg/partition_map.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

## 12. 时间序列与时空个例

本类共有 **10** 种展示形式，回答的问题是：具体的时间段、节点或缺失模式下模型表现如何？

| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |
|---:|---|---|---|
| 12.1 | **Forecast Case Study**（预测个例图） | 历史、真值与多个模型预测的对比 | 个例的选取规则须事先确定并写明，不能挑选对本文有利的窗口。 |
| 12.2 | **Missingness Pattern Map**（缺失模式图） | 传感器 × 时间的观测可用性（随机缺失、块缺失、接入与失效） | 区分自然缺失与人为掩蔽；标注时间范围。 |
| 12.3 | **Hour × Weekday Heatmap**（小时 × 星期热力图） | 误差或流量的日内与周内模式 | 每格的样本量可能不同，必要时标注。 |
| 12.4 | **Polar Clock Plot**（极坐标时钟图） | 24 小时周期内的指标变化 | 极坐标下面积会夸大外圈差异；精确比较仍用折线图。 |
| 12.5 | **Periodogram**（周期图（功率谱）） | 序列中的主要周期成分 | 说明采样间隔、是否去趋势以及坐标是否取对数。 |
| 12.6 | **Autocorrelation Plot**（自相关图） | 序列或残差在不同滞后下的相关性 | 置信带基于独立性假设；残差的自相关提示模型未捕捉的结构。 |
| 12.7 | **Streamgraph**（河流图） | 多个类别的量随时间的此消彼长 | 基线不固定，难以读取绝对数值；用于展示总体形态。 |
| 12.8 | **Time–Space Diagram**（时空图（走廊速度图）） | 沿道路位置与时间的速度或拥堵传播 | 位置轴应为真实里程或明确的传感器顺序。 |
| 12.9 | **Stacked Shared-Axis Panels**（共享时间轴的多层面板图） | 输入、误差与模型内部量在同一时间轴上的对应 | 各面板纵轴独立时须分别标注；事件区间在所有面板中对齐。 |
| 12.10 | **Robustness Curve**（鲁棒性曲线） | 缺失率、噪声强度或分布偏移程度增大时的性能变化 | 各扰动水平使用相同的评估样本；阴影含义须说明。 |

### 对应图形示例

> 以下图片均为模拟数据绘制，仅用于理解图形形式；绘制代码见 `generate_gallery_ext.py` 中的同名函数。

<table>
<tr>
  <td width="50%" valign="top">
    <b>Forecast Case Study · 预测个例图</b><br>
    <img src="figures/forecast_case.png" alt="Forecast Case Study 示例图" width="430"><br>
    <small>历史、真值与多个模型预测的对比（代码：<code>d_forecast_case</code>）</small><br>
    <a href="figures_svg/forecast_case.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Missingness Pattern Map · 缺失模式图</b><br>
    <img src="figures/missing_pattern.png" alt="Missingness Pattern Map 示例图" width="430"><br>
    <small>传感器 × 时间的观测可用性（随机缺失、块缺失、接入与失效）（代码：<code>d_missing_pattern</code>）</small><br>
    <a href="figures_svg/missing_pattern.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Hour × Weekday Heatmap · 小时 × 星期热力图</b><br>
    <img src="figures/hour_weekday_heatmap.png" alt="Hour × Weekday Heatmap 示例图" width="430"><br>
    <small>误差或流量的日内与周内模式（代码：<code>d_hour_weekday</code>）</small><br>
    <a href="figures_svg/hour_weekday_heatmap.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Polar Clock Plot · 极坐标时钟图</b><br>
    <img src="figures/polar_clock.png" alt="Polar Clock Plot 示例图" width="430"><br>
    <small>24 小时周期内的指标变化（代码：<code>d_polar_clock</code>）</small><br>
    <a href="figures_svg/polar_clock.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Periodogram · 周期图（功率谱）</b><br>
    <img src="figures/periodogram.png" alt="Periodogram 示例图" width="430"><br>
    <small>序列中的主要周期成分（代码：<code>d_periodogram</code>）</small><br>
    <a href="figures_svg/periodogram.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Autocorrelation Plot · 自相关图</b><br>
    <img src="figures/acf_plot.png" alt="Autocorrelation Plot 示例图" width="430"><br>
    <small>序列或残差在不同滞后下的相关性（代码：<code>d_acf</code>）</small><br>
    <a href="figures_svg/acf_plot.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Streamgraph · 河流图</b><br>
    <img src="figures/streamgraph.png" alt="Streamgraph 示例图" width="430"><br>
    <small>多个类别的量随时间的此消彼长（代码：<code>d_streamgraph</code>）</small><br>
    <a href="figures_svg/streamgraph.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Time–Space Diagram · 时空图（走廊速度图）</b><br>
    <img src="figures/time_space_diagram.png" alt="Time–Space Diagram 示例图" width="430"><br>
    <small>沿道路位置与时间的速度或拥堵传播（代码：<code>d_time_space</code>）</small><br>
    <a href="figures_svg/time_space_diagram.svg">下载 SVG 矢量版</a>
  </td>
</tr>
<tr>
  <td width="50%" valign="top">
    <b>Stacked Shared-Axis Panels · 共享时间轴的多层面板图</b><br>
    <img src="figures/stacked_panels.png" alt="Stacked Shared-Axis Panels 示例图" width="430"><br>
    <small>输入、误差与模型内部量在同一时间轴上的对应（代码：<code>d_stacked_panels</code>）</small><br>
    <a href="figures_svg/stacked_panels.svg">下载 SVG 矢量版</a>
  </td>
  <td width="50%" valign="top">
    <b>Robustness Curve · 鲁棒性曲线</b><br>
    <img src="figures/robustness_curve.png" alt="Robustness Curve 示例图" width="430"><br>
    <small>缺失率、噪声强度或分布偏移程度增大时的性能变化（代码：<code>d_robustness_curve</code>）</small><br>
    <a href="figures_svg/robustness_curve.svg">下载 SVG 矢量版</a>
  </td>
</tr>
</table>

---

---

## 13. 论文实验图快速选择表

| 论文实验内容 | 主要推荐 | 适合补充 | 你要证明的结论 |
|---|---|---|---|
| Main Results（总体性能） | Table、Dot、Dumbbell | Forest、CD | 相对基线的优势及跨数据集普遍性 |
| Ablation Study（消融） | Table、Waterfall | Lollipop、Forest | 模块贡献和叠加效果 |
| Hyperparameter Sensitivity（参数敏感性） | Contour、Small Multiples、Annotated Heatmap | Parallel Coordinates、Ternary、3D Scatter with Drop Lines、3D Bar、Dual-Axis Bar–Line | 超参数容忍度及交互影响 |
| 三个损失权重 / 三个超参数 | 3D Scatter with Drop Lines | Ternary Contour（和为 1 时）、Slice Plot、数值表 | 最优区域在哪里、对取值是否敏感 |
| Case Study（个例） | Forecast Case Study、Stacked Panels | Missingness Pattern、Time–Space Diagram | 模型在具体时段与缺失模式下的行为 |
| Learned Structure（学到的结构） | Adjacency Comparison、Attention Map | Arc / Chord Diagram、Partition Map | 学到的关系是否合理、与先验图有何不同 |
| Convergence（收敛） | Learning Curve | Ribbon、Spaghetti | 收敛速度与多种子稳定性 |
| Robustness（鲁棒性） | ECDF、Raincloud | Ridge、Box、Forest | 尾部风险、波动性和异常场景表现 |
| Efficiency（效率） | Pareto Frontier | Cost Breakdown、Roofline | 性能改进与时间/FLOPs 的代价 |
| Forecasting（预测） | Error Profile、Residual | Fan、Taylor、Coverage | 不同步长的误差模式和不确定性 |
| Interpretability（可解释性） | SHAP、Activation、Network | CKA、Representation | 模型依赖哪些信息与特征 |
| Dynamic Revision（动态修正） | Revision Timeline、Gain Distribution | Spatial Error、Budget–Performance | 修正发生在何时何处、何时值得修正 |

### 时空预测 / Forecast Revision / 强化学习实验的推荐组合

如果在论文中只安排有限数量的分析图，可以优先考虑这组**互补而不重复**的证据：

1. **Dumbbell + Forest**：不同数据集或不同预测任务的改善幅度与不确定性。
2. **ECDF + Gain Distribution**：样本级收益是否普遍，以及有害修正（negative gain）比例。
3. **Horizon Error Profile**：短期、中期、长期预测中谁受益最多。
4. **Budget–Performance + Pareto**：修正预算提升是否值得额外推理时间。
5. **Spatial Error Map + Revision Timeline**：在哪些空间节点、哪个时刻触发修正。
6. **Contour + Ribbon**：超参数敏感性与不同随机种子训练稳定性。

注意：这些只是图形表达建议，不能替代合理的实验设计、真实指标计算以及独立测试集评估。

## 14. 科研作图原则与常见误区

**一、图形与指标语义匹配。** 例如 MAE 越低越好，Accuracy 越高越好；增益符号需定义清楚，如 $\Delta\mathrm{MAE}=\mathrm{MAE}_{base}-\mathrm{MAE}_{proposed}$，正值代表误差减少。

**二、重复实验与统计区间。** 模型种子数量、重复训练与采样单位不同。若 5 个随机种子对应上万预测窗口，不能简单把所有窗口当作独立随机种子去推断模型间显著性。时序数据相关性和跨节点相关性也应纳入统计处理。

**三、避免视觉夸大。** 不要仅通过截断坐标轴、误差棒缺失、过度平滑或 3D 透视制造优势。若变量只有少量离散观测，不宜画成连续、无缝的响应曲面。

**四、预算与效率必须可复现。** 说明 GPU/CPU、批量大小、精度设置、是否预热、是否包含数据预处理或模型加载，才能解读 latency、FLOPs 或速度对比。

**五、统计图需要明确统计量。** 森林图写明效应量和 CI 计算方式；CD 图应对应具体显著性检验；小提琴/雨云图注明样本单位，避免多重相关样本误导。

**六、不要将示意图当真实解释结果。** 实际 t-SNE、UMAP、PCA、SHAP、CKA 必须使用模型输出或中间特征计算。论文中所有合成示意均应明确注明，不可冒充真实实验结果。

**七、统一排版。** 优先白底、细网格、统一线宽与字体，检查单栏缩小后的标签可读性。使用适合黑白打印与色觉差异读者的配色，并区分线型或标记以避免仅依赖颜色。

### 如何使用本图鉴和源码

解压后的目录结构：

```text
deep_learning_visualization_atlas/
├── README.md                   # 本图鉴，123 种图形及用途说明
├── figures/                   # 123 张 PNG 示例图
├── figures_svg/               # 123 张 SVG 矢量示例图
├── generate_gallery.py        # 重新绘制第 1–8 类的 75 张示例图（需要 seaborn）
├── generate_gallery_ext.py    # 重新绘制第 9–12 类的 48 张示例图（只需 numpy、matplotlib、scipy）
└── build_markdown.py          # 重新生成图鉴文档
```

- 推荐通过 **VS Code / Typora / Obsidian / GitHub Markdown Preview** 阅读 `README.md`。
- 复制科研论文中的图形风格时，优先使用 `figures_svg/` 下的矢量文件；投稿时应从真实实验数据重新生成图形，而不是修改示意数值。
- 需要重新生成示例图时，可在 Python 环境中安装：`numpy matplotlib scipy seaborn networkx pandas`，随后运行 `python generate_gallery.py` 与 `python generate_gallery_ext.py`；两个脚本互不覆盖对方的图。每种图的绘制代码是对应脚本中的一个函数（第 9–12 类为 `d_<文件名>` 或文末 `EXT_CATEGORIES` 表中登记的函数），可直接复制改用真实数据。
- 为确保图片能够显示，**不要将 README.md 和 figures/ 分开保存**；整个文件夹移动、压缩或上传时应保持相对路径不变。

## 15. 补充依据与外部资源

2026-11-16 的补充按两类依据确定图型（记录的是检索到的条目，不是系统综述）：

- **图库与工具**：matplotlib 官方图库的 mplot3d 部分（`plot_surface`、`plot_wireframe`、`plot_trisurf`、`bar3d`、`scatter`、`contour` 投影等，<https://matplotlib.org/stable/gallery/mplot3d/index.html>）；The Python Graph Gallery（按图型分类并附代码，<https://python-graph-gallery.com/>，其三维部分提醒三维图多数情况下不如二维图易读）；SciencePlots（论文用 matplotlib 样式，<https://github.com/garrettj403/SciencePlots>）；损失地形可视化的做法见 Li 等人的 *Visualizing the Loss Landscape of Neural Nets*（<https://arxiv.org/abs/1712.09913>）。
- **论文中的图**：查看了 15 篇时空预测与时间序列论文 arXiv 源码中的图注（DCRNN、Graph WaveNet、IGNNK、GRIN、STID、TimesNet、PatchTST、iTransformer、GinAR、PatchSTG、MegaCRN、ST-SSL、STAEformer、TESTAM 等；只看了图注，未通读全文）。出现较多而原图鉴没有的样式：学习到的邻接矩阵、注意力图、预测个例、传感器或分区的空间分布、消融的分组柱状图、超参数的柱状/折线/三维柱状图、嵌入可视化、缺失率敏感性曲线、计算开销对比。
- 用户提供的参考图（三个损失权重的三维散点图，颜色表示得分）对应 9.1。

三维图的使用建议：只有当第三个维度确实是独立变量、且二维图（带数值的热力图、分面折线图）无法表达时才使用；使用时加垂线或投影、固定并说明视角、同时在正文或附录给出数值表。

*图鉴版本：2026-11-16（第 1–8 类为 2026-10-12 版，未改动）。所有示例均为教学/论文排版演示用途。*
