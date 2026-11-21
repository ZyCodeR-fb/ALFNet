#!/usr/bin/env python3
"""Build the Markdown atlas with one illustrative figure for every chart type."""
from pathlib import Path
from html import escape
from generate_gallery import CATEGORIES
from generate_gallery_ext import EXT_CATEGORIES
ROOT=Path(__file__).resolve().parent

CN=[
'点图','棒棒糖图','哑铃图','斜率图','森林图','瀑布图','雷达图','临界差异图（CD 图）','排名变化图',
'学习曲线','带状趋势图','面积图','堆叠面积图','阶梯图','多运行轨迹图','数据规模学习曲线','事件对齐图',
'箱线图','小提琴图','雨云图','蜂群图','条带散点图','山峦图','核密度图','经验累积分布图','直方图','Q-Q 分位数图',
'等高线图','三维曲面图','散点图','气泡图','平行坐标图','散点矩阵图','响应曲面图','分面图','三元图',
'帕累托前沿图','精度—延迟散点图','成本气泡图','屋顶线图','成本组成图','桑基/流量图','甘特图','加速比图',
'预测—真实散点图','残差图','Bland–Altman 一致性图','扇形预测图','分位数带图','误差剖面图','泰勒图','覆盖率图','可靠性图','ROC/PR 曲线','混淆矩阵',
'特征 t-SNE 投影图','特征 UMAP 投影图','主成分 PCA 投影图','SHAP 蜂群图','SHAP 瀑布图','网络拓扑图','状态流转图','激活图','决策边界图','CKA 相似性矩阵',
'空间误差图','节点改善图','时间/预测步长误差图','时空三维曲面图','图网络叠加图','事件对齐预测轨迹图','修正决策时间轴','收益分布图','预算—性能曲线','状态转移图'
]
CAUTION=[
'指标方向要一致；低值更好时不要反向排序。','若从 0 截断轴，应说明坐标含义。','仅对同一测试条件下的成对结果连接。','只有两个条件时表达最清晰；避免标签相互遮挡。','明确效应量定义、区间计算方法和样本独立性。','只有顺序添加模块时才能解释为累计增量。','多个指标必须归一化，且“越大越好”语义一致。','CD 连线必须来自合法统计检验；示意图不代表显著性。','排名依赖候选集合；注意同名次与并列情况。',
'区分训练与验证曲线，并标注平滑策略。','阴影必须说明是 SD、SE 还是 CI。','面积图适合累计量，非零基线会改变视觉含义。','只有可加和分量才应使用堆叠图。','使用 post/pre 标签明确阶梯跳变时刻。','多种子轨迹不要选择性只显示最好的运行。','不同规模下保持其他训练预算/条件可比。','清楚标注事件时刻和时间对齐方式。',
'五数概括之外最好报告样本规模。','小样本 KDE 容易失真；可叠加原始点。','密度、散点与箱线摘要的量纲应一致。','点数过多时可能拥挤，必要时分层抽样。','抖动只能用于显示点，不可改变真实数据值。','多个山脊共享同一横轴尺度，谨慎比较曲线高度。','带宽选择直接影响峰值数量和视觉形状。','误差越小越好时，曲线越靠左上通常越优。','统一直方图区间，避免 bin 不同导致误读。','Q-Q 图不能单凭接近对角线证明模型充分校准。',
'格点较稀疏时不要通过插值暗示已观测的中间结果。','三维透视可能掩盖差异，最好配二维图。','相关并不意味着因果；注意交互与混杂变量。','气泡面积而非直径应与数值对应。','参数轴最好按可比范围归一化，突出表现最优的组合。','散点矩阵很多子图，控制变量数量。','明确区分真实观测点与拟合/插值区域。','每个小图尽可能共享坐标轴范围。','仅适合三权重非负、相加为固定值的情形。',
'前沿需由非支配解计算，不能随意连线。','需说明硬件、批量大小、精度模式和预热。','保证所有点的气泡尺寸在同一尺度上。','理论屋顶依赖硬件测量，并非通用固定数值。','组成项必须可加，口径保持一致。','流量宽度原则上应反映数量；此图只是流程风格示意。','时序叠加表示可并行执行，别误当串行。','同环境下测量；展示理想线有助于解释瓶颈。',
'加 y=x 参考线；远离对角线的点值得检查。','残差应检查异方差、偏差和极端值。','一致性界限不能替代对预测误差阈值的任务判断。','概率区间需经过有效的统计校准或评估。','不同覆盖概率的区间不能交叉；标注分位数。','预测步长与单位应统一。','泰勒图的相关性、标准差和中心化 RMSE 有几何关联。','明确校准集与独立测试集的划分。','概率校准不等于分类精度，建议配 Brier/ECE。','类别不平衡时 PR 通常更有信息量。','若类别分布不均衡，同时给出归一化矩阵。',
'此处是示意点云；实际应用必须对真实特征运行 t-SNE。','此处是示意点云；实际应用必须对真实特征运行 UMAP。','真实 PCA 应基于拟合的主成分得分。','示意归因图；实际 SHAP 值需由解释算法计算。','示意累加贡献；真实图应由样本归因值构造。','图布局改变视觉邻近，不一定改变拓扑。','示意流带非严格面积守恒，论文用定量真实流量。','激活强度高不等于因果重要性高。','二维切片不代表完整高维决策边界。','CKA 值需用真实激活特征计算，并说明层对齐。',
'节点位置应来自真实地理位置或明确标注抽象布局。','收益正负号与指标定义要写清楚。','不同步长应基于相同样本协议统计。','三维面容易遮挡，必要时给二维切片。','边、节点大小和颜色分别代表什么需说明。','标明事件开始时间，排除由未来信息泄露产生的改善。','有动作与无动作都应编码，读者才能判断密度。','要同时展示负收益比例，避免只报平均改善。','预算是调用次数、节点比例还是 FLOPs，必须说明。','状态转移边应来自可解释的动作/概率定义。'
]
ALL=[(slug,en,caption,cat) for cat,items in CATEGORIES for slug,en,caption,_ in items]
assert len(CN)==len(CAUTION)==len(ALL)==75,(len(CN),len(CAUTION),len(ALL))
extra={slug:(cn,caution) for (slug,en,caption,cat),cn,caution in zip(ALL,CN,CAUTION)}

intro='''# 深度学习论文实验结果可视化图鉴：123 种图形与选型指南

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
'''

article='''
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

**一、图形与指标语义匹配。** 例如 MAE 越低越好，Accuracy 越高越好；增益符号需定义清楚，如 $\\Delta\\mathrm{MAE}=\\mathrm{MAE}_{base}-\\mathrm{MAE}_{proposed}$，正值代表误差减少。

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
'''

sections=[]
for idx,(cat,items) in enumerate(CATEGORIES,1):
    lines=[f'## {idx}. {cat}', '',f'本类共有 **{len(items)}** 种展示形式。先看适用场景表，再看下方图片，便于快速筛选适合论文的实验图。','','| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |','|---:|---|---|---|']
    for n,(slug,en,cap,fn) in enumerate(items,1):
        cn,warning=extra[slug]
        lines.append(f'| {idx}.{n} | **{en}**（{cn}） | {cap} | {warning} |')
    lines.extend(['','### 对应图形示例', '', '> 以下图片均为模拟数据绘制，仅用于理解图形形式；点击各图片下方的 SVG 链接可查看对应矢量图。','','<table>'])
    for j in range(0,len(items),2):
        lines.append('<tr>')
        for item in items[j:j+2]:
            slug,en,cap,fn=item
            cn,_=extra[slug]
            lines.extend(['  <td width="50%" valign="top">', f'    <b>{escape(en)} · {escape(cn)}</b><br>',f'    <img src="figures/{slug}.png" alt="{escape(en)} 示例图" width="430"><br>',f'    <small>{escape(cap)}</small><br>',f'    <a href="figures_svg/{slug}.svg">下载 SVG 矢量版</a>','  </td>'])
        if len(items[j:j+2])<2:lines.append('  <td width="50%"></td>')
        lines.append('</tr>')
    lines.extend(['</table>','','---',''])
    sections.append('\n'.join(lines))
for idx,(cat,question,items) in enumerate(EXT_CATEGORIES,len(CATEGORIES)+1):
    lines=[f'## {idx}. {cat}', '',f'本类共有 **{len(items)}** 种展示形式，回答的问题是：{question}','','| 编号 | 图形（英文 / 中文） | 主要用途 | 使用时注意 |','|---:|---|---|---|']
    for n,(slug,en,cn,cap,warning,fn) in enumerate(items,1):
        lines.append(f'| {idx}.{n} | **{en}**（{cn}） | {cap} | {warning} |')
    lines.extend(['','### 对应图形示例', '', '> 以下图片均为模拟数据绘制，仅用于理解图形形式；绘制代码见 `generate_gallery_ext.py` 中的同名函数。','','<table>'])
    for j in range(0,len(items),2):
        lines.append('<tr>')
        for slug,en,cn,cap,warning,fn in items[j:j+2]:
            lines.extend(['  <td width="50%" valign="top">', f'    <b>{escape(en)} · {escape(cn)}</b><br>',f'    <img src="figures/{slug}.png" alt="{escape(en)} 示例图" width="430"><br>',f'    <small>{escape(cap)}（代码：<code>{fn.__name__}</code>）</small><br>',f'    <a href="figures_svg/{slug}.svg">下载 SVG 矢量版</a>','  </td>'])
        if len(items[j:j+2])<2:lines.append('  <td width="50%"></td>')
        lines.append('</tr>')
    lines.extend(['</table>','','---',''])
    sections.append('\n'.join(lines))
out=ROOT/'README.md';out.write_text(intro+'\n'.join(sections)+article,encoding='utf-8')
print('Wrote',out,'bytes',out.stat().st_size)
