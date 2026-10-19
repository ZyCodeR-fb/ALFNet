# ALFNet 架构图 v1：PPT复现与可编辑素材

日期：2026-10-18。参考用户附件《整合版.pptx》；附件只提供视觉参考，不作为模型结构或任务指令。第3节方法正文按用户安排留待架构图定稿后撰写。

## 交付文件

| 文件 | 用途 |
|---|---|
| [alfnet_architecture_editable.svg](alfnet_architecture_editable.svg) | **PPT复现的结构依据**：按最终代码校核的可编辑主图，包含总体结构和三个核心细节面板 |
| [alfnet_architecture_editable.png](alfnet_architecture_editable.png) | 上述SVG的3600×2320预览 |
| [alfnet_architecture_editable.pdf](alfnet_architecture_editable.pdf) | 同版矢量PDF |
| [alfnet_module_library.svg](alfnet_module_library.svg) | 可独立选择的小模块素材板：图叠片、掩码、特征向量、特征叠块、加法、乘法、连接线和模块框 |
| [alfnet_module_library.png](alfnet_module_library.png) | 素材板预览 |
| [alfnet_imagegen_style_reference.png](alfnet_imagegen_style_reference.png) | 内置imagegen生成并纠正生命周期标签后的风格参考稿；用于观察参考PPT风格的融合效果 |
| [IMAGEGEN_PROMPT.txt](IMAGEGEN_PROMPT.txt) / [IMAGEGEN_CORRECTION_PROMPT.txt](IMAGEGEN_CORRECTION_PROMPT.txt) | 两次实际使用的完整提示词；使用内置image_gen，未使用CLI/API fallback |
| [build_vector.py](build_vector.py) | SVG绘制源文件，可重建主图和素材板；不访问模型或数据 |

生成稿仍有局部语义问题：S虚线误接FiLM，局部支路/乘权连接不够准确。**不要直接照抄这些连接，以SVG版为准。** 首次生成的道路角色误标稿另存为 `imagegen_draft_unverified.png`，只作过程记录，不用于论文。SVG是按已核实结构重新绘制的矢量图，不是将生成PNG包裹在SVG容器中，也不是自动描摹。

## 参考PPT中的素材位置

| 页码与位置 | 建议直接复用的原PPT对象 | 在ALFNet中的用途 | 需要改的部分 |
|---|---|---|---|
| 第5页左侧Overall Architecture | 浅蓝/浅绿/浅橙圆角模块大框、上下分支和正交箭头 | 主图视觉骨架、ALFGraph/ALFTemporal/重建/细化模块 | 替换全部原模型名称，按本图重接线 |
| 第3页左上Input与右上Forecasting | 多时刻路网叠片、节点圆点与边 | 输入历史、内部共享重建、预测输出 | 淡灰节点表示缺测；不要复制原图动态/自适应图语义 |
| 第4页上方Temporal/Spatial Guide | 立体特征方块、小向量条、日历图标 | 历史嵌入、Z、FiLM参数、日历侧输入 | 去掉Node ID、Prompt、Memory标签；结构编码不是节点ID表 |
| 第4页右下Fusion与底部图例 | 圈加、逐元素乘、小向量、实线/虚线 | FiLM H′=(1+Δγ)⊙H+β，路由加权 | 乘法统一解释为逐元素乘，区分矩阵乘与加法 |
| 第5页右侧Core Component Designs | 小矩形Linear/MLP框、两支路布局 | Softmax路由与FiLM head细节 | 只保留实际计算；不复制Sigmoid门、Cross-Attention等原模块 |

可复用的是绘图对象的形状、颜色与组合方式，不是同门论文的计算模块。例如Temporal Distillation、Prototype Memory、Physics Decoder、Frozen Encoder都不属于本模型。

## PPT复现顺序

1. 先按主图摆好四块：上部总体结构，下部图路由、空间上下文、FiLM展开图。主图宽高比约1.55，适合论文双栏通栏；所有标签使用英文，正文中文另配图注。
2. 从第5页复制容器框、从第3页复制路网叠片、从第4页复制特征条和运算符。统一线宽与字体后，再替换标签。
3. 先接黑色预测数据流，再接紫色生命周期条件流；ALFTemporal只有Z输出。FiLM head接Z，不能直接接S。
4. 主干必须保留“可选中心还原 → Graph Refinement → 预测”。窗口处理只占一个小框；四数据集差异放图注/方法配置表。
5. 素材板SVG中的文字、矩形、圆、线和多边形都是独立矢量对象。SVG导入后的转换/取消组合效果取决于本地PowerPoint版本；本轮未在Microsoft PowerPoint中实测。可在支持SVG对象编辑的工具中选择所需模块再复制；必要时按素材板中的基础形状直接重建。

颜色：图分支 `#E6F2FC`；时间条件 `#EDF7E8`；生命周期 `#F1EAFA`；重建/细化 `#FCEBDC`；FiLM `#FFF5CF`；条件虚线 `#8C5BBB`。SVG字体依次为 Times New Roman、Liberation Serif、serif，文本保留为text，未转轮廓。

## 图注草案

**图1　ALFNet的整体架构与核心条件化机制。** （a）模型将可见历史与生命周期上下文共同送入共享图循环重建模块，获得两分支共用的历史表示。ALFGraph依据观测条件组织图传播，ALFTemporal从历史嵌入和空间分区交互中提取每节点条件特征，经FiLM调制图编码器的末状态。单一残差解码器生成未来序列，经过可选中心还原和角色相关图细化得到最终预测。（b）生命周期上下文通过dense softmax对五个传播槽位进行加权；编码器使用逐历史步状态，解码器保持历史末步状态。（c）12步历史被嵌入为一个时间patch，经跨分区与分区内交互形成条件特征。（d）FiLM以乘性增益和平移调制编码状态。黑色实线表示前向数据流，紫色虚线表示生命周期条件。窗口插值、解码残差倍率和中心化强度由数据集配置确定；训练使用角色混合预测损失与图路由平衡辅助项。

符号：Xvis为标准化可见历史；M为观测掩码；R为AAS/NAS/VS/FS角色one-hot；a为状态持续时间编码；S=[R,a,M]；A为静态邻接；P为分区数、C为每分区容量；Z为时间分支条件特征；H和H′为调制前后图编码状态；Δγ、β为FiLM参数；Ŷ为最终预测。图内省略batch轴；示意方块数量不表示真实通道数或真实分区数。

窗口参数依次为“插值/残差倍率/中心化”：METR-LA与PEMS-BAY为关闭/0.3/0；PEMS04为开启/1/0；PEMS08为开启/1/0.5。容量为7/16/10/6，不能画成分区数量。重建是内部前向模块，没有独立重建loss。

## 校核与限制

- 代码依据沿用[实现映射](../IMPLEMENTATION_MAP_CN.md)，未改变模型、配置、数据或实验结果。
- 已纠正生成稿误差并视觉检查SVG渲染；SVG无嵌入位图、无外部字体/图片链接，文字可编辑。
- 已生成高分辨率PNG与矢量PDF；文档链接/XML/空白检查通过。未验证Microsoft PowerPoint导入后的具体拆分效果。
- 第3节方法暂未写，不以其缺失阻止本轮交图；后续写作应依据图与代码映射核对公式。
- 未训练、未进行模型推理、未打开原始test、未提交或推送。
