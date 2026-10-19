# 最终成图核验记录

2026-10-19；分支 `paper/final`；HEAD `e02b036d665e0c82f33afd4195b7a6ea8cad1c18`。本轮为绘图和静态实现审计，不是模型数值重评估。

## 逐项结构检查

| 检查项 | 结论与图中处理 |
|---|---|
| 真实输入 | 一次性绘制 X、Lifecycle Context、Static Graph；S/R/a/M、A/G 是引用端口，不是新增输入。G 与 A 区分图属性包和邻接矩阵 |
| 场景与角色 | 四种观测场景用随时间改变的角色条；不会把 Join/Fault 当作整窗固定角色 one-hot；符号表给出实际角色映射 |
| Visibility | 未画 mean-mask/state_encoder 为有效分支；保留 M 与年龄，q 仅说明可见计数 |
| GRIN 共享性 | 所有输入先统一经过一份 BiGRIL；方向间无串联；正反表示对齐所有历史步再合并，两个主分支共享其处理后输出 |
| 观测保留与插值 | GRIN→保留观测→可选插值覆盖，q=0 保留 GRIN；不是按 q 分为互斥的 GRIN/插值执行路线 |
| 中心化与恢复 | αv 从 Xfilled 减去，解码后加回，再进入 refiner；不是逆标准化；v 无观测时为零 |
| 时间卷积/日历 | 单一 12 步 patch；9 输入通道→64；日历归一化输入与末时刻 embedding 两条实际用途均保留 |
| 结构嵌入 | G→共享结构 MLP→32；说明明确无坐标填零，不暗示有可训练 node-ID embedding |
| 空间分区 | P=32；容量 C 不当作分区数；矩阵标明 P/C 两轴；四数据集 C=7/16/10/6 在配套表列出 |
| 注意力轴与次序 | 先跨分区同槽位，再分区内 4 latent tokens 汇聚/回传；1 个块；未画多个 temporal patches、×5 层或直接 C×C 注意力 |
| 时间输出 | 恢复原节点顺序，Z 为 N×160；不存在时间支路的独立预测输出 |
| FiLM | head 160→32→128，Δγ/β 各 N×64；参数流只进入两个编码末状态；没有 S→FiLM head 直连或每预测步重复调制 |
| 编解码层数 | encoder 两层、decoder 两层、隐藏64；输入含逐步 S，解码持有末步 Sᴸ |
| DCRNN/GRU | 代表 cell 只展开一次；r/u/c 与候选 r⊙h、隐藏状态更新式对应 `GRUCell.forward`；层/门权重独立 |
| 五专家与 router | S→线性→softmax；五路 dense mixture；I/Mean/WM/Fwd/Bwd；WM/Fwd 同传播形式但独立投影的事实在说明中披露 |
| 残差 AR | 显示 yₖ₋₁+dΔyₖ；回环从预测结果回到输入，初值 X̄ᴸ；说明定义 Δyₖ=Linear(hₖ)；没有 refiner 输出回送 |
| Refiner | Y₀ 三路拼接→3/32/1 MLP→tanh→0.05 s(Rᴸ)→加回 Y₀；A 只用于传播，Rᴸ 只用于固定倍率；说明定义 P⁺/P⁻ 与角色倍率 |
| 单一输出 | 仅最右侧一个 12 步预测；曲线是手工示意，不取实验数据 |
| 训练/推理 | 预测/角色均衡/router 辅助目标及 masked teacher forcing 位于独立底部；没有训练真值流入推理主线 |
| 命名与代码归属 | 不新增过多缩写；GRIN、DCRNN、Crossformer、FiLM 来源未改称原创；蓝色为功能区，不宣称 refiner 是 ALFGraph 类内部定义 |

## 本轮发现并修复的图形问题

1. 初始布局中 Lifecycle Context 标题超宽：改为居中，使其完整容纳于输入框。
2. 双向重建的连线曾穿越另一方向的单元：调整为外侧汇合，并明确 `Align + merge / All 12 steps`。
3. FiLM 参数折线穿过图分支标题：重新安排标题位置，为参数线保留专用通道。
4. 第一版 AR 回环直接取第二层隐藏状态：改为经过预测增量与 anchor 相加后再回送。
5. Reference Restore 的圈加未完整连入主路：接入解码输出、αv 和修正前预测。
6. Cross-partition 图示轴容易混淆：增加 P/C 轴，矩阵列表示分区，行表示槽位。
7. 时间分支输出连接落在分区内交互中部：改从 broadcast 后的槽位表示引向恢复节点顺序。
8. 图中文字过小：所有文本提升到至少 28 设计单位，按 180 mm 图宽约 7.14 pt；缩短框内次要标签，完整名称和公式置于说明。
9. 放大字号后出现的标题间距、端口文字、Gather 箭头和 cell 分隔线穿字：逐一调整，最终检查清零。
10. 静态图小矩阵原先只是任意示意：改为与所画六节点边关系一致的 6×6 邻接图元。
11. 结构 MLP 原先仅用 A 端口，容易遗漏坐标/属性来源：改为 G 引用；A 继续专指邻接。
12. 中心化系数与 GRU 候选 c 可能重名：图中采用 α，所有代码保持原样。

## 实际执行的检查

- 逐项复读 ALFNet、ALFGraph、ALFTemporal/PatchSTG、GRIN、GRU、dense router、flow adapters 的相关计算路径，并对照最终配置清单。完整第一阶段审计已覆盖四份外层、八份专家配置和数据角色装配。
- 四个选定外层 YAML 在内存 LF 归一化后 SHA256 与最终 manifest 一致；不改写配置。源文件与生成制品哈希记录见 `qa/final_artifact_checks.json`。
- 本地独立 headless Chrome 解析和渲染主图、素材库总览以及 19 个单独模块；检查画布文字越界、标注宽度、文字重叠、连线穿字及嵌入位图数。
- 主图 148 个文本对象，最小字体 28；PNG 6000×3960；19 个独立模块保留可编辑 SVG 文本和图元；SVG 内位图数为 0。
- 查看整图预览、1200×792 缩小预览和模块库预览，复核连线端点、阅读顺序、颜色/字号和遮挡。最终主图与全部独立模块的自动几何问题数为 0。
- 执行 `git diff --check`，确认 tracked/staged diff 为空；既有未跟踪历史目录不动，只在新绘图目录下新增文件。

自动几何检查用于补充人工看图，不能证明模型语义；语义依据是上表及 README 的代码映射。正文没有第3节完整方法描述可逐句同步，本轮没有擅自补写正文；可直接采用交付的命名、图注和公式说明。

## 证据范围与未验证项

- 没有运行模型 forward、训练、消融或测试集评估；未读取原始数据、测试 payload、checkpoint。没有数值复现实验结论。
- 第一阶段已记录：相关冻结源码指纹中 19/20 一致，`experiments/v4_runner.py` 因退役注册清理与旧计划不同。因此不能宣称完整历史运行环境/旧计划哈希全部重现。当前审计到的 ALFNet 损失路径已再次核对。
- SVG 已验证为基础矢量/文本，并逐一浏览器渲染；PowerPoint 原生拆分/字体替换行为尚未在本机实测，交付不声称其已通过。
- 无模型/配置/结果修改，无 commit、push、分支切换或新依赖安装。

机器报告：`alfnet_architecture_audit.json`、`qa/module_audits.json`、`qa/column_readability_1200_audit.json`、`qa/final_artifact_checks.json`。迭代失败/修复前快照仅保存在 `qa/`，最终以根目录 `alfnet_architecture.png` 为准。
