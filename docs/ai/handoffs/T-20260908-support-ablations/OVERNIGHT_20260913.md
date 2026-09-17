# 2026-09-13 耗时诊断与第二轮串联

## Facts / 授权与状态

用户要求第一轮四个训练继续完成，准备第二轮并在四个完成后自动运行，最多5个训练并行；明确禁止操作其他人的PID1734562。本轮仅只读查看该进程资源与开始时间，未发送任何信号、修改参数、设置优先级或限制资源。无test、commit、push。

第一轮计划仍是 `afd9f69d16e5b59429a8eda619b7e9874e1e21cd28de4af4873ad3e86ee5bdf9`，当前源码/配置哈希检查通过。原四任务不停止、不重启、不改运行中代码。

00:32左右已运行约10小时。读取已保存last.pt：METR-LA route_shrink 已完成epoch66、best59=4.782675、patience_wait7；bounded_mae已完成69、best65=4.792943、wait4。PEMS-BAY route_shrink已完成55、best52=3.006804、wait3；bounded_mae已完成57、best57=3.010424、wait0。对比Full LifeAvg MAE分别4.638117/3.039729，METR-LA仍退化、PEMS-BAY暂有约1%改善；均为未完成结果。没有根据不完整结果宣称通过完整门槛。

## 耗时原因与证据

1. **明确的资源配置原因**：第一轮4进程只共享2个整阶段GPU槽，整次train_one_epoch/validate持有锁，其余任务等待。epoch日志的时间间隔包含训练、验证、等待、保存，不能视为纯GPU计算耗时。该配置由助手上一轮选择，过于保守；不能把它全部归因于外部进程。
2. **实际观测**：METR-LA当前相邻epoch间隔中位数约8.0/8.3分钟，PEMS-BAY约9.9/10.5分钟，范围3.2–22分钟。最新10个平均可达11–12分钟；并非每epoch都十几分钟。两槽的竞争和阶段锁非严格公平会造成任务间波动。
3. **可核验历史对照**：旧六消融Mask-only日志中位数METR-LA9.77分钟、PEMS-BAY12.15分钟（当时6进程4槽）。尚未定位用户记忆中5个并行/5分钟那组的同协议、同数据集记录，因此不能声称当前比所有旧实验慢一倍。
4. **外部竞争边界**：只读观察PID1734562于2026-09-12 20:55:29开始，晚于本轮14:31启动。它不可能解释最早6小时的全部耗时；之后共享GPU/CPU可能增加波动，但缺少持续遥测不能分解其因果占比。显存瞬时观察曾为5GB，与用户观察35GB不同，说明不能拿一次快照假定固定负载。
5. **工作量**：PEMS-BAY每epoch489个train batch（batch64），还要完整验证；METR-LA样本数不同。网络保留12步GRIN及图循环、12步AR decoder，存在大量小kernel/CPU调度；这是代码路径事实，未做profiler，不能把它定量声称瓶颈。新路由只新增144参数，bounded_mae不新增预测参数，没有结构规模翻倍的证据。
6. **验证并未重复反传**：validate已有torch.no_grad；主表和早停共享同一验证forward，非“验证忘关梯度”或“每epoch为了主表重复评估”的问题。

原日志没有为GPU_WAIT/ACQUIRED分别打时间戳，不能追溯精确等待占比。第二轮会分别记录wait_seconds与execution_seconds，后续可真正比较。

## 第二轮独立候选与身份

代码已隔离在 `outputs_v4/round2_workspace_20260913/`，复制当前模型/脚本/配置，数据与历史输出使用只读目的的共享路径（新结果只写新输出目录）。不修改第一轮哈希绑定的任何代码；第一轮原collector可正常收尾。

第二轮计划位于隔离目录 `configs/paper/two_speed_round2.json`，身份
`159322422fa9ece24aa434832405b0824e6e1d867a411da8388d4bb7831c27b8`。
结果根 `outputs_v4/two_speed_round2/<身份>/`；共4个seed32训练。

| 候选 | 单一改动 | 区分的问题 |
|---|---|---|
| freshness（C-E） | 由完整观测mask生成逐步距上次观测的间隔、累计观测比例；零初始化小投影加到时间分支token。保留原图/时间架构、分区和FiLM | 原age是生命周期持续时间，显式新鲜度是否改善Cold/Warm与陈旧历史利用 |
| refiner_uniform（C-R） | 仅把graph_diffusion_role_scale的四角色系数设成原四系数均值0.5625；其余所有状态、路由、重建、时间分支保留 | Mask-only优势是否来自最后细化层的角色缩放，而非应整体去状态 |

两候选都从原Full配方独立训练，不叠加第一轮route_shrink或bounded_mae。freshness只是合法mask的确定性表示，不增加外部信息/节点ID；q=0时缺省gap=窗口长度+1。新投影零初始化无额外随机数消耗。

训练仍为两数据集原Full的100 epoch上限、TF40、patience10、early-stop-start40、batch64、原损失/学习率/验证选点，METR-LA容量8、PEMS-BAY容量11。未为赶早晨偷偷缩短预算或改变主表口径。早停可能继续刷新，无法保证明早四候选全部完成。

筛选：通用LifeAvg三指标最多退化0.5%、任一33单元最多退化2%；freshness要求METR-LA Cold/Warm MAPE各改善至少2%、MAE/RMSE不退化。refiner_uniform要求PEMS-BAY Steady/Fault MAPE各改善至少2%、Join三项最多退化0.5%，METR-LA Unseen三项最多退化0.5%。跨数据集逐一判断；不是已选新Full。

## 完成后自动接续

串联入口：根目录 `scripts/run_two_speed_round2_chain.py`；持久tmux会话
`two_speed_overnight_20260913_v2`。状态文件：
`outputs_v4/two_speed_overnight_20260913/two_round_summary.json`；可读报告：同目录`two_round_report.md`。

- 等第一轮4个status=complete且各自COMPLETE存在，并等原launcher完成第一轮收集；不启动第二个collector争写同一文件。
- 第二轮preflight.json必须passed且计划身份相同，再做builder --check。
- 不覆盖已有第二轮launcher_status；有失败则输出attention/错误文件，不自动重启。
- 第二轮最多4训练进程；启动时根据现有GPU计算进程数进一步限制，使当时合计不超过5。其他人的新任务可随时启动，不能由本任务保证全机器永远不超过5。
- GPU阶段槽由2增为3；每次进入需空闲至少22GiB，正常自身峰值预计最多两PEMS-BAY+一METR-LA约39GiB（另计CUDA上下文/缓存）。给其他35GiB负载留余量；不足时只等待，不干预对方。
- 每30秒更新两轮汇总；第二轮完成后自动收集33单元并结束一次性串联。无需新会话手动启动，无test或Full自动晋升。

## Completed / Modified but unverified / Not started / Ruled out

- Completed：Git和第一轮身份核验；日志/last.pt只读诊断；隔离第二轮代码、计划与启动脚本；24项相关测试通过、3 deselected；串联“缺status/运行中/缺COMPLETE不得启动”检查通过；第二轮builder --check和生产训练入口计划加载通过；串联已启动并显示waiting_for_round1。
- Completed preflight：00:45全部四CPU/四GPU batch64预检通过；八个旧Full/控制预测一致性检查最大误差0。METR-LA峰值9.332/9.341GiB，PEMS-BAY14.670/14.668GiB。等待队列曾单独重载一次以容忍结果文件写入中的JSON，并避免重复collector；未停止任何训练。新queue PID1923794为00:45观察，后续重查。
- Modified but unverified：新候选完整训练收益未验证，完整仓库test相关测例不运行。资源门槛是保守准入，不是共享GPU未来负载保证。
- Not started：第二轮正式训练；第一轮最终结果收集；多seed及test。
- Ruled out：停止当前训练、操作PID1734562、复用旧final_plan、将单次显存或epoch时间作为精确瓶颈归因、承诺明早全部完成。
