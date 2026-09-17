# METR-LA 五对照执行 — 2026-09-13

## 授权与范围

用户同意同时尝试小改动和大替换，随后明确收窄为：先跑5个METR-LA，若效果不好不再跑PEMS-BAY；GWNet架构参考 `references/GWNet`，优先原样复制、最小改动适配。

本轮仅5个METR-LA seed32；PEMS-BAY不在计划/队列中。Full及既往两轮保留，不重启旧消融、trajectory/q分析，不打开test、不commit/push、不操作PID1734562。

代码与生成配置位于独立目录 `outputs_v4/wave_fusion_workspace_20260913/`。工作区根代码不合并、不改写。该目录在Git忽略的输出树内，正式启动会另外保存完整源码/配置snapshot，交接必须保留目录及snapshot。

独立计划：该目录的 `configs/paper/metrla_wave_screen_v1.json`。
身份：`ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85`。
结果根：`outputs_v4/metrla_wave_screen_v1/ef6bfd48f3c4b69840d06a7a3b2a61e03aa798de361efed26dbccbbd2114ba85/`。

## 五个运行

| variant | 结构与损失 | 参数量 | GPU batch64峰值reserved |
|---|---|---:|---:|
| full_mae | Full不改结构，仅MSE及角色误差改MAE | 1,037,529 | 10.064 GiB |
| temporal_mse | 保留ALFGraph、路由、循环解码，图输入侧轻量TCN残差；原MSE | 1,046,026 | 10.328 GiB |
| wave_mse | GRIN+ALFTemporal条件化源GWNet直接预测器，替换ALFGraph；MSE | 980,072 | 4.398 GiB |
| wave_mae | 同上一结构，仅MAE目标 | 980,072 | 4.398 GiB |
| gwnet_mae | 原生源GWNet架构+最小协议适配；MAE参考 | 307,552 | 1.660 GiB |

TCN采用节点共享16通道、门控tanh/sigmoid、源GWNet的1/2重复膨胀序列，总8层；零初始化输出投影。只改变图分支raw前缀，不改生命周期元数据或patch输入；参数归入graph学习率组。它是我们新增的轻量模块，不冒充原样复制的整个GWNet。

源GWNet模型文件逐字复制，源LICENSE同时保留。现代torch四维输入兼容由外层将源Conv1d转换为同权重Conv2d；归纳协议需要将节点专属因子替换为两个共享结构编码器。其余主干forward直接调用源函数。融合FiLM挂在skip ReLU之后、end_conv_1之前；与上一次未训练原型的pre-ReLU位置不同，不能沿用上次速度/初始化作为同一模型证据。详细适配及源训练器差异见 `docs/comparison_model_audit.md` 的本轮补充。

## 公平性与选择

共享：速度v/channel2、12/12、METR-LA容量8、train K/validation K∪V、train时间K的scaler、batch64、num_workers0、100epoch上限、patience10、early-stop-start40、严格33单元LifeAvg MAE选点/早epoch平手保留。

Full/TCN保留TF40线性衰减；直接预测的wave/GWNET明确free_run。Full及wave保留原分组学习率；GWNET保留项目源模型配置的lr0.001。损失对照wave_mse/wave_mae除loss外相同初始化/训练设置。由于结构与源配方不同，GWNET与Full不是单一变量的架构因果实验。

GWNET基线无GRIN、生命周期特征、FiLM或角色损失。使用现有项目AdamW/标准化MAE，不冒充源engine的Adam/raw-MAE/clip5完整训练复现；本次用户指定的是源架构优先。不得把本轮结果称为原论文transductive benchmark。

冻结门槛：相对Full LifeAvg MAE至少改善1%、MAPE至少2%、RMSE不退化；Unseen MAPE至少改善2%、MAE/RMSE不退化；Join/Cold/Warm/Ready各指标最多退化0.5%；其余单元最多2%。大替换还必须相对同轮GWNET LifeAvg MAE/MAPE至少改善3%、RMSE不退化。原生GWNET只作为参考，不给它标记方法晋升通过。

只有达到门槛才有继续PEMS-BAY的证据；本队列不会自动启动PEMS-BAY，也不自动晋升Full。仍仅seed32筛选，不能声称多种子稳定。

## 已完成验证

- 13合成测试passed：源forward/首卷积梯度精确一致、原型恒等FiLM、teacher拒绝/target隔离、TCN零初始化/元数据保持/因果性/节点独立性/梯度、既有损失和33单元collector行为。
- Full旧checkpoint与新代码8自然validation窗口raw预测最大差0；同checkpoint加载到零初始化TCN后也为0。
- 五候选5CPU batch1＋5GPU batch64各一train/validation批，真实训练入口/optimizer路径全部通过；不是精度结果。
- `audit_wave_screen.py`配置审计通过：5个唯一身份、仅METR-LA、共同协议、嵌套图/时间配置不变、GWNET无ALF专用输入、源文件逐字相等。
- 完整仓库测试未执行（包含其他数据集/真实test测例）；不以13个测试代替完整测试或正式泛化结论。
- 最初预检漏传output_root而失败，已修复。失败身份d5e94576…及原计划/配置/错误日志保存在其输出根的failed_preflight下，未开启正式训练、未覆写历史证据。

## 运行与查看

worker对train/validation分别记录时间；没有整epoch锁。launcher逐个检查free显存，按单项预检reserved峰值×1.25＋8GiB准入，最多同时5个。并发预检通过后才允许正式启动。

2026-09-13 14:14 +0800复核：顺序预检及五进程同步16批次并发预检全部passed，五训练已于13:48–13:49启动。tmux `metrla-wave-screen-20260913`、launcher2381162和五worker均存活；额度中断未终止训练。实际状态以同目录 `launcher_status.json` 和当前进程共同核验，不要重新启动。

目录澄清：进程运行的是独立目录代码；data/data_v4/references/paper/outputs_v4是共享根目录的符号链接，不是数据集副本。对根代码的复核显示154个Python文件相同，4个既存文件修改、9个Python文件新增。该源码目录在outputs下会造成混淆，但运行期间不能当缓存清理；结束后整理选中的根目录补丁，保留已有snapshot及实验身份。

结果文件：`report.md`简表、`candidate_summary.json`包含33单元/相对Full和GWNET差异/失败门槛；五日志在logs/；每个运行按正常训练入口保存last/best checkpoint以供恢复。PEMS-BAY的pemsbay_scheduled恒为false。

运行脚本（cwd必须为独立代码目录）：

```bash
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/build_metrla_wave_screen.py --check
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/preflight_metrla_wave_screen.py
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/run_metrla_wave_screen.py --probe
/home/xd/anaconda3/envs/ReaCH/bin/python scripts/run_metrla_wave_screen.py
```

这些是执行记录/恢复线索，不是让下一会话重复启动。launcher拒绝覆写已有status；恢复训练需先核验进程和last checkpoint，再做有记录的单项resume。
