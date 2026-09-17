# METR-LA Full 损失四候选执行

用户已授权开启上一轮建议的四候选。仅METR-LA、seed32；无PEMS-BAY排队、test、commit/push或他人进程操作。代码在根仓库实现，不创建新的outputs源码工作目录。

计划：`configs/paper/metrla_mae_tail_v1.json`。
身份：`d90516065cc844010bdb13ee1e2a2552deb6d58535030bee56909a588e0d93cb`。
输出：`outputs_v4/metrla_mae_tail_v1/d90516065cc844010bdb13ee1e2a2552deb6d58535030bee56909a588e0d93cb/`。
Git：paper/final，HEAD9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb，upstream0/0；全部既存dirty改动保留。

## 冻结候选

令a为标准化绝对残差。

| variant | 逐位置损失 |
|---|---|
| mae_tail025 | a + 0.25·relu(a−1)²/2 |
| mae_tail050 | a + 0.5·relu(a−1)²/2 |
| mae_focal_raw02 | a·[2sigmoid(0.2·sigma_train·a)−1] |
| mae_mse010 | 0.9a + 0.1a² |

作者util.py已通过只读curl成功取得，与前次网页抓取失败状态不同。源实现使用`2sigmoid−1`，gamma1、无detach；engine.py明确先inverse_transform再求loss。本候选保留原始速度下beta0.2，并将loss除以训练std，以匹配项目MAE尺度；mask及role归约沿用项目而非源代码的零值过滤。源文件证据在`paper/optimization/full_mae_residuals/source_audit/`并纳入计划哈希。

模型结构/参数及其初始采样机制不变。所有候选从头训练，batch64、workers0、max_epoch100、patience10、early_stopping_start_epoch40、teacher forcing40。base和role同时替换损失，role混合0.1及辅助项不变。checkpoint仍按validation LifeAvg MAE严格改善选取，平手保留较早epoch。

## 门槛与结果

所选checkpoint必须同时满足：LifeAvg MAE≤4.5668210733、MAPE≤13.8789127228%、RMSE≤6.8068354782；Unseen MAPE改善≥2%、MAE/RMSE不退化；Join/Cold/Warm/Ready各格最多退化0.5%；其余各格最多退化2%。对MoGERNN赢≥33格、原8基线第一≥25格、含GWNET第一≥11格。没有降低旧Full-MAE的失败门槛。

自动汇总由`scripts/collect_metrla_mae_tail.py`完成，写`candidate_summary.json`和`report.md`；不会自动替换Full、追加训练或打开test。

## 验证与运行状态

- 15项合成损失/梯度/mask/role、旧机制兼容测试通过。
- builder `--check`、正式plan loader、四配置完整比较通过；仅loss、workers、标签/路径差异，嵌套模型YAML内容一致。
- 4CPU/4GPU预检通过；Full及四候选严格加载原Full权重后8窗口预测最大误差0。四同步探针均exit0，16训练batch分别约17.98–18.15秒，各峰值预留10.1055GiB，2验证batch约0.68–0.98秒；这不是完整epoch的耗时估计。
- 2026-09-13 19:48后已启动正式tmux `metrla-mae-tail-20260913`；实际四worker状态见输出根`launcher_status.json`与`logs/`。启动器按显存准入间隔拉起，最多四训练，无额外阶段槽。
- 上次自动审批因额度耗尽拒绝并发探针；用户要求继续后核对无旧worker，再完成探针并启动，未绕过拒绝或重复启动。
- 完整仓库测试未执行；其范围包含本轮无授权的数据/真实test相关工作，不以此扩大执行范围。

预检命令：`/home/xd/anaconda3/envs/ReaCH/bin/python -u scripts/preflight_metrla_mae_tail.py`。
并发探针：`.../python -u scripts/run_metrla_mae_tail.py --probe`，每候选16训练batch、2验证batch，无checkpoint。
正式启动：`.../python -u scripts/run_metrla_mae_tail.py`，持久tmux会话；须先通过计划绑定的preflight及concurrency_probe。
worker显式拒绝非METRLA或非train/validation context；只读显存准入，不操作其他任务，最多4训练并行，不使用整epoch/阶段槽排队。

## 兼容性边界

原Full/Full-MAE checkpoint保留。新损失与plan注册改变根代码哈希，因此旧根计划的当前源码检查可能过期；旧完成证据继续有效，不重写旧身份。此前wave与round2隔离源码保持不动。本轮模型新收益、多种子稳定性和PEMS-BAY迁移均未知。
