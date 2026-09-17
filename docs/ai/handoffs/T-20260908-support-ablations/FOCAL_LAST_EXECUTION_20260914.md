# 最后四组METR-LA seed42执行

用户明确选择2个对照+2个新候选。正式身份：
`8e2a8b3b2009ecb5a0ddf3225699290ba39fee68967de9c58449f5980f1f59ca`。
配置 `configs/paper/metrla_focal_last_v1.json`；输出根
`outputs_v4/metrla_focal_last_v1/8e2a8b3b2009ecb5a0ddf3225699290ba39fee68967de9c58449f5980f1f59ca/`。

源码与实际运行cwd均为根仓库；snapshot仅作为身份备份。Full与Focal各seed42确认；新候选分别Focal+0.1正交复数FFT-L1，以及Focal+444参数可见历史直接修正。四组从头训练、最大4并发、最多100epoch，早停规则不变。patch容量8，非分区数量。不扩BAY/test，不自动晋升、不启动下轮。

## 已完成的检查

- `python -m pytest tests/test_focal_last.py tests/test_mae_tail_losses.py -q`：19 passed。覆盖新损失生产路径、完整目标选择、无完整目标退回Focal、缺失梯度、幅相区别、历史隐藏污染、q0、零初始化/RNG/参数梯度、节点置换、平移性质和配对RMSE拒绝门槛。
- `python scripts/build_metrla_focal_last.py` 与 `--check` 通过；独立loader/config审计通过。配置差异仅身份路径、seed32→42、workers8→0（沿用上一轮运行值）、声明的损失/唯一模块开关；其余原Full预算保持。根目录git diff --check通过。未跑可能涉及其他数据/test的完整仓库套件。
- `python -u scripts/preflight_metrla_focal_last.py`：4CPU+4GPU单训练/验证批均通过。旧Full8验证窗口原尺度预测最大差0；四组42主体参数、CPU RNG、暴露/scaler完全配对，初始预测相同，改变未来target预测差0。第一批83完整目标向量/1005合法位置；这是单批覆盖，非全训练覆盖率。
- Full/Focal/频域参数1,037,529；历史修正1,037,973。GPU batch64峰值预留10.066–10.080GiB，分配9.332–9.339GiB。
- `python -u scripts/run_metrla_focal_last.py --probe`：四进程同步16训练批+2验证批，全部exit0。训练段Full17.516s、Focal17.611s、频域18.352s、历史修正17.390s，峰值预留10.105–10.107GiB。短测中频域比Focal约慢4.2%，历史修正未呈明显变慢；单次并发微测不支持精确效率排序或完整epoch耗时保证。

以上python均 `/home/xd/anaconda3/envs/ReaCH/bin/python`。GPU命令通过沙箱外授权执行，只允许train/validation上下文。预检完整记录在preflight.json/preflight.log，探针在concurrency_probe.json/probe_logs。

## 正式启动与查看

2026-09-14 10:48:48开始启动 tmux `metrla-focal-last-20260914`：

```bash
/home/xd/anaconda3/envs/ReaCH/bin/python -u scripts/run_metrla_focal_last.py
```

启动器日志 `/tmp/metrla_focal_last_launcher_20260914.log`；每模型日志在新输出根logs/，实时状态launcher_status.json，自动摘要candidate_summary.json/report.md。控制组尚未完成时门槛显示待配对；不会用seed32绝对值替代。所有结果完成后按预注册门槛审阅，不重新搜索。

10:50核验launcher3065848、Full3065860、Focal3065986、频域3066103、历史修正3066228均存活，四worker日志均有正式train_one_epoch开始事件。GPU利用率91%，四worker分别11028/11028/11036/11026MiB。自动报告及snapshot已存在；后续读取最新launcher_status，禁止重复启动。

Git paper/final HEAD9346dc0/upstream0:0，原dirty保持。10:42只读资源检查GPU约94GiB空闲，无本任务旧训练。没有操作他人进程或原始数据，没有test/commit/push。旧已完成计划因新源码可失效，保留其冻结snapshot及全部证据，不重写旧plan。
