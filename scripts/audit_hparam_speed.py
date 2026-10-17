"""Offline wall-clock audit; never read test data or execute models."""
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
rows = []
for name in ('hparams_completion_20260928', 'baseline_recovery_20260927', 'baseline_extension_20260927'):
    for state in (ROOT / 'artifacts/experiments' / name).glob('*/queue_status.json'):
        jobs = {j['key']: j for j in json.loads(state.read_text())['jobs']}
        old = state.with_name('queue_status_before_powerloss_recovery.json')
        if old.exists():
            jobs = {j['key']: j for j in json.loads(old.read_text())['jobs']}
        for key, job in jobs.items():
            log = state.with_name(key + '.log')
            if not log.exists():
                continue
            matches = re.findall(r'^epoch\s+(\d+).*?\((\d\d):(\d\d):(\d\d)\)', log.read_text(), re.M)
            times = [(int(e), int(h)*3600 + int(m)*60 + int(s)) for e,h,m,s in matches]
            # Exclude stage resets and non-consecutive entries, preserve midnight wrap.
            intervals = [(b-a)%86400/60 for (e,a),(f,b) in zip(times,times[1:]) if f == e+1]
            if not intervals:
                continue
            ds = next(d for d in ('METRLA','PEMSBAY','PEMS03','PEMS04','PEMS08') if d in key)
            model = 'ALFNet' if name.startswith('hparams') else key.split('_')[-1]
            row = dict(study=name,key=key,dataset=ds,model=model,logged_epochs=len(times),
                       median_epoch_minutes=statistics.median(intervals),
                       epoch_minutes=intervals,log=str(log.relative_to(ROOT)),status=job['status'])
            if job.get('finished') and job.get('started'):
                row['worker_hours'] = (job['finished']-job['started'])/3600
            rows.append(row)
out = ROOT / 'paper/hparams_completion_20260928'
(out/'speed_audit_20260930.json').write_text(json.dumps(dict(
    timing_scope='epoch log intervals include training, validation and checkpoint IO; concurrent non-isolated runs',
    rows=rows),indent=2)+'\n')
lines=['# 断电与训练速度核验（2026-09-30）','',
'断电时24项中17项完成、5项中断、2项未开始。五个last.pt可读且计划及配置匹配；保持原计划，从已保存epoch边界恢复优化器、调度器和随机状态。未完成的半轮重新计算，旧日志保留。', '',
'## 日志实测（分钟/轮）','',
'每项先计算连续epoch日志间隔的中位数，下表给出同模型/数据集各项中位数范围。包含训练、完整validation及保存，不是纯训练forward计时；不同日期并发和batch不同，不可作为论文公平速度基准。','',
'| 数据集 | 模型 | 单轮分钟范围 | 已记录轮数范围 |','|---|---|---:|---:|']
for ds in ('METRLA','PEMSBAY','PEMS03','PEMS04','PEMS08'):
    for model in sorted({r['model'] for r in rows if r['dataset']==ds}):
        group=[r for r in rows if r['dataset']==ds and r['model']==model]
        t=[r['median_epoch_minutes'] for r in group];n=[r['logged_epochs'] for r in group]
        lines.append(f'| {ds} | {model} | {min(t):.2f}–{max(t):.2f} | {min(n)}–{max(n)} |')
lines += ['', '## 解释边界', '',
'ALFNet本轮每项参数量为1,037,529，容量变化未增加参数量，但会改变token计算量。因此不能把慢简单归结为参数堆积。当前代码包含GRIN输入填补、图专家与时间专家；串行循环和多分支计算是合理开销来源，具体占比尚未做profiler验证。', '',
'ALFNet已完成项训练50–84轮，early_stopping_start_epoch=40；多数基线更早允许早停。既有单轮开销也有轮数差异。BAY本轮需6个ALFNet任务，而上一轮新增基线大部分是较短flow任务，不能仅用24或29个任务的数量推断总时长。Merlin轮数在此包含教师与学生两个阶段。', '',
'恢复仅增加独立队列包装，不改冻结训练、模型、配置及test状态。GPU利用率高不等价于算子效率高；要定位CPU等待、数据搬运、图循环和小kernel开销，仍需在固定设备/并发/batch/精度下单独剖析。当前先完成获授权实验，不改变研究实现。']
(out/'POWERLOSS_AND_SPEED_20260930.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
