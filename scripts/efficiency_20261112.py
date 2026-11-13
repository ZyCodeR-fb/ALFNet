"""2026-11-12: parameter count, inference latency and peak GPU memory of ALFNet and the four representative
comparison methods on METR-LA (frozen seed-32 weights, validation batches only, no training, no test access).

Each model is built in its own training context, loads its final checkpoint and runs the first 40 validation
batches (batch size 64, 207 nodes, 12 -> 12 steps) without teacher forcing; the first 10 batches are warm-up,
the remaining 30 are timed with torch.cuda.synchronize(). The GPU is shared with other users, so latencies are
indicative (median and inter-quartile range are reported); peak memory is this process's
torch.cuda.max_memory_allocated. Output: paper/experiments_20261112/efficiency/METRLA.json and .csv.
"""
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_release import read, sha, write  # noqa: E402
from scripts.paired_interventions_20261112 import ENTRIES as NEW_ENTRIES, context  # noqa: E402

DS = 'METRLA'
OUT = ROOT / 'paper/experiments_20261112/efficiency'
BATCH, WARMUP, TIMED = 64, 10, 30


def entries():
    old = read(ROOT / 'paper/final_figures_20261017/plan.json')['entries'][DS]
    out = {m: dict(config=old[m]['config'], checkpoint=old[m]['checkpoint'], context='main')
           for m in ('ALFNet', 'MoGERNN', 'STBP')}
    out.update(NEW_ENTRIES)
    return out


def measure(model, device='cuda:0'):
    import torch
    from experiments import v4_runner as vr
    from lib.v4_seeds import seed_process
    torch.set_num_threads(4)
    r = entries()[model]
    cp = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    with context(r['context']):
        s = vr.build_v4_session(ROOT / r['config'], device=device, run_seed=32, batch_size=BATCH, num_workers=0)
        assert cp['seed'] == 32 and cp['dataset'] == DS and cp['model'] == model and cp['scaler'] == s.scaler
        assert cp['validation_exposure_hash'] == s.validation_context.exposure_hash()
        s.model.load_state_dict(cp['model_state'], strict=True)
        s.model.eval()
        seed_process(s.seed_contract['derived']['training_runtime'])
        ctx = s.validation_context
        assert s.validation_loader.dataset.context.phase == 'validation'
        params = sum(p.numel() for p in s.model.parameters())
        trainable = sum(p.numel() for p in s.model.parameters() if p.requires_grad)
        times, shape = [], None
        torch.cuda.reset_peak_memory_stats(device)
        for i, b in enumerate(s.validation_loader):
            if i >= WARMUP + TIMED:
                break
            b = vr.to_device(b, device)
            if b['target'].shape[0] != BATCH:
                continue
            teacher = vr.build_model_teacher_mask(s, b, cp['epoch'], False)
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            with torch.inference_mode():
                pred = vr._forward(s, b, ctx, cp['epoch'], teacher)['pred']
            torch.cuda.synchronize(device)
            if i >= WARMUP:
                times.append((time.perf_counter() - t) * 1e3)
            shape = tuple(pred.shape)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    q = statistics.quantiles(times, n=4)
    return dict(model=model, parameters=params, trainable_parameters=trainable,
                checkpoint_parameter_count=cp.get('parameter_count'), batch_size=BATCH, output_shape=shape,
                timed_batches=len(times), latency_ms_median=statistics.median(times), latency_ms_q1=q[0],
                latency_ms_q3=q[2], peak_memory_mib=peak, checkpoint=r['checkpoint'],
                checkpoint_sha256=sha(ROOT / r['checkpoint']), epoch=int(cp['epoch']))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        print(json.dumps(measure(sys.argv[1])))
        return
    dest = OUT / f'{DS}.json'
    assert not dest.exists(), dest
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for m in ('ALFNet', 'MoGERNN', 'Merlin', 'STBP', 'STGCN'):  # one process per model: clean peak memory
        out = subprocess.run([sys.executable, str(Path(__file__).resolve()), m], cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1]['model'], rows[-1]['parameters'], round(rows[-1]['latency_ms_median'], 2),
              round(rows[-1]['peak_memory_mib'], 1), flush=True)
    write(dest, dict(dataset=DS, split='validation', test_accessed=False, seed=32, gpu_at_start=gpu,
                     note='shared GPU: latencies are indicative', rows=rows))
    with (OUT / f'{DS}.csv').open('w', newline='') as f:
        keys = ['model', 'parameters', 'latency_ms_median', 'latency_ms_q1', 'latency_ms_q3', 'peak_memory_mib']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)


if __name__ == '__main__':
    main()
