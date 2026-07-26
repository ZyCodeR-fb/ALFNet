import sys, io, time, contextlib, torch, numpy as np; sys.path.insert(0,'.')
from args import load_args_from_config
from data_provider.data_factory import data_provider
from model.regist_model import load_model_class
dev = "cuda:0"
print(f"device={dev} torch={torch.__version__}", flush=True)
for cfg, name in (("./configs/ALFNet/rf_l3s.yaml","ALFNet"), ("./configs/MoGERNN/METRLA_lcbt_v3.yaml","MoGERNN")):
    t_build = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        a = load_args_from_config(cfg, overrides={"device": dev})
    a.lcbt_manifest_root = "./_smoke/manifests"; a.ples_manifest_root = "./_smoke/manifests"
    a.num_workers = 0; a.lcbt_overwrite_manifest = True
    with contextlib.redirect_stdout(io.StringIO()):
        tl = data_provider(a, "train"); vl = data_provider(a, "val")
        m = load_model_class(a.model)(a).float().to(dev)
    build = time.perf_counter()-t_build
    nparam = sum(p.numel() for p in m.parameters() if p.requires_grad)
    b = [t.to(dev) for t in next(iter(tl))]
    x_enc, x_dec, tmk, dmk, mask = b[0].float(), b[1].float(), b[2].float(), b[3].float(), b[7]
    adj, _ = tl.dataset.get_adj_and_pos_mark()
    adj = torch.as_tensor(adj, dtype=torch.float32, device=dev)
    x_enc = torch.nan_to_num(x_enc, nan=0.0); y = torch.nan_to_num(x_dec[:, -a.pred_len:], nan=0.0)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.perf_counter()
    out = m(adj, x_enc, tmk if a.t_mark else None, None, y, dmk if a.t_mark else None, 0)
    pred = out["pred"] if isinstance(out, dict) else out
    sel = (mask[:,None,:,None] > 0.5).expand_as(y) & torch.isfinite(y)
    loss = torch.nn.functional.mse_loss(pred[sel], y[sel]); loss.backward()
    torch.cuda.synchronize(); dt = time.perf_counter()-t0
    g = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
    print(f"\n{name}: params={nparam:,} loader+model build={build:.1f}s")
    print(f"  batch={tuple(x_enc.shape)} loss={loss.item():.6f} finite={bool(torch.isfinite(loss))}")
    print(f"  grads={len(g)} all_finite={all(np.isfinite(g))} norm[min={min(g):.2e} max={max(g):.2e}]")
    print(f"  fwd+bwd={dt*1000:.0f} ms  peak_mem={torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    print(f"  train batches={len(tl)} val batches={len(vl)}  -> 单 epoch(仅训练) ≈ {dt*len(tl)/60:.1f} min", flush=True)
    del m; torch.cuda.empty_cache()
