# ALFNet configurations

Three tiers. Only the first is the paper's model.

## Mainline — the frozen deliverable

| file | dataset | notes |
|---|---|---|
| `alfnet.yaml` | METR-LA | **the paper's entry point** |
| `gs.yaml` | METR-LA | same experiment as `alfnet.yaml` by content hash |
| `gs_bay.yaml` | PEMS-BAY | twin |
| `gs_p8.yaml` | PEMS08 | twin |

`alfnet.yaml` and `gs.yaml` are content-identical and therefore hash to the same experiment, so existing
`gs` checkpoints load under the paper's name. `gs.yaml` is kept for that compatibility and because
`scripts/build_router_configs.py` generates it, alongside the twins that
`tests/test_pemsbay_configs.py` pins.

In the paper this is **ALFNet**, and the mechanism is **lifecycle-conditioned soft heterogeneous
propagation aggregation**. It is not sparse routing and it does not claim expert specialisation.

## Routing ablations — kept, for the ablation table

| file | router | why it is not the mainline |
|---|---|---|
| `fg1{,_bay,_p8}.yaml` | `noisy_topk` | the historical mainline. At `moge_top_k: 1` the router receives **exactly zero** forecast gradient, so its advertised mechanism behaves as expert dropout |
| `g1{,_bay,_p8}.yaml` | `st_top1` | hard top-1 with a straight-through gradient; unstable across datasets |
| `gm{,_bay,_p8}.yaml` | `static` | the control arm — one learned mixture, no lifecycle conditioning. Isolates whether conditioning earns its keep |

All four router variants share an identical parameter count except `gm`, which has no `w_gate`/`w_noise`,
so these are pure ablations. Do not add a fifth: the router question is closed.

## Retired candidates — negative evidence only

`a b c d` and their combinations (`ac ad bc bd cd acd bcd abc abcd`), the frozen label-residual family
`rt rd rtg`, the joint form `jr`, the training-objective switches `tq tw wg`, and `fg1r` / `rf_l3s`.

None of these improved the mainline on validation, and none may be quoted as a result — every one predates
the strict-inductive, MoGERNN-adapter and LCBT v3.2 corrections.

**They are still here on purpose.** `CLAUDE.md` says to delete a dead candidate's YAML, code branch and
tests together once the architecture is frozen, and the architecture now is. The deletion has not been done
because it is not free:

* the recipe switches these files exercise are still live code in `model/ALFNet/ALFNet.py` and
  `layers/protected_residual.py`, and their only coverage is `tests/test_abcd_switches.py` and
  `tests/test_protected_residual.py` — 56 tests over ~1,200 lines. Deleting the YAMLs alone would leave
  live code with no tests, which is worse than leaving the YAMLs;
* deleting the code as well touches module construction. Optional modules are built inside `_isolated_rng`
  with a fixed per-module offset precisely so that enabling one switch cannot shift another module's
  initialisation; removing them should therefore be safe for `gs`, which constructs none of them — but
  "should be" is not good enough for a frozen model, and verifying it means a parameter-by-parameter
  comparison before and after.

So this is a deliberate deferral, not an oversight. Do it as its own change — YAMLs, code and tests
together, with a before/after check that every `gs` parameter is bit-identical — and not while the 18-run
matrix is in flight.
