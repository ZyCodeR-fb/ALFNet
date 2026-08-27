# Comparison-model source and v4 adaptation audit

This file records what was compared before the unified-BASE plan was frozen.
Every baseline receives the same v4 batch, graph, split, masks, seeds,
checkpoint rule and evaluation arithmetic. A wrapper may reshape tensors; it
may not load private data or consume ALFNet lifecycle features.

| registered model | local core | checked source | retained method / necessary adaptation |
|---|---|---|---|
| HA | `HA/ha.py` | `references/ST-FiT/src/model/hi/hi.py` | historical-inertia copy; zero-output parameter anchor only permits the common optimiser API |
| FC-LSTM | `FCLSTM/lstm.py` | `references/ST-FiT/src/model/lstm/lstm.py` | LSTM/conv/head equations retained; source-private graph/data trimmer and device literals are replaced by `RuntimeDataContext`/runtime tensors |
| STGCN | `stgcn_n/*` | `references/ST-FiT/src/model/stgcn_n/*` | convolution core retained; all node views use the common full-N graph and source augmentation flags are disabled |
| STGODE | `stgode/*` | `references/ST-FiT/src/model/stgode/*` | ODE/TCN blocks and head retained; spatial matrix comes from the common graph; semantic matrix is topology-row cosine because source DTW would read held-out signals |
| GWNET | `GWNet/gwnet.py` | `references/GWNet/model.py` | WaveNet gated dilated blocks, diffusion GCN and adaptive support retained; trainable node-ID factors are replaced by two independent shared structural encoders |
| STID | `STID/stid.py` | `references/STID/stid/arch/stid_arch.py` | temporal embeddings, MLP encoder and head retained; trainable node-ID table is replaced by a shared structural encoder |
| ST-FiT | `STFiT/*`, `stgcn_n/*` | `references/ST-FiT/src/model`, `src/trainer/train.py` | learned graph, mixup/VAE node synthesis, alternating VAE/forecast epochs and source MAE retained; train targets are visible only during its train-time synthesis, never validation/test |
| MoGERNN | `MoGERNN/*` | `references/MoGERNN/model`, `layers` | MoGE/DCRNN architecture retained; receives the same full-N masked tensor and runtime graph, with no lifecycle context |
| SLPF | `SLPF/SLPF.py` | `references/SLPF/model/SLPF.py` | source stages and forecast blocks retained; node-ID embedding is replaced by the shared structural encoder and device-bound indexing is made runtime-safe |
| PatchSTG | `PatchSTG/patchstg.py` | `references/PatchSTG/models/model.py` | patch attention retained; spatial membership comes from the manifest-resolved layout rather than private coordinate loading |
| STBP | `STBP/stbp.py` | `references/STBP/src/model/model.py` | DLGA/frequency/encoder/head computation retained; graph/positions are injected, node-ID capacity is structural, and source MAE is used |
| MCPST-FSL | `MCPST_FSL/model.py` | `references/MCPST-FSL/models/model.py` | multi-scale, oscillator, spectral and uncertainty branches retained; flow MSE is lifecycle-masked and target-free physics/consistency regularisers remain active |

## Deliberate fairness adaptations

The following are not architecture claims and are shared wherever applicable:

- disk loaders, splitters, scalers and metrics from source repositories are not
  used; the v4 manifest/loader/evaluator is the sole authority;
- every model runs on the full static node axis, while unavailable dynamic
  values are mean-filled and masked identically;
- per-node trainable embeddings that would give held-out nodes learned identity
  are replaced by shared functions of the same topology/verified positions;
- source optimisation loss is declared in YAML (MAE for HA/FC-LSTM/STGCN/
  STGODE/GWNET/STID/ST-FiT/SLPF/PatchSTG/STBP; MSE for ALFNet/MoGERNN/
  MCPST-FSL), while early stopping and all reported metrics use the same v4
  definitions;
- target tensors never enter a model on validation/test. ST-FiT's use of train
  targets for source-defined synthetic-node training is explicit and tested.

The adapters were accepted only after a real PEMS08 one-batch CPU
forward/backward/parameter-update check for every registered model. The final
plan hashes every Python core and adapter listed here.
