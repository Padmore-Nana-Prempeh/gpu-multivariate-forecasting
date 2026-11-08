# Phase 19: Final analysis of the frozen official A40 experiment campaign

**Status:** 60 completed official training runs, 0 incomplete. **Hardware:** NVIDIA A40; PyTorch 2.8.0+cu128, CUDA runtime 12.8. **Dataset:** ETTm1, 7 features; lookback 96; forecast horizon 24. **Design:** four architectures × seeds 17, 42, 101 × five CUDA execution configurations, 20 training epochs each. This report uses the archived `result.json` metrics and epoch histories as transformed into the accompanying CSV files.

## Reproducibility and scope

* C0: baseline FP32 execution; C1: AMP; C2: AMP + pinned-memory DataLoader + nonblocking H2D transfers; C3: adds CUDA prefetch stream; C4: adds `torch.compile`.
* Official frozen manifest and 60 run identities remain unchanged. Data preprocessing, batch size 256, optimizer, and train/validation/test splits were held constant.
* 60 run-level result files with nonempty checkpoints were validated by the frozen runner and again by the local integrity audit; the local backup SHA-256 was verified after transfer from RunPod: `6ff19b88aa870b0103e7f9c5f1cf16d894e87da1992edbc9f24bbbbafec7237d`.
* From the uploaded CSVs: 60 complete run rows, 20 model/config summary groups, 48 seed-paired comparison rows, and 1,200 epoch rows (20 per run). No missing values in the 60-row complete run table.
* End-to-end training timing includes DataLoader and training-loop work, and epoch timing synchronizes CUDA before and after the epoch. Runner wall time additionally includes startup, validation, checkpointing, and testing. These are not isolated kernel times.

## Forecasting accuracy

Mean ± sample standard deviation of test RMSE on inverse-transformed targets across three seeds; lower is better.

| Model | C0 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| LSTM | 2.8717 ± 0.1189 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 |
| GRU | 2.7651 ± 0.0884 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 |
| TCN | 2.4684 ± 0.0441 | 2.4527 ± 0.0480 | 2.4527 ± 0.0480 | 2.4527 ± 0.0480 | 2.4460 ± 0.0222 |
| Transformer | 2.4281 ± 0.0359 | 2.3958 ± 0.0536 | 2.3958 ± 0.0536 | 2.3958 ± 0.0536 | 2.3791 ± 0.0142 |

**Descriptive best by model:** LSTM C0; GRU C0; TCN C4; Transformer C4. The lowest mean RMSE is Transformer C4 (2.3791 ± 0.0142). Relative to the best recurrent mean, GRU C0 (2.7651 ± 0.0884), this is a 13.96% smaller mean RMSE. Every seed individually favors Transformer C4 over GRU C0; three seeds are too few to support a strong significance claim. Model selection comparisons should be presented as descriptive and exploratory where test outcomes informed the final choice.

For all 12 matched model–seed pairs, C1, C2 and C3 have identical test metrics and per-epoch training losses/validation RMSE as exported in the records. C4 differs for TCN and Transformer but not LSTM or GRU. Identical outcomes support numerical consistency of the compared execution configurations; they do **not** prove a particular transfer optimization was active or that `torch.compile` fully captured the model.

## Steady-state throughput: matched-seed ratios

Geometric mean of the three seed-matched ratios of epochs-2–20 samples/second against C0; larger is faster. C0 is exactly 1.000× by definition.

| Model | C0 | C1 | C2 | C3 | C4 | C2/C1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 1.000× | 0.827× | 0.217× | 0.211× | 0.205× | 0.262× |
| GRU | 1.000× | 0.870× | 0.210× | 0.202× | 0.204× | 0.241× |
| TCN | 1.000× | 0.792× | 0.246× | 0.254× | 0.276× | 0.310× |
| Transformer | 1.000× | 0.887× | 0.270× | 0.270× | 0.312× | 0.304× |

**Observed:** C1 lowers throughput relative to C0 by approximately 11%–21% (model-dependent). At the next transition, C2 retains only 24%–31% of C1 throughput in the three-seed geometric mean (roughly 3.2×–4.2× slower than C1). C3 and C4 remain materially below C0. This regression is consistent across all twelve model–seed combinations. The magnitude warrants investigation but, without isolated profiling, does not identify the causal mechanism.

## Peak PyTorch memory allocated

Mean of three seeds; allocated GPU memory only, not total device occupancy or reserved allocator memory.

| Model | C0 allocated MB | C1 allocated MB | C1 reduction vs C0 | C4 allocated MB | C4 reduction vs C0 |
|---|---:|---:|---:|---:|---:|
| LSTM | 462.2 | 243.4 | 47.3% | 244.2 | 47.2% |
| GRU | 448.9 | 236.4 | 47.3% | 237.2 | 47.2% |
| TCN | 266.6 | 153.3 | 42.5% | 156.0 | 41.5% |
| Transformer | 576.9 | 265.9 | 53.9% | 229.8 | 60.2% |

Memory reductions are descriptive measurements; PyTorch peak reserved memory, CUDA context and framework allocations, and total GPU occupancy are distinct quantities. The memory savings cannot be used to infer throughput gains.

## Configuration integrity and the C1 → C2 throughput cliff

Direct comparison of the frozen published `transformer_seed17_c1.yaml` and `transformer_seed17_c2.yaml` confirms the following: `batch_size=256` in both, `num_workers=0` in both, and all model/training settings identical **except** `data.pin_memory: false → true` and `train.non_blocking_h2d: false → true`. The DataLoader implementation sets `persistent_workers=(num_workers > 0)` and only adds `prefetch_factor=2` when workers are positive; thus persistent workers and worker prefetch remain inactive in both. The completed offline audit confirmed `batch_size=256`, `num_workers=0`, and `epochs=20` for all 60 saved run configurations, with no unintended changes in any of the 48 adjacent comparisons. A workers drop from four to zero did **not** occur in the official campaign.

The configuration sets change two transfer-related factors together, so this experiment **cannot isolate the independent effect of pinning from nonblocking transfers**. The size of the slowdown is surprising but does not by itself demonstrate a configuration bug. Possible contributors include CPU-side batch collation/pinning overhead, synchronization, transfer implementation, or interactions with the zero-worker, main-process data-loading path. These remain hypotheses, not established causes. Do not describe the measured C2/C3/C4 performance as the inherent cost of pinned memory alone.

**Completed offline integrity audit:** `analysis/phase18/phase19_integrity_audit.py` passed against all 60 archived runs. It verified 60 nonempty checkpoints, 1,200 epoch records, all model/seed/config combinations, and 48 adjacent configuration comparisons. C1→C2 changed only `data.pin_memory` and `train.non_blocking_h2d`. This rules out an unintended configuration drift as the explanation for the observed comparison, but does not establish the precise performance mechanism.

## Accuracy and compiler caveats

* C1/C2/C3 equality on recorded accuracy is a useful numerical regression check, but not sufficient to prove correct `pin_memory` performance characteristics.
* C4 accuracy differences for TCN/Transformer are consistent with compilation affecting floating-point execution, but do not uniquely prove altered reduction order. Conversely, identical recurrent-model results do not prove a graph break. The frozen code calls `torch.compile(model)` at default settings; collect compiler traces if that question becomes material.
* The frozen `train.py` environment metadata does **not** log TF32 backend flags, and it does not explicitly set them. PyTorch 2.8 generally defaults to TF32 **off** for CUDA matmul but **on** for cuDNN operations. Do not claim TF32 already accelerated the entire FP32 baseline without runtime evidence. Future benchmark runs should record both flags, alongside any precision-policy changes.
* The custom forward-only fused LayerNorm+GELU extension was compiled and numerically validated independently on A40; it is not integrated into the official training graph shown in `src/gpuforecast/train.py`. Its operator-level speedup cannot be transferred to this end-to-end report.

## Claim ledger

| Claim | Status | Evidence / qualification |
|---|---|---|
| 60 official experiments completed | **Verified** | All run states completed; archived results and backup validated |
| Transformer C4 lowest mean RMSE | **Supported descriptively** | 2.3791 across three seeds; no significance claim |
| AMP reduces PyTorch allocated GPU memory | **Supported** | C0→C1 reductions of 42.5%–53.9% across models |
| C2 path is substantially slower than C1 | **Supported observationally** | Three-seed matched throughput C2/C1 ≈ 0.241×–0.310× |
| Worker-count change explains slowdown | **Contradicted by completed audit** | All 60 configs set `num_workers=0`; all 48 adjacent comparisons passed |
| Pinned memory by itself causes 4× slowdown | **Not established** | Pinning and nonblocking copies enabled together; no isolated profile |
| TF32 explains AMP regression | **Not established** | TF32 flags absent from recorded environment |
| torch.compile fell back on LSTM/GRU | **Not established** | Identical accuracy is insufficient to diagnose graph breaks |
| Fused CUDA extension speeds official models | **Not measured** | Extension is a standalone forward-operator benchmark |

## Finalization workflow

1. Preserve the frozen Phase 18 results and verified archive unchanged. Leave the RunPod A40 off.
2. **Completed:** The offline configuration-integrity audit passed for all 60 records and 48 adjacent comparisons; the audit log is part of the finalized reproducibility evidence.
3. Commit the analysis script(s), compact CSV tables, audit findings, and report to GitHub, using explicit paths. Exclude archive backups, `best.pt`, the extracted `analysis/phase18/results/` tree, and any secrets.
4. Publish the measured speed-memory-accuracy tradeoffs as observations, not as evidence that pinning alone caused the slowdown. A targeted C1/C2 profiling investigation is useful future work, **not required to complete documentation of the frozen 60-run campaign**.
5. Keep the $7/month RunPod network volume until you deliberately decide whether the verified local backup plus a second independent copy are sufficient; do not leave unused storage billing indefinitely.

## Provenance

* Local source tables: `official_v1_full_runs.csv`, `official_v1_epoch_history.csv`, `official_v1_model_config_summary.csv`, `official_v1_paired_vs_c0.csv`.
* Completed local configuration audit: `analysis/phase18/phase19_integrity_audit.py` and `analysis/phase18/phase19_integrity_audit.txt` (60 records, 60 checkpoints, 1,200 epoch records, 48 adjacent comparisons passed).
* Frozen experiment configs: [C1](https://github.com/Padmore-Nana-Prempeh/gpu-multivariate-forecasting/blob/ea2c8f9/experiments/official_v1/configs/transformer_seed17_c1.yaml) and [C2](https://github.com/Padmore-Nana-Prempeh/gpu-multivariate-forecasting/blob/ea2c8f9/experiments/official_v1/configs/transformer_seed17_c2.yaml).
* DataLoader implementation: [data.py](https://github.com/Padmore-Nana-Prempeh/gpu-multivariate-forecasting/blob/ea2c8f9/src/gpuforecast/data.py); [training loop](https://github.com/Padmore-Nana-Prempeh/gpu-multivariate-forecasting/blob/ea2c8f9/src/gpuforecast/train.py).
* PyTorch precision behavior: [TF32 CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html); [compiler graph breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html).
