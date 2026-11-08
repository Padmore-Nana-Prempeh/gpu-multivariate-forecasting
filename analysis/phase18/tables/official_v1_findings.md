# Phase 19: Official A40 experiment results (local audit)

Four models × three random seeds × five CUDA configurations = 60 official 20-epoch runs.
This report is generated from the archived `result.json` epoch histories and `runner_status.json` timestamps, not from model checkpoints.

## Test forecasting accuracy

Mean ± sample SD of test RMSE across three seeds (lower is better):

| Model | C0 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| LSTM | 2.8717 ± 0.1189 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 | 2.9388 ± 0.1521 |
| GRU | 2.7651 ± 0.0884 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 | 2.7665 ± 0.0907 |
| TCN | 2.4684 ± 0.0441 | 2.4527 ± 0.0480 | 2.4527 ± 0.0480 | 2.4527 ± 0.0480 | 2.4460 ± 0.0222 |
| Transformer | 2.4281 ± 0.0359 | 2.3958 ± 0.0536 | 2.3958 ± 0.0536 | 2.3958 ± 0.0536 | 2.3791 ± 0.0142 |

## Steady-state training throughput relative to C0

Geometric mean of matched-seed throughput ratios (Ck / C0), using epochs 2–20. Higher is better.

| Model | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| LSTM | 0.827× | 0.217× | 0.211× | 0.205× |
| GRU | 0.870× | 0.210× | 0.202× | 0.204× |
| TCN | 0.792× | 0.246× | 0.254× | 0.276× |
| Transformer | 0.887× | 0.270× | 0.270× | 0.312× |

## Interpretation and limitations

- C0 = FP32; C1 = AMP; C2 adds pinned/nonblocking host-to-device transfer; C3 adds CUDA prefetch; C4 adds `torch.compile`.
- Accuracy is evaluated on inverse-transformed test targets. Smaller RMSE is better.
- The throughput ratio is paired by **model and seed**. It is not evidence of statistical significance with only three seeds.
- Epoch 1 includes potential initialization/JIT compilation effects. Both all-epoch and epochs-2–20 metrics are retained in the CSVs.
- `runner_wall_seconds` includes process startup, training, validation, checkpointing, testing, and compilation; it is not GPU kernel execution time alone.
- GPU memory is PyTorch's peak allocated/reserved memory recorded during training epochs, not total hardware memory occupancy.
- These end-to-end numbers do **not** benchmark the standalone forward-only fused LayerNorm + GELU CUDA extension unless it is explicitly integrated into the training graph.
- Hardware was an NVIDIA A40; per-run environment and image digest should be checked before publication.
- This is a single-model/seed/config run per cell. Running additional replications requires a separate protocol and is not part of this frozen campaign.

## Generated data files

- `official_v1_full_runs.csv`: 60 runs with accuracy, total/steady training timing, throughput, peak GPU memory, and total wall time.
- `official_v1_epoch_history.csv`: 1,200 per-epoch records.
- `official_v1_model_config_summary.csv`: 20 model/config group means and sample SDs (three seeds each).
- `official_v1_paired_vs_c0.csv`: 48 matched-seed comparisons against the FP32 baseline.
