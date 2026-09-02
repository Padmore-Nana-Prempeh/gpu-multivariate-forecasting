# GPU-Accelerated Deep Learning for Multivariate Forecasting

A reproducible systems + modeling project that benchmarks **LSTM, GRU, TCN, and Transformer** forecasters under controlled GPU configurations, then profiles and optimizes the training path with AMP, pinned memory, asynchronous H2D copies, CUDA streams, Nsight Systems, and a custom fused CUDA LayerNorm+GELU kernel.

> **Integrity rule:** performance numbers are not hard-coded project claims. `results/` is generated from actual runs on the target GPU. Resume bullets should be updated only after the benchmark report confirms them.

## Benchmark question

How much forecasting accuracy and systems performance do we gain from model choice and GPU optimization when the dataset, splits, windows, seeds, optimizer, and metrics are controlled?

## Dataset

The starter benchmark uses **ETTm1**, a 15-minute electricity-transformer time-series dataset with six load variables plus oil temperature (`OT`). The loader selects all numeric columns and performs train-only standardization to avoid leakage.

Default task:
- Input window: 96 timesteps (24 hours)
- Forecast horizon: 24 timesteps (6 hours)
- Mode: multivariate -> multivariate
- Split: 70% train / 10% validation / 20% test, chronological
- Metrics: RMSE, MAE, MSE on the original feature scale

## Experiment matrix: exactly 60 controlled training runs

Four architectures × three seeds × five execution configurations:

| Configuration | FP32 | AMP | Pinned memory | CUDA prefetch stream | `torch.compile` |
|---|---:|---:|---:|---:|---:|
| fp32 | ✓ | | | | |
| amp | | ✓ | | | |
| amp_pin | | ✓ | ✓ | | |
| amp_pin_prefetch | | ✓ | ✓ | ✓ | |
| amp_pin_prefetch_compile | | ✓ | ✓ | ✓ | ✓ |

Architectures: LSTM, GRU, TCN, Transformer. Seeds: 17, 42, 101.

## Repository layout

```text
configs/base.yaml               core experiment settings
scripts/download_ett.sh         reproducible dataset download
scripts/run_matrix.py           60-run controlled experiment matrix
scripts/profile_nsys.sh         Nsight Systems capture
src/gpuforecast/data.py         time split, train-only scaling, window datasets
src/gpuforecast/models.py       LSTM / GRU / TCN / Transformer
src/gpuforecast/gpu.py          dedicated CUDA prefetch stream
src/gpuforecast/train.py        AMP training + timing + memory + metrics
cuda/fused_ln_gelu/             custom CUDA forward microbenchmark
results/                        generated checkpoints, metrics, profiler output
```

## 1. Local setup

Create the environment and install PyTorch for your machine first. For an NVIDIA system, use the wheel matching the installed CUDA environment from the official PyTorch install selector. Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
./scripts/download_ett.sh
pytest -q
```

CPU or Apple Silicon can run model correctness tests and small training runs. **CUDA optimization, Nsight, and the custom extension require an NVIDIA CUDA GPU.**

## 2. First baseline

```bash
python -m gpuforecast.train \
  --model lstm \
  --seed 42 \
  --amp off \
  --pin-memory off \
  --prefetch-stream off \
  --compile off
```

Then run the Transformer under the exact same configuration:

```bash
python -m gpuforecast.train \
  --model transformer \
  --seed 42 \
  --amp off \
  --pin-memory off \
  --prefetch-stream off \
  --compile off
```

## 3. GPU optimized run

```bash
python -m gpuforecast.train \
  --model transformer \
  --seed 42 \
  --amp on \
  --pin-memory on \
  --prefetch-stream on \
  --compile off
```

Each epoch records:
- training loss
- seconds / epoch
- samples / second
- CUDA peak allocated memory
- validation RMSE / MAE / MSE

The final JSON also records test metrics, PyTorch version, CUDA version, and run configuration in its filename.

## 4. Run all 60 experiments

```bash
python scripts/run_matrix.py
```

Do this only after a short smoke test, because the full matrix is intentionally expensive.

## 5. Nsight Systems

On a Linux/Windows NVIDIA environment with `nsys` installed:

```bash
./scripts/profile_nsys.sh
```

Inspect:
- blank GPU regions / starvation
- H2D memcpy placement
- copy/compute overlap across CUDA streams
- synchronization calls
- expensive kernels
- CPU DataLoader gaps

The purpose is to demonstrate *why* epoch time changes, not merely report a stopwatch number.

## 6. Custom CUDA LayerNorm + GELU microbenchmark

This is deliberately separated from the training path until numerical correctness and speed are demonstrated.

Build:

```bash
cd cuda/fused_ln_gelu
python setup.py build_ext --inplace
```

Benchmark:

```bash
python benchmark.py --rows 24576 --hidden 128 --dtype fp32
python benchmark.py --rows 24576 --hidden 128 --dtype fp16
```

The script prints:
- maximum absolute error vs `torch.nn.functional.layer_norm` + GELU
- eager latency
- fused latency
- measured speedup

Only if the numerical error and speed are acceptable should the fused operator be wired into the forecasting Transformer. The current extension is a **forward microbenchmark**, not yet a custom-autograd training operator.

## 7. Claims we will compute, not assume

The final analysis script should derive these directly from JSON/benchmark output:

```text
Transformer RMSE improvement vs strongest recurrent baseline = ... %
AMP peak-memory reduction vs FP32                         = ... %
AMP throughput improvement vs FP32                      = ... x
optimized epoch-time speedup vs FP32                    = ... x
fused LN+GELU microbenchmark speedup                     = ... x
max absolute fused-kernel error                         = ...
```

Those measured values become the final GitHub README table and résumé bullet.

## Next implementation stages

1. Smoke-test dataset and all four model shapes.
2. Run one-seed FP32 baseline for every architecture.
3. Validate AMP accuracy drift and memory/throughput.
4. Validate pinned-memory + non-blocking transfer behavior.
5. Profile copy/compute overlap with Nsight Systems.
6. Tune the prefetch stream only if the timeline shows transfer stalls.
7. Benchmark and validate the fused CUDA op.
8. Add backward/custom autograd only if training-level fusion is justified.
9. Aggregate all run JSON into publication-quality tables/plots.
10. Add Docker/NVIDIA Container Toolkit, CI smoke tests, and a final reproducibility report.

## 8. Aggregate benchmark results

After the experiment matrix finishes:

```bash
python scripts/summarize_results.py
```

This creates:
- `results/summary.csv` — one row per run
- `results/grouped_summary.csv` — mean/std benchmark table by model and system configuration
- `results/system_speedups.csv` — per-seed throughput, epoch-time, memory, and RMSE changes relative to FP32

This is the source of truth for the final README and résumé claims.
