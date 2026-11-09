# GPU-Accelerated Multivariate Forecasting

**A reproducible NVIDIA A40 study of forecasting accuracy, GPU memory use, and training throughput.**

This project implements and compares four deep-learning architectures for multivariate time-series forecasting: **LSTM, GRU, TCN, and Transformer**. Beyond model accuracy, it investigates what happens when a PyTorch training pipeline uses automatic mixed precision (AMP), pinned host memory, nonblocking host-to-device transfers, CUDA prefetching, and `torch.compile`.

**Study status: complete.** All **60 official A40 training runs** finished successfully, and the saved configurations, metrics, checkpoints, and epoch histories passed an offline integrity audit. The findings are reported as measured, including a substantial throughput regression in the more heavily optimized configurations.

**[Read the final research report](reports/phase19_final_report.md)** · **[Explore the results tables](analysis/phase18/tables/)** · **[View the frozen experiment manifest](experiments/official_v1/manifest.json)**

## Results at a glance

| Measure | Verified outcome |
|---|---|
| Experimental design | **4 models × 3 seeds × 5 CUDA configurations = 60 runs** |
| Training | 20 epochs per run; 1,200 epoch records |
| Hardware | NVIDIA A40 (48 GB), PyTorch 2.8.0+cu128, CUDA 12.8 |
| Best mean test RMSE | **2.3791 ± 0.0142**, Transformer C4 |
| Versus best recurrent model | **13.96% lower mean RMSE** than GRU C0 (descriptive comparison) |
| C0 → C1 peak allocated GPU memory | **42.5%–53.9% lower** across the four models |
| C2 versus C1 throughput | **0.241×–0.310×** across models; cause not yet isolated |
| Data integrity | 60 checkpoints and all 48 adjacent configuration comparisons verified |

### Forecasting accuracy

Mean ± sample standard deviation of **test RMSE** across seeds **17, 42, and 101**, evaluated after inverse-transforming predictions to the original data scale. Lower is better.

| Architecture | C0: FP32 | C4: full execution stack | Best configuration |
|---|---:|---:|---|
| LSTM | **2.8717 ± 0.1189** | 2.9388 ± 0.1521 | C0 |
| GRU | **2.7651 ± 0.0884** | 2.7665 ± 0.0907 | C0 |
| TCN | 2.4684 ± 0.0441 | **2.4460 ± 0.0222** | C4 |
| Transformer | 2.4281 ± 0.0359 | **2.3791 ± 0.0142** | C4 |

The **Transformer C4** configuration achieved the lowest mean test RMSE in the study. Its 13.96% advantage over the best recurrent configuration is a **descriptive result**, not a statistical-significance claim; each configuration was evaluated using only three seeds. Model/configuration selection based on test results should be viewed as exploratory.

### Systems performance: memory savings, but no end-to-end speedup

The table reports the geometric mean of **matched-seed training-throughput ratios** relative to C0, computed using epochs **2–20** (higher is better). Memory reductions compare C1 with C0, using peak **PyTorch allocated** GPU memory, not total device occupancy.

| Architecture | C1 / C0 throughput | C4 / C0 throughput | C0 → C1 allocated-memory reduction |
|---|---:|---:|---:|
| LSTM | 0.827× | 0.205× | 47.3% |
| GRU | 0.870× | 0.204× | 47.3% |
| TCN | 0.792× | 0.276× | 42.5% |
| Transformer | 0.887× | 0.312× | 53.9% |

**Key engineering finding:** Mixed precision reduced allocated GPU memory substantially, but the tested configurations did **not** improve end-to-end training throughput. In particular, the C1 → C2 transition was associated with a major slowdown across all architectures. The integrity audit verified that only the intended pinned-memory and nonblocking-transfer flags changed at this transition. Because those two settings changed together, the independent cause of the slowdown **has not been established**.

See the [full performance analysis and limitations](reports/phase19_final_report.md) and [paired comparisons](analysis/phase18/tables/official_v1_paired_vs_c0.csv).

## Experimental design

**Data:** [ETTm1](https://github.com/zhouhaoyi/ETDataset), a 15-minute electricity-transformer series with seven numeric features (six load variables and oil temperature). The forecasting task predicts all seven variables.

| Setting | Official study value |
|---|---|
| Input window | 96 time steps (24 hours) |
| Forecast horizon | 24 time steps (6 hours) |
| Split | Chronological 70% train / 10% validation / 20% test |
| Preprocessing | Standardization fitted only on training data |
| Batch size / DataLoader workers | 256 / 0 |
| Seeds | 17, 42, 101 |
| Training | 20 epochs, AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping 1.0 |
| Evaluation | MSE, MAE, RMSE on the original feature scale |

The five execution configurations are **cumulative**:

| Config | AMP | Pinned memory + nonblocking H2D | CUDA prefetch stream | `torch.compile` |
|---|:---:|:---:|:---:|:---:|
| **C0**: FP32 baseline | No | No | No | No |
| **C1** | Yes | No | No | No |
| **C2** | Yes | Yes | No | No |
| **C3** | Yes | Yes | Yes | No |
| **C4** | Yes | Yes | Yes | Yes |

The 60 resolved YAML configurations and SHA-256 hashes are preserved in [`experiments/official_v1/`](experiments/official_v1/). The official runner checks the frozen plan and prevents incomplete attempts from being silently overwritten.

## What is implemented

- **Forecasting models:** LSTM, GRU, temporal convolutional network (TCN), and Transformer with a shared training/evaluation interface.
- **Leakage-safe data processing:** chronological splitting, training-only standardization, and sliding input/forecast windows.
- **GPU execution paths:** FP32, AMP, pinned memory, nonblocking copies, dedicated CUDA prefetch stream, and model compilation.
- **Experiment controls:** fixed seeds, resolved configurations, manifest hashes, run-status tracking, checkpoints, and per-epoch telemetry.
- **Container environment:** CUDA 12.8 development image and a recorded 40-package dependency environment for Linux AMD64.
- **Custom CUDA work:** a forward-only fused LayerNorm + GELU operator, compiled and numerically validated on A40 **separately** from the official forecasting training graph. Its operator-level benchmarks are **not** end-to-end training speedup claims.

## Getting started

These commands install the project and run the basic validation checks. A local CPU or Apple Silicon system can run code correctness checks; reproducing the **official A40 systems benchmarks** requires compatible NVIDIA CUDA hardware.

```bash
# Clone the repository
git clone https://github.com/Padmore-Nana-Prempeh/gpu-multivariate-forecasting.git
cd gpu-multivariate-forecasting

# Create an environment (Python 3.12 recommended for matching the study)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Download ETTm1 and run the validation suite
bash scripts/download_ett.sh
python -m pytest -q
python scripts/verify_matrix.py

# Inspect all 60 frozen configurations; SAFE: dry run, no training
python scripts/run_official_matrix.py
```

**Single-model example:** `python -m gpuforecast.train --config configs/transformer_baseline.yaml`. This is a training run, not a quick validation command; it can be slow without a GPU.

### Reproducing the official campaign

The original experiments used an **NVIDIA A40**, a separate persistent results volume, the frozen manifest, and the validated CUDA/PyTorch environment. The official runner requires **both** an explicit `--execute` flag and `GPUFORECAST_OFFICIAL_RUNS=YES`, and refuses execution if an A40 is not present. A complete run incurs GPU costs.

```bash
# Only on a compatible, deliberately provisioned A40 environment:
GPUFORECAST_OFFICIAL_RUNS=YES python scripts/run_official_matrix.py --execute
```

To match the original environment more closely, see [`Dockerfile`](Dockerfile), [`requirements/gpu-resolved.lock`](requirements/gpu-resolved.lock), and [`scripts/verify_gpu_environment.py`](scripts/verify_gpu_environment.py). The official image digest and run provenance are captured in the archived experiment records. The GitHub Container Registry package may require access if it is still private.

### Optional profiling and CUDA-operator experiments

These tools are **separate from the frozen official results** and require an NVIDIA GPU plus the necessary profiling/compiler tools:

```bash
# Nsight Systems profiling (requires `nsys`)
bash scripts/profile_nsys.sh

# Forward-only fused LayerNorm + GELU microbenchmark
cd cuda/fused_ln_gelu
python setup.py build_ext --inplace
python benchmark.py --rows 24576 --hidden 128 --dtype fp32
```

The fused operator is **not** an autograd-enabled replacement for the training model. Its standalone benchmark results must not be presented as official training speedups.

### Reproducing the analysis

**Available directly in GitHub (no GPU required):**

- [60-run summary](analysis/phase18/tables/official_v1_full_runs.csv)
- [20 model/configuration summaries](analysis/phase18/tables/official_v1_model_config_summary.csv)
- [48 paired C0 comparisons](analysis/phase18/tables/official_v1_paired_vs_c0.csv)
- [1,200 epoch records](analysis/phase18/tables/official_v1_epoch_history.csv)
- [Offline audit findings](analysis/phase18/phase19_integrity_audit.txt)
- [Final report with claim-level qualifications](reports/phase19_final_report.md)

The raw `result.json` files and `best.pt` checkpoints are **not** committed because of their size. For users who possess the verified experiment archive, extract its `results/official_v1` directory under `analysis/phase18/`, then run:

```bash
python3 analysis/phase18/phase19_integrity_audit.py
python3 analysis/phase18/phase19_full_analysis.py
```

These scripts operate on the archived experiment records; **the second command does not work from a clean clone alone** because the raw results are intentionally excluded. The published CSV tables are available to inspect and analyze without that archive.

## Repository guide

```text
src/gpuforecast/               Models, data preparation, training, metrics, GPU prefetch
configs/                       Baseline model and local experiment configurations
experiments/official_v1/       Frozen 60-run manifest and per-run YAML configs
scripts/                       Verification, data download, and official safe runner
cuda/                          Custom CUDA extension experiments
Dockerfile                     Linux AMD64 CUDA environment
requirements/                  Resolved GPU software dependencies
analysis/phase18/              Offline analysis scripts, summary tables, integrity audit
reports/phase19_final_report.md  Completed study, results, caveats, claim ledger
```

## Interpretation and next questions

This is a **completed and audited experimental study**, not a claim that every CUDA optimization improved performance. It establishes:

1. The Transformer had the best **observed mean accuracy** for this ETTm1 task and study design.
2. AMP produced substantial **peak allocated-memory savings** across all architectures.
3. The combined C2 host-to-device execution changes coincided with a large **throughput regression** that still needs isolated profiling to explain.

The next engineering investigation, if undertaken, would profile host-side batch preparation, pinning, synchronization and transfer behavior independently, and inspect compiler traces. That work would be a **new diagnostic study**, not a retroactive change to the frozen 60 runs.

**Acknowledgments:** Thanks to Dr. Fredrick Aviyor and Patrick Tabiri for their guidance and encouragement during the project journey.
