# Phase 9 - CUDA FP32 Reference Benchmark

## Objective

Establish the unoptimized CUDA FP32 reference path, C0, before introducing
automatic mixed precision, pinned host memory, CUDA prefetch streams, or
torch.compile.

This benchmark is the systems-performance reference for later C1-C4
comparisons.

## Workload

- Dataset: ETTm1
- Dataset SHA-256:
  `6ce1759b1a18e3328421d5b75fadcb316c449fcd7cec32820c8dafda71986c9e`
- Lookback: 96 time steps
- Forecast horizon: 24 time steps
- Features: 7
- Model: Transformer
- Trainable parameters: 222,072
- Hidden dimension: 92
- Encoder layers: 2
- Attention heads: 4
- Batch size: 256
- Seed: 42
- Epoch budget: 20

## C0 Execution Configuration

- Precision: FP32
- AMP: disabled
- Pinned host memory: disabled
- Non-blocking H2D optimization: disabled
- CUDA prefetch stream: disabled
- torch.compile: disabled
- DataLoader workers: 0
- Device: CUDA

## Environment

- GPU: NVIDIA A40
- GPU VRAM: 48 GB class
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Python: 3.12.3
- Platform: Linux x86_64
- Git SHA recorded by experiment:
  `9387589eb56c82bab42bb85123fe7c18ea24dd8a`

## Measurement Protocol

Epoch 1 was treated as CUDA/runtime warm-up and excluded from the
steady-state systems summary.

Steady-state performance is therefore summarized over epochs 2-20.

CUDA synchronization is performed at epoch timing boundaries.

Peak CUDA allocated and reserved memory are reset/measured for the training
epoch.

Validation RMSE selects the best checkpoint. The test set is evaluated only
after reloading that checkpoint.

## Results

### Forecast quality

- Best epoch: 4
- Best validation RMSE: 2.0705508689
- Test MSE: 5.8764653206
- Test RMSE: 2.4241421824
- Test MAE: 1.3101302385

### Steady-state systems performance

- Mean epoch time: 3.712429 seconds
- Mean throughput: 13,139.781 samples/second
- Peak allocated CUDA memory: 576.900879 MB
- Peak reserved CUDA memory: 670.0 MB
- Measured steady-state epochs: 19

## Interpretation

C0 establishes the reference GPU execution path against which all later
framework optimizations are measured.

No speedup or memory-reduction claim is made from C0 alone.

Future configurations must preserve the same forecasting workload and
hardware while changing only the intended execution feature:

- C1: AMP
- C2: AMP + pinned memory + non-blocking H2D
- C3: AMP + pinned memory + CUDA prefetch stream
- C4: optimized framework path with torch.compile

The C0 values above are therefore the denominators for future throughput,
epoch-time, and GPU-memory comparisons.
