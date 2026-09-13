# Phase 9: C0 FP32 CUDA Reference Benchmark

## Status

**Phase:** 9 - GPU Execution Fundamentals and FP32 Reference Benchmark  
**Execution configuration:** C0 - FP32 Reference  
**Hardware:** NVIDIA A40  
**Development seed:** 42  
**Benchmark status:** Completed  
**Role in project:** Reference/control configuration for C1-C4 systems comparisons

---

## 1. Objective

The purpose of C0 is to establish a reproducible GPU reference path before introducing mixed precision, pinned host memory, asynchronous host-to-device copies, CUDA prefetch streams, or graph compilation.

C0 is intentionally the simplest CUDA execution configuration. It is not designed to be artificially slow. Instead, it acts as the control condition against which every later execution configuration is measured.

The core question answered by C0 is:

> What are the forecasting quality, epoch time, throughput, and CUDA memory characteristics of the focal Transformer when trained using the standard FP32 CUDA path?

---

## 2. C0 Execution Configuration

The frozen C0 configuration is:

| Setting | C0 value |
|---|---|
| Precision | FP32 |
| AMP | Disabled |
| Pinned host memory | Disabled |
| Non-blocking H2D transfer | Disabled |
| CUDA prefetch stream | Disabled |
| torch.compile | Disabled |
| Device | CUDA |
| Batch size | 256 |
| Seed | 42 |

C0 therefore represents the least optimized framework execution path in the controlled systems study.

Later configurations modify one systems dimension at a time whenever possible.

---

## 3. Forecasting Workload

The benchmark uses the ETTm1 multivariate forecasting task defined by the project.

### Input contract

Each model input has shape:

`[batch_size, 96, 7]`

The 96 historical observations correspond to a 24-hour lookback window at 15-minute sampling intervals.

### Forecast contract

Each model target has shape:

`[batch_size, 24, 7]`

The model predicts the next 24 observations for all seven variables, corresponding to a six-hour forecasting horizon.

### Focal model

The C0 systems benchmark uses the Transformer forecaster.

Recorded model size:

`222,072 parameters`

---

## 4. Benchmark Environment

The frozen benchmark was executed in the following environment:

| Component | Recorded value |
|---|---|
| GPU | NVIDIA A40 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Batch size | 256 |
| Model | Transformer |
| Parameters | 222,072 |
| Seed | 42 |
| Frozen benchmark Git SHA | `25809b3b2d8fede7eec4e369dc60051c68f016ad` |

The benchmark engine was frozen before the controlled C0-C4 comparison so that later execution paths used the same timing, evaluation, checkpoint, and result schema.

---

## 5. Recorded Evidence

The frozen C0 result was written by the benchmark system to:

`results/cuda_fp32/transformer_seed42_amp0_pin0_prefetch0_compile0/result.json`

The aggregated result also appears in:

`results/summary.csv`

and:

`results/system_speedups.csv`

The raw GPU evidence from the development A40 session was preserved separately in the local evidence archive.

---

## 6. Why CUDA Timing Requires Synchronization

CUDA execution is asynchronous with respect to the Python host.

A Python statement can enqueue GPU work and regain control before the GPU has completed that work.

Therefore a naive timing pattern can under-report GPU execution time.

Conceptually:

**Python**

`launch CUDA work -> continue executing`

**GPU**

`execute queued kernels asynchronously`

If the host stops a wall-clock timer immediately after launching a kernel, the timer may stop before the actual GPU work has completed.

For this reason, the benchmark engine synchronizes CUDA at declared timing boundaries.

The conceptual timing sequence is:

`Synchronize -> start timer -> measured GPU work -> synchronize -> stop timer`

This ensures that previously queued work does not contaminate the beginning of the measurement and that measured GPU work has completed before the timer stops.

Synchronization is not inserted after every CUDA operation because excessive synchronization would unnecessarily serialize CPU and GPU execution and change the workload being measured.

---

## 7. Epoch Time, Throughput, and Latency

The project distinguishes three related performance concepts.

### Epoch time

Epoch time measures how long one complete training pass through the training dataset takes.

For C0:

`4.209343 seconds per steady-state epoch`

Lower epoch time is better when the workload is otherwise identical.

### Throughput

Throughput measures the number of training samples processed per second.

For C0:

`11,570.598 samples/second`

Higher throughput is better.

Throughput and epoch time should move in opposite directions for a fixed dataset:

- lower epoch time generally means higher throughput
- higher epoch time generally means lower throughput

### Latency

Latency measures how long one individual operation or request takes.

Operator latency becomes especially important later in the CUDA kernel microbenchmark, but the primary end-to-end training metrics for C0-C4 are epoch time and samples per second.

---

## 8. Allocated Versus Reserved CUDA Memory

PyTorch exposes multiple CUDA memory measurements.

Two important values in this project are allocated memory and reserved memory.

### Allocated CUDA memory

Allocated memory is memory currently occupied by live PyTorch tensor allocations.

The C0 peak allocated value was:

`576.901 MB`

### Reserved CUDA memory

PyTorch uses a caching allocator.

Instead of repeatedly returning every unused allocation to the CUDA driver, PyTorch can retain memory in a reserved pool for efficient reuse by future tensor allocations.

The C0 peak reserved value was:

`670.000 MB`

Therefore:

`reserved memory >= allocated memory`

does not automatically indicate a leak.

The difference can represent memory retained by the PyTorch CUDA caching allocator for reuse.

Both values are reported because they answer different questions.

Allocated memory describes active tensor allocation pressure.

Reserved memory describes the larger CUDA memory pool retained by PyTorch.

---

## 9. Warm-up and Steady-State Measurement

The benchmark separates initial execution behavior from steady-state behavior.

The first epoch is excluded from the aggregate steady-state timing.

The frozen C0 summary therefore reports:

`19 steady-state epochs`

with:

`warmup_excluded = True`

This avoids treating initial runtime setup behavior as representative of normal training performance.

The same measurement convention is used for the C0-C4 development comparison.

---

## 10. Frozen C0 Results

The final frozen-engine C0 results are:

| Metric | C0 FP32 |
|---|---:|
| Best validation RMSE | 2.070551 |
| Test RMSE | 2.424142 |
| Test MAE | 1.310130 |
| Mean steady-state epoch time | 4.209343 s |
| Mean throughput | 11,570.598 samples/s |
| Peak allocated CUDA memory | 576.901 MB |
| Peak reserved CUDA memory | 670.000 MB |
| Steady-state epochs | 19 |

These values define the systems reference used when calculating speedup, slowdown, and memory reduction for C1-C4.

---

## 11. Interpretation of the C0 Result

C0 provides a clean answer to the question:

> What does the focal Transformer cost when trained using the basic FP32 CUDA execution path?

The answer on the NVIDIA A40 development environment was approximately:

- 4.21 seconds per steady-state epoch
- 11.57K training samples per second
- 576.90 MB peak allocated CUDA memory
- 670 MB peak reserved CUDA memory
- 2.4241 test RMSE for the seed-42 development run

These measurements are not assumptions or target values.

They were produced by the recorded benchmark execution.

---

## 12. Why C0 Is Necessary

An optimization has no meaning without a reference path.

For example, saying:

> AMP reduced memory.

is incomplete unless the reduction is calculated relative to the same workload executed without AMP.

Likewise, saying:

> torch.compile accelerated training.

would require a reference configuration using the same model, dataset, seed, batch size, training budget, metric definitions, and timing protocol.

C0 therefore provides the denominator for later systems comparisons.

The development comparison is structured as:

- C0: FP32 reference
- C1: AMP
- C2: AMP + pinned memory + non-blocking H2D
- C3: AMP + pinned memory + prefetch stream
- C4: C3 + torch.compile

---

## 13. Superseded Pre-Freeze C0 Measurement

Before the configurable benchmark engine was frozen, an earlier development C0 run produced approximately:

- epoch time: 3.712 seconds
- throughput: 13,139.8 samples/second

That result was collected under an earlier state of the benchmark code.

It is retained only as historical development evidence.

It must not be mixed with the frozen C0-C4 comparison.

The official development C0-C4 comparison uses the C0 run produced at Git SHA:

`25809b3b2d8fede7eec4e369dc60051c68f016ad`

with:

- epoch time: 4.209343 seconds
- throughput: 11,570.598 samples/second

This preserves experimental consistency.

---

## 14. Scientific Interpretation

C0 does not need to be slower than later configurations.

It is the reference, not a deliberately inefficient strawman.

If a later optimization makes the workload slower, that negative result is still scientifically meaningful.

The project therefore does not assume that C1-C4 must outperform C0.

Every optimization must earn its claim through measurement.

---

## 15. Phase 9 Learning Summary

Phase 9 established the following systems concepts.

### Host versus device execution

The CPU host prepares and launches work while tensors used for CUDA computation reside in GPU device memory.

### CUDA kernels

PyTorch operations ultimately invoke GPU kernels or CUDA library operations that execute parallel work on the GPU.

### Asynchronous execution

CUDA work can continue after Python regains control.

This is why synchronization matters for timing.

### Throughput versus epoch time

Throughput measures samples processed per second.

Epoch time measures the time required for one training pass.

Both are recorded because they provide complementary views of end-to-end performance.

### CUDA memory accounting

Allocated memory and reserved memory are not identical.

PyTorch's caching allocator explains why reserved memory can exceed the memory currently occupied by live tensors.

---

## 16. Phase 9 Conclusion

C0 successfully established the reproducible FP32 CUDA reference path.

The final frozen development reference is:

- Test RMSE: `2.424142`
- Test MAE: `1.310130`
- Epoch time: `4.209343 s`
- Throughput: `11,570.598 samples/s`
- Peak allocated memory: `576.901 MB`
- Peak reserved memory: `670.000 MB`

These numbers now serve as the reference for the remaining execution configurations.

The main engineering lesson from Phase 9 is:

> GPU optimization begins with a trustworthy baseline. Timing must respect asynchronous CUDA execution, memory measurements must be clearly defined, and every later speedup or memory claim must be calculated relative to the same controlled workload.

---

## 17. Phase Gate

Before considering Phase 9 conceptually complete, the project owner should be able to explain:

1. Why CUDA operations can return control to Python before GPU work finishes.
2. Why synchronization is required at wall-clock timing boundaries.
3. Why synchronization should not be inserted throughout the hot loop.
4. The difference between throughput, latency, and epoch time.
5. The difference between allocated and reserved CUDA memory.
6. Why C0 remains useful even if a later optimized configuration is slower.
7. Why the older pre-freeze C0 result cannot be mixed with the frozen C0-C4 experiment.

---

## 18. Phase 9 Evidence Status

| Requirement | Status |
|---|---|
| Explicit CUDA device path | Complete |
| Synchronization-safe timing | Complete |
| Steady-state epoch timing | Complete |
| Throughput measurement | Complete |
| Peak allocated memory measurement | Complete |
| Peak reserved memory measurement | Complete |
| GPU/PyTorch/CUDA metadata | Complete |
| FP32 reference result | Complete |
| Frozen benchmark provenance | Complete |
| C0 report | Complete |

**Phase 9 development benchmark status: COMPLETE.**
