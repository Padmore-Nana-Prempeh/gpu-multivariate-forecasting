# Phase 10: C1 Automatic Mixed Precision Benchmark

## Status

**Phase:** 10 - Automatic Mixed Precision  
**Execution configuration:** C1 - AMP  
**Hardware:** NVIDIA A40  
**Development seed:** 42  
**Reference configuration:** C0 FP32  
**Benchmark status:** Completed  
**Role in project:** Isolate the effect of mixed precision on memory, speed, and model quality

---

## 1. Objective

The purpose of C1 is to isolate the effect of Automatic Mixed Precision while
holding the remaining forecasting workload and systems configuration constant.

C1 changes precision behavior relative to C0 but does not yet introduce pinned
host memory, non-blocking host-to-device transfer, CUDA prefetch streams, or
torch.compile.

The central question is:

> What happens to GPU memory consumption, throughput, epoch time, and forecast
> quality when AMP is enabled for the same Transformer workload used in C0?

---

## 2. C1 Execution Configuration

The frozen C1 configuration is:

| Setting | C1 value |
|---|---|
| AMP | Enabled |
| CUDA autocast | Enabled |
| GradScaler | Enabled |
| Pinned host memory | Disabled |
| Non-blocking H2D transfer | Disabled |
| CUDA prefetch stream | Disabled |
| torch.compile | Disabled |
| Device | CUDA |
| Batch size | 256 |
| Seed | 42 |

Therefore the C0-to-C1 systems comparison primarily isolates the change from
the FP32 reference path to Automatic Mixed Precision.

---

## 3. Benchmark Environment

C1 was executed in the same frozen benchmark environment used for C0.

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

The same dataset, split, model architecture, optimizer settings, training
budget, checkpoint-selection logic, timing instrumentation, and evaluation
pipeline were used.

---

## 4. Recorded Evidence

The frozen C1 result was written to:

`results/cuda_amp/transformer_seed42_amp1_pin0_prefetch0_compile0/result.json`

The aggregated result also appears in:

`results/summary.csv`

and:

`results/system_speedups.csv`

The A40 development evidence was preserved in the GPU evidence archive after
the session.

---

## 5. What Automatic Mixed Precision Means

AMP does not simply convert every computation in the model to FP16.

Instead, AMP uses a mixture of numerical precisions.

Operations that can benefit from lower precision can execute in a
lower-precision format, while operations that need greater numerical stability
can remain in higher precision.

The goal is to obtain some of the memory and compute benefits of lower
precision without blindly forcing every operation into FP16.

Conceptually:

`training operation -> autocast policy -> appropriate execution precision`

The exact precision used by an operation is determined by PyTorch's autocast
behavior and the underlying CUDA implementation.

---
  
  ## 6. FP32 and FP16
  
  FP32 uses 32 bits per floating-point value.

FP16 uses 16 bits per floating-point value.

At a simplified storage level:
  
  `FP32 value -> 4 bytes`

`FP16 value -> 2 bytes`

This means lower-precision tensors can require substantially less memory.

However, FP16 also has less numerical range and precision than FP32.

That creates potential numerical problems during training, particularly for
very small gradient values.

---
  
  ## 7. Autocast
  
  The C1 execution path uses CUDA autocast around forward and loss computation.

The conceptual purpose of autocast is:
  
  - use lower precision where it is supported and advantageous
- retain safer precision for operations where lower precision can be
problematic
- avoid manually assigning a dtype to every operation in the network

Autocast therefore makes mixed-precision execution practical while preserving
a common training loop.

---
  
  ## 8. Tensor Core Opportunity
  
  Modern NVIDIA GPUs contain Tensor Cores optimized for many matrix operations in
lower-precision formats.

Transformers contain many matrix-heavy operations, including:
  
  - query projection
- key projection
- value projection
- attention matrix multiplication
- attention output projection
- feed-forward linear layers

AMP can therefore create an opportunity for these operations to use
Tensor-Core-friendly execution paths.

However, the existence of Tensor Cores does not guarantee an end-to-end
training speedup.

Actual performance depends on factors including:
  
  - matrix dimensions
- model size
- batch size
- sequence length
- operator mix
- kernel launch overhead
- data movement
- framework overhead
- hardware utilization

AMP must therefore be measured rather than assumed to be faster.

---
  
  ## 9. Why Gradient Scaling Is Needed
  
  FP16 cannot represent the same numerical range as FP32.

During backpropagation, some gradients may become extremely small.

A sufficiently small value can underflow toward zero in FP16.

If important gradients disappear numerically, optimization can become unstable
or ineffective.

Gradient scaling reduces this risk.

The basic idea is:
  
  `loss -> multiply by scale -> backward -> larger temporary gradients`

Before the optimizer uses the gradients, they are returned to their proper
scale.

This allows FP16 training to preserve small gradient information more
effectively.

---
  
  ## 10. GradScaler Execution Order
  
  The C1 training path follows the important conceptual order:
  
  `autocast forward/loss`

then:
  
  `scale loss`

then:
  
  `backward`

then:
  
  `unscale gradients`

then:
  
  `gradient clipping`

then:
  
  `optimizer step`

then:
  
  `update GradScaler`

The order matters.

Gradient clipping must operate on the real unscaled gradient magnitudes.

If clipping were performed before unscaling, the clipping decision would be
based on artificially enlarged gradient values produced by the temporary loss
scale.

Therefore:
  
  `unscale before clipping`

is required for the intended gradient-clipping behavior.

---
  
  ## 11. Dynamic Gradient Scaling
  
  GradScaler can adapt its scaling factor during training.

Conceptually:
  
  - if gradients remain numerically safe, the scale can remain large or increase
- if overflow or invalid values are detected, unsafe optimizer updates can be
avoided and the scale can be reduced

This allows the system to search for a useful scale without requiring one fixed
manual value for the entire training process.

---
  
  ## 12. Frozen C0 and C1 Results
  
  The matched seed-42 development comparison produced:
  
  | Metric | C0 FP32 | C1 AMP |
  |---|---:|---:|
  | Best validation RMSE | 2.070551 | 2.035426 |
  | Test RMSE | 2.424142 | 2.339474 |
  | Test MAE | 1.310130 | 1.258119 |
  | Mean steady-state epoch time | 4.209343 s | 4.492150 s |
  | Mean throughput | 11,570.598 samples/s | 10,849.658 samples/s |
  | Peak allocated CUDA memory | 576.901 MB | 265.917 MB |
  | Peak reserved CUDA memory | 670.000 MB | 370.000 MB |
  | Steady-state epochs | 19 | 19 |
  
  The workload and benchmark protocol were held constant while the AMP path was
enabled.

---
  
  ## 13. Allocated Memory Reduction
  
  C0 peak allocated memory:
  
  `576.901 MB`

C1 peak allocated memory:
  
  `265.917 MB`

The percentage reduction is:
  
  `(576.901 - 265.917) / 576.901 x 100`

which gives:
  
  `53.91%`

Therefore C1 reduced peak allocated CUDA memory by approximately:
  
  **53.91%**
  
  relative to C0.

This is the clearest positive systems effect observed in the C0-to-C1
development comparison.

---
  
  ## 14. Reserved Memory Reduction
  
  C0 peak reserved CUDA memory:
  
  `670.000 MB`

C1 peak reserved CUDA memory:
  
  `370.000 MB`

The percentage reduction is:
  
  `(670 - 370) / 670 x 100`

which gives:
  
  `44.78%`

Therefore C1 reduced peak reserved CUDA memory by approximately:
  
  **44.78%**
  
  relative to C0.

Allocated and reserved memory are reported separately because they represent
different aspects of PyTorch CUDA memory management.

---
  
  ## 15. Throughput Comparison
  
  C0 throughput:
  
  `11,570.598 samples/s`

C1 throughput:
  
  `10,849.658 samples/s`

The throughput ratio is:
  
  `10,849.658 / 11,570.598`

which gives:
  
  `0.937692x`

Rounded:
  
  **0.938x C0 throughput**
  
  The percentage throughput change is:
  
  `-6.23%`

Therefore AMP did not improve end-to-end throughput for this development
workload.

Instead, C1 processed approximately 6.23% fewer training samples per second
than C0.

---
  
  ## 16. Epoch-Time Comparison
  
  C0 mean steady-state epoch time:
  
  `4.209343 s`

C1 mean steady-state epoch time:
  
  `4.492150 s`

The relative change is:
  
  `(4.492150 / 4.209343 - 1) x 100`

which gives approximately:
  
  `+6.72%`

Therefore the AMP configuration required approximately 6.72% more time per
steady-state training epoch than the FP32 reference.

This agrees with the throughput result.

Lower throughput and longer epoch time both indicate that C1 was slower than
C0 for this workload.

---
  
  ## 17. Why Lower Memory and Lower Speed Are Not Contradictory
  
  Memory efficiency and execution speed are different system properties.

AMP can reduce the amount of memory needed for eligible tensors because
lower-precision values require less storage.

At the same time, AMP introduces or interacts with additional execution costs,
including:
  
  - autocast behavior
- dtype conversions where required
- GradScaler bookkeeping
- loss scaling
- gradient unscaling
- overflow checks
- kernel-launch behavior
- hardware utilization characteristics

The focal Transformer is relatively small:
  
  `222,072 parameters`

with:
  
  `sequence length = 96`

and:
  
  `hidden dimension = 92`

For a workload of this scale, the amount of matrix computation may be too small
for Tensor Core acceleration to dominate all of the additional mixed-precision
overhead.

Therefore the measured combination:
  
  `large memory reduction + small throughput regression`

is entirely plausible.

It is not internally contradictory.

---
  
  ## 18. Development Accuracy Observation
  
  C0 test RMSE:
  
  `2.424142`

C1 test RMSE:
  
  `2.339474`

The relative difference is:
  
  `(2.339474 / 2.424142 - 1) x 100`

which is approximately:
  
  `-3.49%`

Therefore the seed-42 C1 development run happened to achieve a lower test RMSE
than the seed-42 C0 run.

However, this is not sufficient evidence to claim:
  
  > AMP improves forecasting accuracy by 3.49%.

Only one development seed is represented in this comparison.

The final controlled experiment campaign uses:
  
  - seed 17
- seed 42
- seed 101

Accuracy conclusions will therefore be based on aggregation across the planned
seeds.

The correct current interpretation is:
  
  > In the seed-42 development benchmark, C1 produced a lower test RMSE than C0,
> but the direction and magnitude of any accuracy effect must be evaluated
> across the final planned seeds.

---
  
  ## 19. C0-to-C1 Systems Summary
  
  Relative to C0, C1 produced:
  
  | Quantity | C1 relative result |
  |---|---:|
  | Allocated-memory reduction | 53.91% |
  | Reserved-memory reduction | 44.78% |
  | Throughput ratio | 0.938x |
  | Throughput change | -6.23% |
  | Epoch-time change | +6.72% |
  | Seed-42 RMSE change | -3.49% |
  
  The central systems result is:
  
  > AMP substantially improved GPU memory efficiency but did not improve
> end-to-end training speed for this small Transformer forecasting workload on
> the NVIDIA A40.

---
  
  ## 20. Why This Is a Useful Result
  
  The purpose of the project is not to force every common optimization to appear
successful.

A technically sound benchmark can show that an optimization helps one resource
while hurting another.

C1 is a good example.

AMP provided a strong memory advantage:
  
  `53.91% lower peak allocated memory`

but the development run also showed:
  
  `6.23% lower throughput`

This demonstrates why GPU optimizations must be evaluated on the actual
workload and hardware rather than copied as universal rules.

---
  
  ## 21. Scientific Limitations
  
  The C0-C1 development comparison currently has several limitations.

### Single seed

Only seed 42 is represented.

Forecast-quality differences cannot yet be treated as final architecture or
precision conclusions.

### One focal architecture

This development systems comparison uses the Transformer.

The final benchmark matrix will also evaluate LSTM, GRU, and TCN.

### One GPU model

The measurements were collected on an NVIDIA A40.

Performance behavior may differ on other NVIDIA architectures.

### Small model

The focal Transformer has approximately 222K parameters.

Larger models with more matrix-heavy computation may obtain a different speed
benefit from AMP.

These limitations do not invalidate the development measurement.

They define the scope of the current result.

---
  
  ## 22. Phase 10 Learning Summary
  
  Phase 10 establishes the following concepts.

### Automatic Mixed Precision

AMP mixes numerical precision rather than forcing every operation into FP16.

### Autocast

Autocast selects appropriate execution precision for supported operations.

### Gradient underflow

Very small gradients may become numerically difficult to represent in FP16.

### Gradient scaling

GradScaler temporarily scales the loss so that backpropagated gradients occupy
a safer numerical range.

### Unscale before clipping

Gradient clipping must operate on the true gradient magnitudes, not temporary
scaled values.

### Tensor Cores

Lower-precision matrix operations can create Tensor Core acceleration
opportunities, but end-to-end benefit depends on the actual workload.

### Measurement over assumption

AMP is not automatically faster.

For this workload, the measured primary benefit was memory reduction.

---
  
  ## 23. Phase 10 Conclusion
  
  The frozen C1 development benchmark successfully measured the impact of AMP
relative to the C0 FP32 reference.

The strongest measured result was memory efficiency.

Peak allocated CUDA memory decreased from:
  
  `576.901 MB`

to:
  
  `265.917 MB`

which corresponds to:
  
  **53.91% less allocated memory**
  
  Peak reserved CUDA memory decreased by:
  
  **44.78%**
  
  However, end-to-end training speed did not improve.

Throughput changed from:
  
  `11,570.598 samples/s`

to:
  
  `10,849.658 samples/s`

or:
  
  **0.938x C0 throughput**
  
  The seed-42 development result therefore supports the conclusion:
  
  > On the NVIDIA A40, AMP dramatically reduced the memory footprint of the
> focal Transformer but slightly reduced end-to-end throughput.

The result is retained exactly as measured rather than being altered to fit the
original performance hypothesis.

---
  
  ## 24. Phase Gate
  
  Before considering Phase 10 conceptually complete, the project owner should be
able to explain:
  
  1. The conceptual difference between FP32 and FP16.
2. Why lower precision can reduce GPU memory consumption.
3. Why forcing every operation into FP16 is unsafe.
4. What autocast does.
5. Why FP16 gradients can underflow.
6. What GradScaler protects against.
7. Why gradients must be unscaled before gradient clipping.
8. Why Tensor Cores can improve performance for some workloads.
9. Why AMP can reduce memory while making a small model slower.
10. Why the seed-42 RMSE difference is not yet a final accuracy claim.

---
  
  ## 25. Phase 10 Evidence Status
  
  | Requirement | Status |
  |---|---|
  | AMP configurable | Complete |
  | CUDA autocast path | Complete |
  | FP16 GradScaler path | Complete |
  | Gradient unscale before clipping | Complete |
  | NaN/inf-safe scaler behavior | Complete |
  | Matched C0/C1 workload | Complete |
  | Peak allocated memory measured | Complete |
  | Peak reserved memory measured | Complete |
  | Throughput measured | Complete |
  | Epoch time measured | Complete |
  | Forecast quality recorded | Complete |
  | C0-vs-C1 comparison calculated | Complete |
  | Single-seed limitation documented | Complete |
  | C1 report | Complete |
  
  **Phase 10 development benchmark status: COMPLETE.**
