# Phase 17: CUDA Kernel Microbenchmarking and Performance Analysis

## GPU-Accelerated Multivariate Forecasting

**Phase:** 17  
**Topic:** CUDA-event timing, operator-level speedups, numerical agreement, benchmark reproducibility, and performance interpretation  
**GPU:** NVIDIA A40  
**Compute Capability:** 8.6  
**PyTorch:** 2.8.0+cu128  
**CUDA Toolkit:** 12.8  
**Precision:** FP32  
**Random Seed:** 42  
**Warm-up Iterations:** 100 per timed function and shape  
**Measured Repetitions:** 500 per timed function and shape  
**Primary Metric:** Median CUDA-event elapsed time  
**Fused Operator:** LayerNorm + tanh-approximate GELU  
**Source:** `cuda/fused_ln_gelu_a40/`  
**Results:** `reports/evidence/phase17/`  
**New GPU Execution During Documentation:** None  
**Archived Benchmark Analysis:** Complete  
**Dockerized GPU Revalidation:** Pending

---

# 1. Executive Summary

Phase 17 evaluates the performance of the custom fused LayerNorm+GELU CUDA operator developed and analyzed during Phase 16.

The purpose is to establish whether the fused implementation offers a measurable performance advantage over the mathematically corresponding eager PyTorch implementation.

The benchmark compares:

```python
F.gelu(
    F.layer_norm(
        x,
        (hidden,),
        weight,
        bias,
        eps,
    ),
    approximate="tanh",
)
```

against:

```python
fused_ln_gelu_cuda.forward(
    x,
    weight,
    bias,
    eps,
)
```

Both operations are evaluated using the same FP32 CUDA tensors, affine parameters, and epsilon.

The source benchmark uses 100 warm-up calls and 500 CUDA-event timing repetitions for each implementation at each tested shape.

Four tensor configurations were evaluated:

```text
4096  × 92
24576 × 92
4096  × 127
4096  × 256
```

The archived results show that the fused operation achieved lower median CUDA-event elapsed time in all four cases.

Median speedups ranged from:

```text
1.2545× to 3.3858×
```

The representative Transformer activation shape was:

```text
24576 × 92
```

For that shape:

```text
Eager median: 112.768 microseconds
Fused median:  89.888 microseconds
Speedup:        1.254539×
Time reduction: 20.2894%
```

The largest recorded maximum absolute difference between eager and fused outputs across the four cases was:

```text
1.9073486328125e-06
```

These observations support close numerical agreement for the tested FP32 inputs.

However, the benchmark did not perform a formal numerical tolerance assertion.

The results represent forward-operator microbenchmarks.

They do not demonstrate improvements in:

- Full Transformer training.
- Backward propagation.
- Total epoch duration.
- Model-level throughput.
- Validation RMSE.
- Test RMSE.
- Full-model memory consumption.

The final Phase 17 conclusion is that the custom fused forward operator was faster than the eager reference for the four measured A40 configurations, with its most relevant measured speedup being approximately 1.25×.

---

# 2. Position in the Overall Project

The GPU optimization progression is:

```text
Phase 9:
C0 FP32 baseline

Phase 10:
C1 Automatic Mixed Precision

Phase 11:
C2 Pinned memory and non-blocking H2D

Phase 12:
C3 CUDA prefetch stream

Phase 13:
PyTorch Profiler and Nsight Systems

Phase 14:
C4 torch.compile

Phase 15:
CUDA programming foundations

Phase 16:
Fused LayerNorm + GELU implementation

Phase 17:
CUDA kernel microbenchmarking
```

Phases 9-14 examined framework-level execution strategies.

Phase 15 introduced custom CUDA programming.

Phase 16 examined the fused kernel, shared-memory reductions, and synchronization.

Phase 17 measures the standalone operator's performance and numerical agreement.

The phase also establishes the limitations of microbenchmark evidence.

---

# 3. Research Questions

Phase 17 addresses the following questions.

1. Does the fused CUDA forward operator execute faster than eager PyTorch?
2. How does the observed speedup depend on tensor dimensions?
3. Which tested shape is most representative of the forecasting Transformer?
4. How closely do the fused outputs match eager PyTorch?
5. How are CUDA-event timings collected?
6. Why are warm-up iterations necessary?
7. Why is median latency used for speedup calculations?
8. What do p95 and mean timings add?
9. Which conclusions are supported directly by the archived evidence?
10. Which conclusions require additional experiments?
11. What should be preserved for future reproduction?
12. How will Dockerized GPU validation extend the existing evidence?

---

# 4. The Mathematical Operation

The fused operator computes:

```math
Y=\mathrm{GELU}_{\tanh}(\mathrm{LayerNorm}(X))
```

For each row with hidden dimension \(D\), LayerNorm computes:

```math
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j
```

and:

```math
\sigma^2=
\frac{1}{D}
\sum_{j=1}^{D}(x_j-\mu)^2
```

Then:

```math
z_j=
\gamma_j
\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_j
```

Finally, the tanh-approximate GELU function is applied:

```math
y_j=
\frac{1}{2}z_j
\left[
1+
\tanh
\left(
\sqrt{\frac{2}{\pi}}
(z_j+0.044715z_j^3)
\right)
\right]
```

The benchmark uses:

```text
epsilon = 1e-5
```

The fused CUDA kernel is FP32-only.

It implements the forward computation.

---

# 5. Why Microbenchmarking Matters

A microbenchmark isolates a relatively small operation or code path.

In this project, the measured operation is the forward calculation of LayerNorm followed by GELU.

The purpose is to reduce interference from unrelated training components.

Those unrelated components include:

- Data loading.
- Host-to-device transfers.
- Optimizer updates.
- Backward propagation.
- Loss calculations.
- Validation passes.
- Checkpoint serialization.
- Full-model execution.

By isolating the operation, we can evaluate whether the fused implementation is efficient for particular input shapes.

However, improving one isolated operation does not guarantee an improvement in a much larger system.

---

# 6. Eager PyTorch Reference

The archived benchmark defines:

```python
def eager():
    return F.gelu(
        F.layer_norm(
            x,
            (hidden,),
            weight,
            bias,
            EPS,
        ),
        approximate="tanh",
    )
```

The reference includes the same five mathematical components as the custom operator:

1. Mean.
2. Variance.
3. Normalization.
4. Affine transformation.
5. GELU activation.

The GELU approximation is explicitly selected as:

```python
approximate="tanh"
```

This is essential because the custom CUDA source also implements tanh-approximate GELU.

The comparison is therefore designed to evaluate corresponding mathematical functions.

---

# 7. Fused CUDA Implementation

The archived benchmark invokes:

```python
def fused():
    return fused_ln_gelu_cuda.forward(
        x,
        weight,
        bias,
        EPS,
    )
```

The compiled extension is built from:

```text
cuda/fused_ln_gelu_a40/binding.cpp
cuda/fused_ln_gelu_a40/fused_kernel.cu
cuda/fused_ln_gelu_a40/setup.py
```

The source benchmark is:

```text
cuda/fused_ln_gelu_a40/benchmark.py
```

The implementation uses:

```text
128 CUDA threads per block
One block per row
Two-pass centered variance
Dynamic shared-memory reductions
Explicit block synchronization
FP32 arithmetic
Tanh-approximate GELU
```

The implementation is forward-only.

---

# 8. Source Version Control

The original repository contained an earlier fused kernel under:

```text
cuda/fused_ln_gelu/
```

The archived A40 implementation differs in:

- Block size.
- Dtype handling.
- Shared-memory strategy.
- Variance calculation.
- Input dimension handling.
- Build configuration.

The A40 implementation was recovered and preserved separately under:

```text
cuda/fused_ln_gelu_a40/
```

The two versions must not be treated as identical.

The Phase 17 benchmark results correspond to the recovered A40 implementation.

---

# 9. Archived Environment

The relevant development environment is:

| Component | Value |
|---|---|
| GPU | NVIDIA A40 |
| Compute capability | 8.6 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| CUDA compiler | nvcc 12.8.93 |

These values describe the original A40 development environment.

The archived performance results were obtained in that environment.

No new GPU measurements were collected during the Mac-based documentation stage.

---

# 10. Benchmark Source and Results

The benchmark implementation is preserved at:

```text
cuda/fused_ln_gelu_a40/benchmark.py
```

The original summarized outputs are preserved at:

```text
reports/evidence/phase17/fused_ln_gelu.json
reports/evidence/phase17/fused_ln_gelu.csv
```

Both result files were copied from the A40 evidence archive.

The copies were compared against the archived originals using `diff -q`.

No differences were found.

This establishes that the preserved results were not silently changed during source recovery.

---

# 11. Random Seed

The benchmark begins with:

```python
torch.manual_seed(42)
```

This controls PyTorch random-number generation for the benchmark's generated tensors, subject to relevant execution and environment behavior.

The script generates:

- Input tensor.
- LayerNorm weight vector.
- LayerNorm bias vector.

The seed improves reproducibility.

It does not establish full experimental determinism or guarantee identical GPU timing across machines or sessions.

The benchmark uses one seed.

It is separate from the future multi-seed forecasting experiment campaign.

---

# 12. Tested Tensor Shapes

The archived script defines:

```python
SHAPES = [
    (4096, 92),
    (24576, 92),
    (4096, 127),
    (4096, 256),
]
```

The shapes vary in row count and hidden dimension.

For each shape:

```text
rows = number of independent activation rows

hidden = number of features normalized per row
```

Each row is normalized independently.

These shapes were selected for the development benchmark.

They are not an exhaustive search over all possible CUDA workloads.

---

# 13. Number of Elements

For a two-dimensional tensor:

```math
N=R\times H
```

where:

- \(R\) is the row count.
- \(H\) is the hidden dimension.

The element counts are:

| Shape | Elements |
|---|---:|
| 4096 × 92 | 376,832 |
| 24576 × 92 | 2,260,992 |
| 4096 × 127 | 520,192 |
| 4096 × 256 | 1,048,576 |

These differences influence the amount of arithmetic, input storage, and memory traffic involved.

They also affect the number of CUDA blocks launched by the fused implementation.

---

# 14. Representative Transformer Shape

The development Transformer configuration includes:

```text
Batch size: 256
Sequence length: 96
Hidden size: 92
```

A corresponding activation tensor has shape:

```text
[256, 96, 92]
```

The number of activation rows is:

```math
256\times96=24576
```

Therefore a flattened representation has shape:

```text
[24576, 92]
```

This is one of the archived benchmark cases.

It is the most directly relevant shape for evaluating the custom operator against the project's Transformer activation dimensions.

However, dimensional correspondence does not establish actual integration into the training graph.

---

# 15. Warm-Up Procedure

The benchmark uses:

```python
WARMUP = 100
```

The measurement helper begins with:

```python
for _ in range(WARMUP):
    fn()

torch.cuda.synchronize()
```

The goal of warm-up is to reduce the influence of initial execution effects.

Potential influences include:

- Initial CUDA runtime activity.
- Lazy initialization.
- Initial memory allocation.
- Kernel loading.
- GPU clock-state changes.
- Framework preparation.

Warm-up does not guarantee identical execution conditions for every later measurement.

It is one component of sound GPU benchmarking practice.

---

# 16. Measured Repetitions

The benchmark defines:

```python
REPS = 500
```

For each tested shape, it collects 500 timing observations for the eager reference.

It then collects 500 timing observations for the fused implementation.

There are four shapes.

Therefore the benchmark performs:

```math
4\times2\times500=4000
```

timed function calls across the full experiment.

This excludes the separate warm-up calls and preliminary correctness evaluations.

The 500 measurements within one benchmark session are not equivalent to 500 independent GPU experiments.

---

# 17. Why Ordinary CPU Timing Can Mislead

CUDA operations are often asynchronous with respect to the CPU.

A Python call can submit GPU work and return before the GPU finishes the calculation.

A naive CPU timer might measure primarily host submission overhead.

For example:

```python
start = time.perf_counter()

fn()

elapsed = time.perf_counter() - start
```

Without suitable synchronization, this does not necessarily measure completed GPU execution.

The archived benchmark instead uses CUDA events.

---

# 18. CUDA-Event Timing

The timing loop creates:

```python
start = torch.cuda.Event(
    enable_timing=True
)

end = torch.cuda.Event(
    enable_timing=True
)
```

It then records:

```python
start.record()

fn()

end.record()
```

The benchmark waits for completion of the ending event:

```python
end.synchronize()
```

Finally:

```python
start.elapsed_time(end)
```

returns the elapsed interval in milliseconds.

This is an appropriate method for measuring short GPU operations.

It avoids relying solely on unsynchronized host-side elapsed time.

---

# 19. Interpretation of CUDA-Event Elapsed Time

The benchmark measures the CUDA-stream interval between its recorded start and end events.

This interval includes the submitted operation's execution on the relevant stream.

Depending on the launch sequence and scheduling, it can also include gaps between GPU activities.

Therefore the measurement should not automatically be interpreted as the sum of pure kernel instruction execution times.

For the eager implementation, the interval covers the LayerNorm and GELU function sequence.

For the fused implementation, the interval covers the custom forward call.

The benchmark is an operator-call timing comparison, not a standalone hardware instruction profiler.

---

# 20. Warm-Up and Synchronization Details

After the warm-up loop:

```python
torch.cuda.synchronize()
```

ensures previously submitted CUDA work has completed before timed repetitions begin.

Inside each timed repetition:

```python
end.synchronize()
```

waits for the ending event.

This means the benchmark measures one function call per timing interval and waits before collecting the next observation.

The timing statistics describe repeated calls under this serialized measurement protocol.

They do not describe the throughput of many simultaneously queued or overlapped operations.

---

# 21. Timing Units

The archived JSON stores times in milliseconds.

For interpretation, this report commonly converts them to microseconds.

The conversion is:

```math
1\ \mathrm{ms}=1000\ \mu\mathrm{s}
```

For example:

```text
0.112768 ms
```

equals:

```text
112.768 microseconds
```

These are submillisecond GPU operations.

Accurate units are essential when reporting their performance.

---

# 22. Mean Latency

For timing observations:

```math
t_1,t_2,\ldots,t_n
```

the mean is:

```math
\bar{t}=
\frac{1}{n}
\sum_{i=1}^{n}t_i
```

The benchmark calculates:

```python
a.mean()
```

Mean latency describes the average recorded execution interval.

It can be influenced by unusually slow observations.

The archived JSON preserves mean latency for both eager and fused execution.

---

# 23. Median Latency

The median, or p50, describes the central point of the recorded timing distribution.

The benchmark calculates:

```python
np.percentile(a, 50)
```

This is the primary statistic used for the archived speedup calculations.

Median latency is often useful for short GPU operations because it is less influenced by isolated extreme observations than the mean.

The measured speedup is defined using the two median values.

---

# 24. 95th-Percentile Latency

The benchmark calculates:

```python
np.percentile(a, 95)
```

The p95 describes a slower point in the measured timing distribution.

It is useful for evaluating variation beyond the median.

For example, an operation may have a fast median but occasional slower executions.

Comparing p50 and p95 can reveal differences in the upper tail of recorded timings.

However, p95 values from one benchmark run do not establish long-term operational reliability.

---

# 25. Standard Deviation and Minimum

The measurement helper also calculates:

```python
a.std()
```

and:

```python
a.min()
```

NumPy's default `std()` uses the population standard deviation convention with `ddof=0`.

However, the archived output row does not include these two fields.

The saved results contain:

```text
mean
p50
p95
```

but not:

```text
std
min
```

The 500 individual observations were also not preserved in the supplied JSON/CSV.

Therefore the standard deviations and minima cannot be reconstructed exactly from the archived summaries.

No values should be invented for those missing metrics.

---

# 26. Speedup Definition

The archived benchmark calculates:

```python
speedup = (
    eager_stats["p50_ms"]
    / fused_stats["p50_ms"]
)
```

Mathematically:

```math
S=
\frac{T_{\mathrm{eager,p50}}}
{T_{\mathrm{fused,p50}}}
```

If:

```math
S>1
```

the fused implementation has lower median latency.

If:

```math
S=1
```

the median times are equal.

If:

```math
S<1
```

the fused implementation has higher median latency.

This is a relative comparison.

It does not directly describe absolute time saved.

---

# 27. Percentage Latency Reduction

The percentage reduction in median execution time is:

```math
R=
100\left(
1-
\frac{T_{\mathrm{fused,p50}}}
{T_{\mathrm{eager,p50}}}
\right)
```

Using the speedup definition:

```math
R=
100\left(1-\frac{1}{S}\right)
```

A speedup of 1.25× does not imply a 25% reduction in execution time.

Instead:

```math
100\left(1-\frac{1}{1.25}\right)
=20\%
```

This distinction is important for accurate performance reporting.

---

# 28. Full Archived Results

The following table contains the saved median and p95 measurements.

All timings below are in microseconds.

| Shape | Eager p50 | Fused p50 | Eager p95 | Fused p95 |
|---|---:|---:|---:|---:|
| 4096 × 92 | 35.072 | 22.592 | 37.8064 | 23.0736 |
| 24576 × 92 | 112.768 | 89.888 | 113.856 | 93.344 |
| 4096 × 127 | 81.152 | 23.968 | 82.208 | 24.736 |
| 4096 × 256 | 42.048 | 27.584 | 43.296 | 28.288 |

The values are converted from the archived millisecond values.

No new runtime measurements were generated.

---

# 29. Archived Mean Timings

The saved mean timing results are:

| Shape | Eager mean (µs) | Fused mean (µs) |
|---|---:|---:|
| 4096 × 92 | 35.439680 | 22.665216 |
| 24576 × 92 | 112.875904 | 91.547712 |
| 4096 × 127 | 81.258752 | 24.077056 |
| 4096 × 256 | 42.149952 | 27.907200 |

The fused operator had lower recorded mean latency in all four cases.

This agrees with the direction of the median comparisons.

The exact mean-derived speedups differ slightly from the median-derived speedups because the statistics summarize different aspects of the timing distributions.

The official archived speedup field uses medians.

---

# 30. Full Median Speedup Results

| Shape | Median speedup |
|---|---:|
| 4096 × 92 | 1.552408× |
| 24576 × 92 | 1.254539× |
| 4096 × 127 | 3.385848× |
| 4096 × 256 | 1.524362× |

The fused implementation was faster by the median timing criterion in all four tested cases.

The smallest speedup was:

```text
1.254539×
```

The largest speedup was:

```text
3.385848×
```

These values describe the measured benchmark matrix only.

---

# 31. Case A: 4096 × 92

The first shape contains:

```math
4096\times92=376832
```

elements.

The archived median timings were:

```text
Eager: 35.0719988346 microseconds
Fused: 22.5920006633 microseconds
```

The speedup is:

```math
S=
\frac{35.0719988346}
{22.5920006633}
\approx1.552408
```

The median latency reduction is approximately:

```text
35.58%
```

The absolute median time saved is approximately:

```text
12.480 microseconds
```

The maximum absolute numerical error was:

```text
9.5367431640625e-07
```

This case shows a measurable advantage for the fused operator at the project's hidden dimension of 92 with a smaller row count than the representative Transformer batch.

---

# 32. Case B: 24576 × 92

The second shape contains:

```math
24576\times92=2260992
```

elements.

This is the most representative tested activation shape for the development Transformer configuration.

The archived median timings were:

```text
Eager: 112.7680018544 microseconds
Fused:  89.8879989982 microseconds
```

The speedup is:

```math
S=
\frac{112.7680018544}
{89.8879989982}
\approx1.254539
```

The median latency reduction is:

```text
20.2894%
```

The absolute median time saved is:

```text
22.880 microseconds
```

The maximum absolute numerical error was:

```text
1.9073486328125e-06
```

The relevant conclusion is:

> For the representative Transformer activation dimensions, the archived A40 benchmark recorded approximately 1.25× forward-operator speedup with close FP32 numerical agreement.

This is not a measured training-step speedup.

---

# 33. Case C: 4096 × 127

The third shape contains:

```math
4096\times127=520192
```

elements.

The archived median timings were:

```text
Eager: 81.1519995332 microseconds
Fused: 23.9679999650 microseconds
```

The speedup is:

```math
S=
\frac{81.1519995332}
{23.9679999650}
\approx3.385848
```

The median latency reduction is:

```text
70.4653%
```

The absolute median time saved is:

```text
57.184 microseconds
```

The maximum absolute numerical error was:

```text
1.9073486328125e-06
```

This was the largest observed speedup.

The eager implementation was particularly slow relative to the fused implementation for this shape.

Possible explanations involve shape-dependent kernel choices or execution efficiency.

However, the archived benchmark does not identify the cause.

A shape-specific profiler trace would be needed to investigate it.

Therefore no specific bottleneck mechanism is claimed.

---

# 34. Case D: 4096 × 256

The fourth shape contains:

```math
4096\times256=1048576
```

elements.

The archived median timings were:

```text
Eager: 42.0479997993 microseconds
Fused: 27.5839995593 microseconds
```

The speedup is:

```math
S=
\frac{42.0479997993}
{27.5839995593}
\approx1.524362
```

The median latency reduction is:

```text
34.3988%
```

The absolute median time saved is:

```text
14.464 microseconds
```

The maximum absolute numerical error was:

```text
1.430511474609375e-06
```

The fused kernel processes hidden dimension 256 using 128 threads per block.

Each thread processes multiple columns across the row.

The result demonstrates that the kernel can produce a latency improvement for a hidden dimension larger than its thread-block size.

---

# 35. Shape-Dependent Performance

The four speedup ratios are not identical.

They range from approximately:

```text
1.25× to 3.39×
```

This demonstrates that the relative advantage varies across the tested shapes.

Possible influencing factors include:

- Number of rows.
- Hidden dimension.
- Reduction work.
- Global-memory access.
- Kernel launch overhead.
- Eager implementation behavior.
- GPU scheduling.
- Data caching.
- Instruction throughput.

The benchmark does not isolate these factors experimentally.

Therefore these are plausible considerations rather than identified causes.

---

# 36. Why the 127-Feature Case Needs Caution

The 4096 × 127 case achieved the largest recorded speedup.

It would be misleading to present that figure as the expected speedup for our forecasting Transformer.

Our development Transformer uses hidden size 92.

The representative benchmark is therefore the 24576 × 92 case.

Furthermore, a single unusually favorable shape does not establish general performance across all hidden dimensions.

The correct reporting strategy is to show the complete matrix and identify which case corresponds most closely to the target workload.

---

# 37. Upper-Tail Latency Analysis

The archived p95 values also favor the fused implementation.

| Shape | Eager p95 (µs) | Fused p95 (µs) |
|---|---:|---:|
| 4096 × 92 | 37.8064 | 23.0736 |
| 24576 × 92 | 113.856 | 93.344 |
| 4096 × 127 | 82.208 | 24.736 |
| 4096 × 256 | 43.296 | 28.288 |

The corresponding ratios of eager p95 to fused p95 are approximately:

```text
4096 × 92:   1.6385×
24576 × 92:  1.2197×
4096 × 127:  3.3234×
4096 × 256:  1.5305×
```

These are ratios of two percentile values.

They are not the 95th percentiles of paired per-repetition speedup observations.

The raw per-repetition data was not preserved, so paired latency analysis is unavailable.

---

# 38. Representative Shape Tail Behavior

For 24576 × 92:

```text
Eager p50: 112.768 microseconds
Eager p95: 113.856 microseconds
```

and:

```text
Fused p50: 89.888 microseconds
Fused p95: 93.344 microseconds
```

The fused implementation has lower recorded p50 and p95 values.

However, its p95 is farther above its median proportionally than the eager implementation's p95 is above its median.

The p95 observations therefore support lower upper-tail absolute latency, but they do not imply the fused timing distribution is uniformly less variable in relative terms.

The saved summary is insufficient for a complete variability analysis.

---

# 39. Numerical Correctness Methodology

Before timing, the archived benchmark calculates:

```python
reference = eager()
candidate = fused()

torch.cuda.synchronize()

max_abs_error = (
    reference - candidate
).abs().max().item()
```

The error metric is:

```math
E_{\max}
=
\max_i
\left|
y_i^{\mathrm{eager}}
-
y_i^{\mathrm{fused}}
\right|
```

This measures the largest absolute output difference over all elements of the tested tensor.

The benchmark records the result for each shape.

---

# 40. Numerical Error Results

| Shape | Maximum absolute error |
|---|---:|
| 4096 × 92 | 9.5367431640625e-07 |
| 24576 × 92 | 1.9073486328125e-06 |
| 4096 × 127 | 1.9073486328125e-06 |
| 4096 × 256 | 1.430511474609375e-06 |

The maximum recorded value is:

```text
1.9073486328125e-06
```

These are small absolute differences for the tested FP32 calculations.

They are consistent with close numerical agreement.

However, the benchmark did not specify or enforce a formal `torch.testing.assert_close` tolerance threshold.

Therefore the historical results should be described as recorded numerical-error observations rather than as a comprehensive automated correctness test.

---

# 41. Why Numerical Error Can Be Nonzero

Both implementations express the same mathematical operation, but their floating-point execution pathways differ.

Potential differences include:

- Reduction order.
- Rounding.
- Reciprocal-square-root behavior.
- Instruction selection.
- Intermediate arithmetic.
- Floating-point approximation details.

Floating-point addition is not generally associative.

For example:

```math
(a+b)+c
```

does not necessarily produce exactly the same floating-point result as:

```math
a+(b+c)
```

The custom CUDA reduction uses a tree-based summation order.

A reference implementation may use a different reduction strategy.

Therefore nonzero numerical differences can occur without indicating an incorrect mathematical formula.

A complete correctness evaluation would use explicit error tolerances and broader test coverage.

---

# 42. Limitations of Maximum Absolute Error

Maximum absolute error is useful but incomplete.

It does not by itself describe:

- Relative error.
- Mean absolute error.
- Error distribution.
- Behavior near zero.
- Sensitivity to extreme input magnitudes.
- NaN or infinity handling.
- Behavior on every possible tensor shape.

The Phase 17 archive only provides maximum absolute output differences.

Additional correctness checks should be performed in the future Dockerized GPU validation stage.

---

# 43. Potential Benefits of Fusion

The benchmark results are consistent with the possibility that fusion reduces overhead.

The eager implementation performs LayerNorm and GELU as separate operations.

The custom implementation combines both within one kernel.

Potential advantages include:

1. Avoiding a complete intermediate normalized tensor.
2. Reducing separate kernel-launch overhead.
3. Applying GELU immediately after the affine transformation.
4. Reusing normalization statistics within the same block.

However, the benchmark does not isolate each factor.

The observed speedups should not be attributed quantitatively to any single optimization mechanism without further profiling.

---

# 44. Illustrative Intermediate Tensor Size

Consider the representative shape:

```text
24576 × 92
```

The tensor contains:

```text
2,260,992 FP32 values
```

Each FP32 value occupies four bytes.

Therefore one such dense tensor contains:

```math
2260992\times4=9043968\ \mathrm{bytes}
```

This equals:

```text
8.625 MiB
```

An eager implementation may materialize an intermediate LayerNorm output of this size before applying GELU.

The fused operator avoids a separate full intermediate tensor between those operations.

This illustrates a potential memory-traffic benefit.

It is not a measurement of total allocated or reserved GPU memory.

Caches and implementation details also affect physical memory traffic.

No actual peak-memory reduction was measured by the Phase 17 benchmark.

---

# 45. Why Operator Speedup Does Not Equal Training Speedup

A complete training iteration includes much more work than LayerNorm and GELU.

For example:

```text
Data loading
    |
    v
Host-to-device transfer
    |
    v
Model forward pass
    |
    v
Loss calculation
    |
    v
Backward pass
    |
    v
Optimizer update
```

The fused benchmark measures only one forward operation.

Even if a fused operator is faster, its influence on total training time depends on:

- Whether the operation occurs in the model.
- How often it executes.
- Its fraction of total runtime.
- Whether it has a compatible backward implementation.
- Whether integration introduces other overhead.

The archived benchmark does not evaluate these conditions.

---

# 46. Forward-Only Limitation

The recovered fused operator exposes:

```python
fused_ln_gelu_cuda.forward(...)
```

It does not include a custom backward implementation or registered autograd formula.

Consequently, it is not established as a drop-in differentiable replacement for training.

A training-ready fused implementation would require valid gradients for relevant inputs and affine parameters.

It would also require backward correctness checks.

The Phase 17 speedups therefore apply only to the forward calculation.

---

# 47. Relationship to Earlier GPU Benchmark Phases

The earlier C0-C4 benchmarks measured development Transformer training behavior.

Those metrics included:

- Epoch time.
- Samples per second.
- GPU memory allocation.
- GPU memory reservation.
- Validation metrics.
- Test metrics.

Phase 17 instead measures a small standalone forward operation.

These results must not be merged as if they were the same experimental unit.

For example:

```text
C4 epoch time:
Seconds per training epoch

Phase 17 kernel time:
Microseconds per forward operator call
```

The scales and workloads are different.

A valid system-level speedup claim requires an integrated benchmark under controlled conditions.

---

# 48. Benchmark Ordering Limitation

The archived benchmark measures:

```python
eager_stats = measure(eager)
fused_stats = measure(fused)
```

Therefore eager timing is collected before fused timing for each shape.

The benchmark does not randomly interleave the two implementations.

Potential time-dependent influences include:

- GPU clock behavior.
- Thermal conditions.
- Competing device activity.
- Cache effects.
- System scheduling changes.

The saved results do not quantify these effects.

A future stronger benchmark could alternate or randomize measurement order across repeated trials.

The existing observations remain useful development evidence.

---

# 49. Single-Session Limitation

The source contains 500 repetitions per measured function and shape.

These repetitions were collected within the original benchmarking procedure.

They are repeated observations under one development environment, not independent replications across multiple machines or sessions.

Therefore the results should not be treated as providing 500 independent experimental replicates.

The archive does not establish run-to-run variance across independently restarted benchmark sessions.

---

# 50. Statistical Inference Limitation

The archived JSON/CSV preserves summarized timings but not individual measurements.

Consequently, the following cannot be reliably reconstructed:

- Full timing histograms.
- Exact empirical standard deviations.
- Paired repetition differences.
- Distributional confidence intervals.
- Bootstrap intervals derived from raw timing observations.
- Independent-run variance.

No statistical significance test is reported.

The benchmark should be characterized as a measured engineering comparison rather than a formal statistical study of GPU performance.

---

# 51. Why the Median Is the Primary Metric

Short-duration GPU timings can be affected by occasional unusual observations.

The median provides a useful central summary.

It is less affected by extreme timing observations than the arithmetic mean.

The archived source explicitly computes speedup from p50 values.

Therefore Phase 17 preserves the source-defined metric rather than choosing a different speedup formula after inspecting results.

Mean and p95 values provide additional context.

---

# 52. Why the Largest Speedup Is Not the Main Result

The largest measured speedup was:

```text
3.385848×
```

for:

```text
4096 × 127
```

The most representative development Transformer shape was:

```text
24576 × 92
```

Its measured speedup was:

```text
1.254539×
```

For project-specific conclusions, workload relevance matters more than selecting the largest number.

Therefore the 1.25× result should be emphasized when discussing relevance to the forecasting Transformer.

The 3.39× result remains part of the complete benchmark evidence.

---

# 53. Artifact Consistency

The historical results were saved in two formats:

```text
JSON
CSV
```

Each contains four result records.

The fields include:

```text
rows
hidden
elements
max_abs_error
eager_mean_ms
eager_p50_ms
eager_p95_ms
fused_mean_ms
fused_p50_ms
fused_p95_ms
speedup_x
```

The files were copied unchanged from the archived A40 session.

A useful offline verification is to confirm that the JSON and CSV records agree and that:

```math
\mathrm{speedup}
=
\frac{\mathrm{eager\ p50}}
{\mathrm{fused\ p50}}
```

for each row.

This verification does not require a GPU.

It does not produce new benchmark measurements.

---

# 54. Reproducibility Requirements

Reproducing the original benchmark requires:

1. The recovered CUDA source.
2. The recovered C++ binding.
3. The extension build configuration.
4. A compatible PyTorch/CUDA environment.
5. An NVIDIA GPU.
6. The benchmark script.
7. The input seed and shapes.
8. The warm-up count.
9. The repetition count.
10. Preservation of the resulting artifacts.

The recovered files provide the original implementation and configuration.

The future Docker environment will supply a reproducible build and execution platform.

---

# 55. GPU Revalidation Plan

The future Dockerized validation will include:

- Confirming the NVIDIA driver and GPU visibility.
- Confirming the CUDA toolkit and compiler.
- Building the recovered extension.
- Importing the compiled module.
- Checking supported FP32 inputs.
- Running explicit numerical assertions.
- Testing edge cases and invalid inputs.
- Running the benchmark under controlled conditions.
- Preserving the new timing observations separately.
- Comparing new measurements with the historical archive.

The historical JSON/CSV should remain unchanged.

New measurements should use new filenames or directories.

The purpose is reproducibility rather than silently replacing historical evidence.

---

# 56. Recommended Future Benchmark Improvements

A stronger later benchmark could include:

- Multiple independently repeated sessions.
- Alternating implementation order.
- Additional hidden dimensions.
- More row counts.
- CUDA Graph-based timing where appropriate.
- Device clock and temperature monitoring.
- Explicit memory measurements.
- Per-repetition timing preservation.
- Confidence intervals based on suitable independent experiments.
- Separate profiling of unusual shapes.
- Explicit numerical tolerance assertions.

These are future improvements, not claims about the archived experiment.

---

# 57. Observation, Evidence, Diagnosis, Decision

## Observation

The fused forward operator recorded lower median CUDA-event latency than eager PyTorch for all four tested FP32 tensor shapes.

The archived output differences were small.

## Evidence

The benchmark source uses:

```text
100 warm-up calls
500 timing repetitions
CUDA events
Mean, p50, and p95 summaries
Median-derived speedup
```

The saved results report:

```text
1.552408× at 4096 × 92

1.254539× at 24576 × 92

3.385848× at 4096 × 127

1.524362× at 4096 × 256
```

The maximum observed absolute numerical difference was approximately:

```text
1.91e-6
```

## Diagnosis

The fused implementation demonstrates shape-dependent operator-level performance advantages.

The representative Transformer activation shape showed a measured speedup of approximately 1.25×.

The results are compatible with the expected benefits of reducing intermediate materialization and combining operations.

However, the benchmark does not isolate the precise cause of each speedup.

The archived evidence is not sufficient to establish statistical significance or end-to-end training improvement.

## Decision

Preserve the original source and result files unchanged.

Report all four measured shapes.

Emphasize the representative 24576 × 92 case.

Do not claim full-model speedup.

Keep the archived benchmark separate from the official C0-C4 forecasting campaign.

Revalidate the extension and performance in the planned NVIDIA Docker environment.

---

# 58. Engineering Lessons

Phase 17 provides several important lessons.

### A faster kernel must be measured

Kernel fusion is a hypothesis until performance is compared against an appropriate reference.

### Mathematical equivalence matters

The reference must match the custom operation, including GELU approximation and epsilon.

### GPU timing requires synchronization discipline

Unsynchronized host timers can misrepresent asynchronous GPU work.

### Units matter

Microseconds, milliseconds, and seconds must not be confused.

### A speedup ratio is not a percentage time reduction

A 1.25× speedup corresponds to approximately 20% lower execution time.

### Shape matters

Performance can vary considerably with tensor dimensions.

### Numerical correctness matters

A fast kernel that produces inaccurate outputs is not a successful optimization.

### Microbenchmarks have limited scope

Standalone operator performance does not automatically imply model-level improvement.

### Raw observations matter

Saving only summaries limits later statistical analysis.

### Reproducibility matters

Source code, toolchain information, benchmark settings, and original results should be preserved together.

---

# 59. Interview-Level Explanation

A concise explanation of the Phase 17 work is:

> I benchmarked a custom fused LayerNorm+GELU CUDA forward operator against the corresponding eager PyTorch implementation on an NVIDIA A40. The test used FP32 tensors, tanh-approximate GELU, 100 warm-up calls, and 500 CUDA-event timing repetitions per implementation and shape. Across four tested shapes, the fused operator achieved median speedups ranging from approximately 1.25× to 3.39×, with maximum absolute differences from the PyTorch reference around 1e-6. The most relevant Transformer activation shape, 24,576 by 92, achieved 1.25× operator speedup, corresponding to about a 20% reduction in measured latency. I preserved the original source, benchmark script, and JSON/CSV results while distinguishing the forward microbenchmark from full-model training performance.

This explanation accurately communicates the engineering contribution without exaggerating the scope of the result.

---

# 60. Relationship to Docker Containerization

The next major engineering stage is reproducible environment containerization.

Before running the final forecasting experiment campaign, the project must establish a consistent execution environment.

The Dockerized system should include:

```text
Linux AMD64 environment
Pinned Python dependencies
PyTorch CUDA runtime
CUDA compilation toolkit
nvcc
C++ build tools
Project source
Experiment scripts
GPU correctness tests
Persistent output directories
```

The original A40 extension should be compiled and validated within that environment.

The historical benchmark data should remain immutable.

Future measurements can be stored separately and compared with the archive.

---

# 61. Relationship to the Official 60-Run Campaign

The planned Phase 18 campaign contains:

```text
4 models
×
3 seeds
×
5 execution configurations
=
60 runs
```

The models are:

```text
LSTM
GRU
TCN
Transformer
```

The seeds are:

```text
17
42
101
```

The execution configurations are:

```text
C0
C1
C2
C3
C4
```

The Phase 17 fused-kernel benchmark is separate from this matrix.

It does not add a sixth execution configuration to the frozen campaign.

Any future integrated fused-kernel training experiment would need its own clearly defined design and backward-correctness validation.

---

# 62. Phase 17 Completion Gate

The documentation milestone requires:

1. Understanding the benchmark reference.
2. Understanding the custom fused operation.
3. Explaining warm-up and CUDA-event timing.
4. Explaining the recorded statistics.
5. Reproducing speedup calculations from saved data.
6. Interpreting all four shapes.
7. Identifying the representative Transformer shape.
8. Explaining maximum absolute error.
9. Recognizing source and benchmark limitations.
10. Preserving the original JSON/CSV files.
11. Separating operator results from training results.
12. Documenting future GPU validation requirements.

The archived evidence supports completion of the source-analysis and documentation stage.

New GPU runtime validation remains pending.

---

# 63. Final Conclusion

Phase 17 completes the historical microbenchmark analysis of the custom fused LayerNorm+GELU CUDA forward operator.

The benchmark compared the fused implementation with an equivalent eager PyTorch reference using FP32 tensors.

It used:

```text
NVIDIA A40
100 warm-up iterations
500 timed repetitions
CUDA-event elapsed time
Median-based speedup
Four tensor shapes
```

All four archived shapes showed lower median latency for the fused implementation.

The recorded speedups were:

```text
4096 × 92:   1.5524×

24576 × 92:  1.2545×

4096 × 127:  3.3858×

4096 × 256:  1.5244×
```

The representative Transformer activation shape achieved:

```math
\boxed{1.2545\times\ \mathrm{forward\ operator\ speedup}}
```

This corresponds to:

```math
\boxed{20.29\%\ \mathrm{lower\ median\ latency}}
```

The largest recorded maximum absolute output difference was approximately:

```math
1.91\times10^{-6}
```

The archived measurements support an operator-level performance advantage for the tested FP32 cases.

They do not establish end-to-end training improvement, backward correctness, or a general speedup across all possible shapes.

The original benchmark source and summarized results were preserved without modification.

The project is ready to transition from documenting historical GPU experiments to preparing the reproducible Docker environment required before the official 60-run campaign.

The central Phase 17 engineering principle is:

> GPU optimization must be evaluated with synchronized measurements, numerically appropriate references, representative workloads, and clear boundaries between microbenchmark evidence and system-level conclusions.

**Phase 17 archived evidence analysis and documentation: COMPLETE.**

**Phase 17 independent Dockerized NVIDIA GPU revalidation: PENDING.**
