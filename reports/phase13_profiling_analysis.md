# Phase 13: GPU Profiling, Bottleneck Diagnosis, and C1-C3 Execution Analysis

## Status

**Phase:** 13 - Profiling with PyTorch Profiler and NVIDIA Nsight Systems  
**Hardware:** NVIDIA A40  
**PyTorch:** 2.8.0+cu128  
**CUDA runtime:** 12.8  
**Development seed:** 42  
**Configurations analyzed:** C1, C2, C3  
**Profiler evidence:** PyTorch Profiler + NVIDIA Nsight Systems  
**Timeline database:** Nsight Systems SQLite export  
**Analysis status:** Complete  
**Primary outcome:** C2/C3 regressions are dominated by host/framework orchestration rather than slower Transformer CUDA computation  
**Copy/compute overlap outcome:** No measurable memcpy/kernel temporal overlap was observed in the captured C3 Nsight timeline

---

## 1. Objective

The purpose of Phase 13 is to move from performance observation to evidence-based diagnosis.

Phases 10-12 established the following development results:

```text
C1 AMP
epoch time   = 4.492150 s
throughput   = 10,849.658 samples/s

C2 AMP + pinned H2D
epoch time   = 14.353531 s
throughput   = 3,394.871 samples/s

C3 AMP + pinned H2D + prefetch stream
epoch time   = 13.590975 s
throughput   = 3,590.779 samples/s
```

Those measurements answered:

> What happened?

They did not fully answer:

> Why did it happen?

Phase 13 therefore investigates the execution behavior using profiler evidence rather than speculation.

The main questions are:

1. Did the Transformer CUDA computation itself become slower in C2 or C3?
2. Did CPU/framework overhead increase?
3. What happened to H2D transfer behavior?
4. Did C3 actually move transfer work onto another CUDA stream?
5. Did the C3 transfer stream achieve measurable copy/compute overlap?
6. What optimization decision is justified by the evidence?

---

## 2. Profiling Tools

Two complementary profiling tools are used.

### PyTorch Profiler

PyTorch Profiler provides a framework/operator-level view.

It helps answer questions such as:

- which PyTorch operations consume CPU time
- which operations consume CUDA time
- how many calls occur
- where copy operations appear
- where framework overhead increases
- whether the CUDA computation associated with major operators changes

A useful mental model is:

> PyTorch Profiler helps explain **what operations consumed the time**.

### NVIDIA Nsight Systems

Nsight Systems provides a systems/timeline-oriented view.

It helps investigate:

- CUDA kernel execution
- H2D, D2D, and D2H memory operations
- CUDA API calls
- streams
- events
- synchronization
- kernel launch activity
- temporal relationships between copies and kernels

A useful mental model is:

> Nsight Systems helps explain **when and where GPU work occurred**.

The tools are complementary.

Their raw timing totals should not be mixed as though they were produced by one identical measurement system.

---

## 3. Preserved Profiling Evidence

The A40 development session preserved the following Phase 13 artifacts:

```text
profiling/c1_amp_torch_table.txt
profiling/c1_amp_torch_trace.json
profiling/c1_nsys.nsys-rep
profiling/c1_nsys.sqlite
profiling/c1_nsys_stats.txt

profiling/c2_amp_pin_torch_table.txt
profiling/c2_amp_pin_torch_trace.json

profiling/c3_amp_pin_prefetch_torch_table.txt
profiling/c3_amp_pin_prefetch_torch_trace.json
profiling/c3_nsys.nsys-rep
profiling/c3_nsys.sqlite
profiling/c3_nsys_stats.txt
```

The evidence archive also contains:

```text
profiling/gpu_session_environment.txt
```

The original evidence is preserved in:

`local_gpu_artifacts/a40_gpu_evidence_complete.tar.gz`

A working copy for local analysis was extracted into:

`local_gpu_artifacts/phase13_analysis/profiling/`

---

## 4. Why C1, C2, and C3 Are Compared Together

C1 is the simple AMP reference:

```text
AMP                    = ON
pin_memory             = OFF
non_blocking_h2d       = OFF
prefetch_stream        = OFF
compile                = OFF
```

C2 adds:

```text
pin_memory             = ON
non_blocking_h2d       = ON
```

C3 retains C2 and additionally enables:

```text
prefetch_stream        = ON
```

The controlled progression is therefore:

```text
C1
AMP
 |
 v
C2
AMP
+ pinned host memory
+ non-blocking H2D
 |
 v
C3
AMP
+ pinned host memory
+ non-blocking H2D
+ dedicated CUDA prefetch stream
```

This allows profiler evidence to be interpreted as a progression rather than as three unrelated runs.

---

## 5. End-to-End Benchmark Context

Before inspecting profiler internals, the steady-state benchmark results were:

| Configuration | Epoch time | Throughput | Test RMSE |
|---|---:|---:|---:|
| C1 AMP | 4.492150 s | 10,849.658 samples/s | 2.339474 |
| C2 AMP + pinned H2D | 14.353531 s | 3,394.871 samples/s | 2.339474 |
| C3 + prefetch stream | 13.590975 s | 3,590.779 samples/s | 2.339474 |

Relative to C1:

```text
C2 throughput change = -68.71%
C3 throughput change = -66.90%
```

Relative to C2:

```text
C3 throughput change = +5.77%
C3 epoch-time change = -5.31%
```

The identical recorded RMSE across C1-C3 indicates that the execution-path changes did not alter the selected model-quality outcome in the seed-42 development runs.

---

## 6. First Profiling Question: Did GPU Compute Become Slower?

The PyTorch Profiler self-time totals were:

| Configuration | Self CPU time | Self CUDA time |
|---|---:|---:|
| C1 | 813.905 ms | 217.213 ms |
| C2 | 2.217 s | 218.917 ms |
| C3 | 2.294 s | 218.931 ms |

This is one of the most important Phase 13 findings.

CPU-side profiler time changed dramatically.

CUDA time barely changed.

Relative to C1, C2 self CPU time increased by approximately:

```text
2.217 / 0.813905
≈ 2.72x
```

while CUDA time changed from:

```text
217.213 ms
```

to:

```text
218.917 ms
```

an increase of less than 1%.

C3 showed the same general pattern.

The captured GPU computation therefore did not become approximately three times slower, even though end-to-end epoch time did.

This strongly shifts the diagnosis away from slower Transformer CUDA math and toward host/framework execution overhead.

---

## 7. Major Matrix Multiplication Evidence

The CUDA time attributed to `aten::mm` was:

```text
C1 = 28.668 ms
C2 = 28.653 ms
C3 = 28.640 ms
```

These values are effectively unchanged.

Matrix multiplication is a major component of Transformer execution.

If the large C2/C3 regression had been caused by dramatically slower GPU matrix computation, a corresponding large increase would be expected in these CUDA timings.

That increase is not present.

---

## 8. LayerNorm Backward Evidence

The CUDA time for native LayerNorm backward was also effectively constant:

```text
C1 ≈ 28.162 ms
C2 ≈ 28.191 ms
C3 ≈ 28.173 ms
```

Again, the CUDA operation itself did not become materially slower across the configurations.

This reinforces the conclusion that the primary regression occurs outside the core Transformer GPU computation.

---

## 9. C1 Host-Side Profile

C1 showed the lowest framework/host overhead of the three configurations.

Important C1 CPU-side measurements included approximately:

```text
DataLoader total CPU       = 142.197 ms
cudaLaunchKernel self CPU  = 82.084 ms
aten::_to_copy self CPU    = 21.202 ms
cudaMemcpyAsync CPU        = 10.376 ms
```

The total profiler CPU time was:

`813.905 ms`

while total self CUDA time was:

`217.213 ms`

This provides the reference framework behavior for the faster AMP-only execution path.

---

## 10. C2 Kernel-Launch Overhead

C2 showed a major increase in host-side CUDA kernel-launch overhead.

PyTorch Profiler recorded:

```text
C1 cudaLaunchKernel self CPU = 82.084 ms
C2 cudaLaunchKernel self CPU = 357.673 ms
```

The ratio is approximately:

```text
357.673 / 82.084
≈ 4.36x
```

Therefore the host spent more than four times as much self CPU time in the CUDA kernel-launch path during the captured C2 profiler window.

Importantly, the number and cost of major GPU compute operations did not increase by a similar factor.

This is evidence of increased host-side launch/orchestration cost.

---

## 11. C2 `_to_copy` Overhead

C1:

```text
aten::_to_copy
self CPU  = 21.202 ms
CPU total = 98.350 ms
```

C2:

```text
aten::_to_copy
self CPU  = 197.031 ms
CPU total = 476.047 ms
```

The self CPU ratio is approximately:

```text
197.031 / 21.202
≈ 9.29x
```

This represents a very large increase in CPU-side copy-related framework overhead.

---

## 12. C2 `cudaMemcpyAsync` Overhead

PyTorch Profiler recorded:

```text
C1 cudaMemcpyAsync CPU = 10.376 ms
C2 cudaMemcpyAsync CPU = 78.778 ms
```

The C2 value is approximately:

```text
7.59x
```

the C1 value.

This is another strong indication that the C2 regression is associated with the host/framework input-transfer execution path.

---

## 13. Actual Copy CUDA Time Did Not Explode

The GPU-side CUDA time associated with `aten::copy_` was approximately:

```text
C1 = 46.959 ms
C2 = 48.663 ms
C3 = 48.696 ms
```

The GPU copy work therefore remained similar.

This produces an important distinction:

> The C2 regression was not caused by the actual GPU copy operation becoming three times slower.

Instead, the CPU/framework work surrounding transfer and launch activity increased substantially.

---

## 14. C2 DataLoader Overhead

The single-process DataLoader also became more expensive in C2.

C1:

```text
DataLoader CPU total = 142.197 ms
average per call     = 4.740 ms
```

C2:

```text
DataLoader CPU total = 254.401 ms
average per call     = 8.480 ms
```

The total increase is approximately:

```text
+78.9%
```

The recorded configurations use:

```yaml
num_workers: 0
```

This is relevant because batch preparation is not distributed across multiple DataLoader worker processes.

However, the profiler evidence does not establish that `num_workers=0` alone caused the regression.

It remains an experimental condition rather than a sole causal explanation.

---

## 15. C2 Diagnosis

Before profiling, the C2 report correctly stated:

> C2 became much slower, but the root cause is not yet assigned.

Phase 13 allows that statement to be refined.

The profiler supports the diagnosis:

> The C2 regression is dominated by increased host/framework orchestration, copy-path overhead, and CUDA-launch overhead rather than a material slowdown in the Transformer CUDA computation itself.

This does not mean:

> pinned memory is universally slow.

It means:

> the recorded C2 execution path, using pinned memory, non-blocking H2D, and `num_workers=0`, produced substantially greater host-side overhead for this workload.

---

## 16. C3 Copy-Path Recovery

C3 introduced the prefetch stream and reduced several C2 host-side costs.

### `_to_copy`

```text
C2 self CPU = 197.031 ms
C3 self CPU = 90.219 ms
```

The reduction is approximately:

```text
54.2%
```

### PyTorch Profiler `cudaMemcpyAsync`

```text
C2 CPU = 78.778 ms
C3 CPU = 7.201 ms
```

The reduction is approximately:

```text
90.9%
```

### `cudaLaunchKernel`

```text
C2 self CPU = 357.673 ms
C3 self CPU = 274.507 ms
```

The reduction is approximately:

```text
23.3%
```

These reductions are consistent with the fact that C3 recovered some end-to-end throughput relative to C2.

---

## 17. C3 Did Not Eliminate Host Overhead

Despite improvements in some copy-path measurements, total self CPU profiler time remained high.

```text
C2 self CPU total = 2.217 s
C3 self CPU total = 2.294 s
```

C3 shifted where framework overhead appeared rather than returning the complete execution path to C1 behavior.

Large C3 host-side entries included operations such as:

```text
aten::empty
aten::t
aten::constant_pad_nd
Optimizer.step
aten::select
cudaMemsetAsync
cudaLaunchKernel
aten::copy_
```

The result therefore remained host/framework expensive even though some transfer-path metrics improved.

---

## 18. C3 End-to-End Recovery

The benchmark result showed:

```text
C2 throughput = 3,394.871 samples/s
C3 throughput = 3,590.779 samples/s
```

This is approximately:

`+5.77%`

C2 epoch time:

`14.353531 s`

C3 epoch time:

`13.590975 s`

This is approximately:

`-5.31%`

Therefore C3 produced a real but modest recovery relative to C2.

However, C3 remained approximately:

`66.90%`

lower in throughput than C1.

The prefetch stream therefore did not repair the overall input-pipeline regression.

---

## 19. Important Profiler Interpretation Rule

Profiler operator tables can contain nested or attributed activity.

Individual rows should not be manually summed as though every entry were disjoint.

For example, an operation can contain lower-level CUDA work that also appears under more specific operators.

For high-level comparison, the profiler's reported total self CPU and self CUDA time should be used as the consistent reference.

The major high-level totals were:

```text
C1 CUDA = 217.213 ms
C2 CUDA = 218.917 ms
C3 CUDA = 218.931 ms
```

The near-equality of those totals is one of the strongest pieces of evidence in the Phase 13 diagnosis.

---

## 20. Nsight Systems Kernel Evidence

Nsight Systems independently supports the conclusion that C1 and C3 perform essentially the same GPU computational workload.

The leading kernels have almost identical:

- instance counts
- total execution times
- average execution times
- median execution times

Examples include:

- elementwise kernels
- Gamma/Beta backward kernels
- CUTLASS GEMM kernels
- reduction kernels
- LayerNorm kernels
- FlashAttention backward kernels
- GELU kernels
- dropout kernels

The leading C1 kernel accounted for approximately:

`31.774 ms`

while the equivalent C3 kernel accounted for approximately:

`31.787 ms`

with the same:

`1,190 instances`

This pattern repeats across the major kernel list.

Therefore Nsight independently confirms:

> The fundamental CUDA kernel workload is essentially unchanged between the fast C1 path and the slower C3 path.

---

## 21. Nsight C1 Memory Operations

The C1 Nsight memory-operation summary recorded:

| Operation | Count | Total time |
|---|---:|---:|
| Device-to-Device memcpy | 315 | 4.858 ms |
| Host-to-Device memcpy | 101 | 1.715 ms |
| CUDA memset | 315 | 0.155 ms |
| Device-to-Host memcpy | 35 | 0.063 ms |

The C1 H2D average duration was approximately:

`16.98 µs`

per operation.

---

## 22. Nsight C3 Memory Operations

The C3 Nsight memory-operation summary recorded:

| Operation | Count | Total time |
|---|---:|---:|
| Device-to-Device memcpy | 315 | 4.867 ms |
| Host-to-Device memcpy | 103 | 3.401 ms |
| CUDA memset | 315 | 0.149 ms |
| Device-to-Host memcpy | 35 | 0.051 ms |

The C3 H2D average duration was approximately:

`33.02 µs`

per operation.

Therefore the prefetch implementation did not make individual H2D copies intrinsically faster.

The intended benefit of prefetching was instead to create an opportunity to hide transfer latency behind computation.

---

## 23. Prefetching Does Not Mean Faster Copies

This distinction is important.

The goal of a prefetch stream is not necessarily:

```text
make memcpy duration smaller
```

The goal is closer to:

```text
perform memcpy while useful computation is already occurring
```

Therefore an H2D copy could theoretically take the same amount of time, or even somewhat longer, while still improving end-to-end execution if enough of that transfer occurs concurrently with compute.

This is why aggregate copy duration alone cannot prove whether C3 succeeded.

A timeline-level overlap analysis is required.

---

## 24. Nsight CUDA API Evidence

The C1 CUDA API summary reported:

```text
cudaLaunchKernel
7622 calls
343.340 ms total

cudaMemcpyAsync
451 calls
11.384 ms total
```

C3 reported:

```text
cudaLaunchKernel
7622 calls
717.390 ms total

cudaMemcpyAsync
453 calls
11.426 ms total
```

The kernel-launch count is identical:

`7,622 calls`

but the total C3 CPU API time attributed to `cudaLaunchKernel` is approximately:

`2.09x`

the C1 value.

This independently supports the presence of significantly greater host-side launch/orchestration overhead in the more complex C3 execution path.

---

## 25. PyTorch Profiler and Nsight Are Not Numerically Interchangeable

PyTorch Profiler and Nsight Systems report timing from different instrumentation layers and collection contexts.

For example, PyTorch Profiler showed a large C2-to-C3 reduction in CPU attribution for `cudaMemcpyAsync`.

The C1 and C3 Nsight API summaries show similar aggregate `cudaMemcpyAsync` API time.

This is not treated as a contradiction.

The tools have different:

- collection windows
- attribution semantics
- instrumentation layers
- overhead characteristics

The correct approach is:

> compare like with like within each profiling tool and use the tools together for diagnosis.

The project does not directly subtract a PyTorch Profiler number from an Nsight number.

---

## 26. Evidence of Additional C3 Stream/Event Machinery

The C3 Nsight API summary contains explicit stream and event operations that reflect the more complex prefetch execution path.

Examples include:

```text
cudaStreamCreateWithPriority
128 calls

cudaEventQuery
140 calls

cudaEventRecord
70 calls

cudaEventRecordWithFlags
107 calls

cudaStreamWaitEvent
35 calls

cudaStreamSynchronize
66 calls
```

These operations are evidence that the C3 execution path introduced additional stream/event coordination.

This machinery is necessary for some forms of safe asynchronous execution, but it is not free.

---

## 27. Why Aggregate Nsight Statistics Are Not Enough to Prove Overlap

Aggregate Nsight summaries answer:

- how many kernels executed
- how much total kernel time occurred
- how many memory copies occurred
- how much total copy time occurred
- how many CUDA API calls occurred

They do not directly answer:

> Did a specific H2D copy occur at the same time as a compute kernel?

That is a timeline question.

Therefore the preserved C3 Nsight SQLite database was queried directly using the CUPTI activity tables.

---

## 28. Nsight SQLite Tables Used

The relevant C3 SQLite tables include:

```text
CUPTI_ACTIVITY_KIND_KERNEL
CUPTI_ACTIVITY_KIND_MEMCPY
CUPTI_ACTIVITY_KIND_MEMSET
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
CUPTI_ACTIVITY_KIND_CUDA_EVENT
```

Both kernel and memcpy records contain:

```text
start
end
deviceId
contextId
streamId
```

These fields make it possible to test temporal overlap directly.

---

## 29. C3 Memcpy Stream Distribution

The C3 SQLite query produced:

```text
copyKind  streamId  copies  total_ms  avg_us
8         7         315     4.867     15.450
1         13         72     3.176     44.113
2         7         35     0.051      1.447
1         7         31     0.224      7.240
```

The copy-kind mapping is identified by matching these counts with the Nsight memory-operation summary:

```text
copyKind 8 = Device-to-Device
copyKind 1 = Host-to-Device
copyKind 2 = Device-to-Host
```

The H2D total is:

```text
72 + 31 = 103 H2D copies
```

which exactly matches the Nsight summary.

---

## 30. Dedicated Transfer Stream Was Actually Used

The H2D distribution shows:

```text
72 H2D copies on stream 13
31 H2D copies on stream 7
```

Therefore C3 did not merely create a stream object without using it.

A substantial portion of the H2D transfer workload was actually scheduled on stream 13.

This proves that the dedicated transfer-stream execution path was active.

---

## 31. Compute Kernel Stream

The C3 kernel-stream query produced:

```text
streamId = 7
kernels  = 8,602
total kernel time = 247.070 ms
```

All recorded CUDA kernels in the query executed on:

`stream 7`

Therefore the captured C3 execution had the conceptual structure:

```text
Stream 7
- all compute kernels
- some H2D copies
- D2D copies
- D2H copies

Stream 13
- 72 H2D copies
```

This created the structural possibility for transfer/compute concurrency.

---

## 32. Direct Memcpy/Kernel Overlap Query

To test actual overlap, every memcpy interval was joined against every kernel interval sharing the same CUDA device and context.

An overlap exists only when:

```text
memcpy.start < kernel.end
AND
kernel.start < memcpy.end
```

For matching intervals, overlap duration was computed as:

```text
MIN(memcpy.end, kernel.end)
-
MAX(memcpy.start, kernel.start)
```

Only positive overlap durations were retained.

The result of the query was:

```text
NO ROWS
```

Therefore the captured C3 Nsight timeline contained:

```text
0 memcpy/kernel interval intersections
```

---

## 33. C3 Copy/Compute Overlap Result

This is the decisive Phase 13 timeline result.

C3 successfully:

- created an additional stream
- placed 72 H2D copies on stream 13
- kept compute kernels on stream 7

However:

> No memcpy interval temporally overlapped a compute-kernel interval in the captured Nsight trace.

Therefore the intended C3 mechanism:

```text
copy next batch
WHILE
compute current batch
```

was not observed in the captured profiling window.

---

## 34. Multiple Streams Did Not Produce Concurrency

The captured behavior can be represented conceptually as:

```text
Transfer stream 13:

H2D COPY  =====


Compute stream 7:

                 KERNEL  =========
```

rather than:

```text
Transfer stream 13:

       H2D COPY =====


Compute stream 7:

KERNEL =================
```

The first pattern uses multiple streams without overlap.

The second pattern represents the intended concurrent copy/compute behavior.

The profiler evidence supports the first pattern for the captured C3 window.

---

## 35. Why This Result Matters

This directly demonstrates an important CUDA systems lesson:

> Multiple streams create the possibility of concurrency. They do not guarantee concurrency.

A program can:

- allocate a second CUDA stream
- issue asynchronous copies
- use events
- use stream waits
- execute correctly

and still achieve zero actual temporal overlap between transfer and compute.

Profiler evidence is therefore required before claiming that a multi-stream implementation improved concurrency.

---

## 36. C3 Performance Recovery Cannot Be Attributed to Proven Overlap

C3 improved throughput relative to C2 by approximately:

`5.77%`

However, because the timeline query found zero memcpy/kernel overlap, the project does not claim:

> C3 improved because H2D copies overlapped GPU kernels.

That causal statement is not supported by the trace.

A more defensible statement is:

> C3 changed transfer scheduling and reduced several host-side copy-path costs relative to C2, producing a modest end-to-end recovery, but no H2D/compute temporal overlap was observed in the captured Nsight trace.

This distinction is important.

---

## 37. Phase 13 Observation

The central observation is:

> C2 and C3 are dramatically slower than C1 even though their recorded Transformer CUDA computation is essentially unchanged.

End-to-end:

```text
C1 = 4.492 s/epoch
C2 = 14.354 s/epoch
C3 = 13.591 s/epoch
```

Profiler CUDA totals:

```text
C1 = 217.213 ms
C2 = 218.917 ms
C3 = 218.931 ms
```

The mismatch between end-to-end regression and stable CUDA compute time is the key diagnostic clue.

---

## 38. Phase 13 Evidence

The evidence supporting the diagnosis includes:

1. Self CUDA time is almost identical across C1-C3.
2. Major GEMM CUDA time is nearly identical.
3. LayerNorm backward CUDA time is nearly identical.
4. C2 self CPU profiler time is approximately 2.72x C1.
5. C2 `cudaLaunchKernel` self CPU time is approximately 4.36x C1.
6. C2 `_to_copy` self CPU time is approximately 9.29x C1.
7. C2 PyTorch-profiler `cudaMemcpyAsync` CPU time is approximately 7.59x C1.
8. C2 DataLoader CPU total increases substantially.
9. C3 reduces several C2 copy-path host costs.
10. Nsight shows nearly identical C1/C3 GPU kernel execution.
11. C3 introduces explicit stream/event orchestration.
12. C3 schedules 72 H2D copies on stream 13.
13. All 8,602 captured compute kernels execute on stream 7.
14. The direct SQLite interval query finds zero memcpy/kernel overlaps.

---

## 39. Phase 13 Diagnosis

The evidence supports the following diagnosis:

> The C2/C3 performance regression is primarily associated with increased host/framework orchestration and input-path overhead rather than slower Transformer CUDA computation.

C2 introduces substantial increases in:

- host-side copy handling
- framework conversion/copy overhead
- CUDA kernel-launch overhead
- single-process DataLoader overhead

C3 changes this behavior and recovers some performance, but introduces additional stream/event coordination and does not achieve measurable H2D/kernel overlap in the captured timeline.

The workload should therefore not be described as suffering from a simple GPU-compute bottleneck.

The regression is better characterized as:

**host/framework orchestration dominated**

within the captured development profiling evidence.

---

## 40. What Is Not Proven

The profiling evidence does not establish that:

- pinned memory is universally harmful
- `num_workers=0` is the sole cause
- non-blocking copies are inherently slower
- CUDA streams are ineffective in general
- another model size would behave the same way
- another GPU architecture would behave the same way
- all possible C3 executions would show zero overlap

The correct scope is:

> No memcpy/kernel temporal overlap was observed in the captured C3 Nsight profiling window for this Transformer workload and execution environment.

This scoped wording preserves scientific accuracy.

---

## 41. Optimization Decision

The evidence changes the optimization decision.

It would be inappropriate to continue adding input-pipeline complexity simply because pinned memory, asynchronous copies, and streams are conventional GPU optimization techniques.

For this workload:

```text
C1
simple AMP path
```

is substantially faster than:

```text
C2
AMP + pinned/non-blocking H2D
```

and:

```text
C3
AMP + pinned/non-blocking H2D + prefetch stream
```

Therefore the decision is:

> Preserve C1 as the strongest simple development execution path, retain C2 and C3 as controlled experimental configurations, and report their measured regressions transparently rather than trying to force them to outperform C1.

C2/C3 remain valuable in the final matrix because they test the original systems hypotheses under controlled conditions.

---

## 42. Why C2 and C3 Should Not Be Deleted

A tempting response to the negative results would be to remove the configurations.

That would weaken the scientific design.

The purpose of the configuration matrix is not:

> keep only optimizations that worked.

The purpose is:

> test each declared execution strategy consistently and report its actual effect.

C2 and C3 are therefore retained because they provide evidence about:

- workload dependence
- host-side overhead
- transfer scheduling
- CUDA-stream behavior
- the limits of optimization folklore

---

## 43. Why We Did Not Rewrite the Trainer After C2/C3

The poor C2/C3 results were initially treated as profiling questions.

That was the correct decision.

If the training engine had been repeatedly edited until pinned memory or streams produced a positive result, the experiment would risk becoming outcome-driven.

Instead, the workflow was:

```text
observe regression
      |
      v
freeze evidence
      |
      v
profile execution
      |
      v
diagnose behavior
      |
      v
document result
```

This preserves the integrity of the final experiment campaign.

---

## 44. Profiling Overhead Versus Benchmark Timing

Profiler runs are diagnostic runs.

They are not substitutes for the synchronized steady-state epoch benchmark.

Profilers introduce additional instrumentation and can change execution timing.

Therefore:

```text
benchmark timing
```

is used for end-to-end performance claims, while:

```text
profiler timing
```

is used to understand where execution time and activity occur.

The project does not report profiler-wall-time as ordinary training throughput.

---

## 45. Timeline Profiling Versus Operator Profiling

Phase 13 demonstrates the distinction between two profiling perspectives.

### Operator profiling

PyTorch Profiler showed:

- `_to_copy` overhead
- `cudaLaunchKernel` overhead
- DataLoader overhead
- operator-level CPU/CUDA attribution

It was especially useful for showing that CUDA compute remained stable while host-side overhead increased.

### Timeline/system profiling

Nsight Systems showed:

- kernel execution summaries
- memory-copy summaries
- CUDA API calls
- stream/event machinery
- stream IDs
- exact start/end intervals

It was especially useful for proving that the second transfer stream existed but did not overlap compute kernels in the captured window.

---

## 46. Why a Profiler-Based Diagnosis Is Stronger Than Intuition

Before profiling, multiple explanations were possible:

```text
GPU compute became slower
H2D transfer became slower
pinning overhead dominated
DataLoader became slower
stream synchronization caused waiting
CPU launch overhead increased
no useful overlap occurred
```

The profiler eliminated several possibilities.

GPU computation did not materially change.

The large difference appeared primarily on the host/framework side.

The timeline also directly rejected the hypothesis that C3 achieved actual memcpy/kernel overlap in the captured window.

This transforms the discussion from speculation into evidence.

---

## 47. Interview-Level Explanation

A concise technical explanation of Phase 13 is:

> We benchmarked AMP, pinned/non-blocking H2D, and a CUDA prefetch-stream path on an A40. C2 and C3 were dramatically slower than the simple AMP configuration, so instead of changing the code until the optimization looked successful, we profiled the runs. PyTorch Profiler showed that total CUDA compute stayed almost constant at roughly 217-219 ms while host-side CPU time increased from about 0.81 s in C1 to more than 2.2 s in C2/C3. Copy-related framework overhead and CUDA kernel-launch overhead increased substantially. Nsight Systems independently showed nearly identical GPU kernel workloads between C1 and C3. C3 did move 72 H2D transfers onto a separate stream, but a direct interval query against the Nsight SQLite trace found zero memcpy/kernel temporal overlaps. So the regression was primarily host/orchestration dominated, and the extra stream created concurrency potential without realizing copy/compute overlap in the captured trace.

---

## 48. Observation -> Evidence -> Diagnosis -> Decision

### Observation

C2 and C3 were dramatically slower than C1.

```text
C1 throughput = 10,849.658 samples/s
C2 throughput = 3,394.871 samples/s
C3 throughput = 3,590.779 samples/s
```

### Evidence

PyTorch Profiler:

```text
C1 CPU  = 813.905 ms
C2 CPU  = 2.217 s
C3 CPU  = 2.294 s

C1 CUDA = 217.213 ms
C2 CUDA = 218.917 ms
C3 CUDA = 218.931 ms
```

Nsight:

- major C1/C3 kernel execution is essentially unchanged
- C3 contains additional stream/event orchestration
- 72 H2D transfers execute on stream 13
- 31 H2D transfers execute on stream 7
- all 8,602 kernels execute on stream 7
- no memcpy/kernel interval intersections were observed

### Diagnosis

The development regression is primarily associated with host/framework orchestration and input-path overhead rather than slower Transformer CUDA computation.

C3 created a real secondary transfer stream, but the captured execution did not achieve actual copy/compute overlap.

### Decision

Do not add further transfer-pipeline complexity solely on theoretical grounds.

Retain C1-C3 as controlled configurations.

Use C1 as the strongest simple development path.

Carry all declared configurations into the final controlled campaign and report actual multi-seed results.

---

## 49. Phase 13 Learning Summary

Phase 13 established the following concepts.

### Profiling before optimizing

A performance regression should be diagnosed before code is changed.

### CPU time versus CUDA time

Large end-to-end slowdowns do not necessarily mean GPU computation became slower.

### Kernel-launch overhead

The CPU must submit GPU work, and launch overhead can matter for small workloads with many kernels.

### Operator attribution

Framework-level profiling helps locate expensive host-side operations.

### Timeline profiling

Aggregate statistics are not enough to prove concurrency.

Start/end timestamps and stream information are required.

### CUDA streams

A separate stream creates concurrency opportunity but not guaranteed concurrency.

### H2D transfer

Prefetching attempts to hide transfer time rather than necessarily make individual transfers faster.

### Events and synchronization

Multi-stream execution introduces coordination machinery that also has overhead.

### Scoped conclusions

A profiler finding must be limited to the captured workload, hardware, configuration, and profiling window.

### Evidence over folklore

Pinned memory, non-blocking copies, and CUDA streams are mechanisms, not guaranteed speedups.

---

## 50. Phase Gate

Before considering Phase 13 conceptually complete, the project owner should be able to explain:

1. The difference between PyTorch Profiler and Nsight Systems.
2. The difference between operator profiling and timeline profiling.
3. Why profiler timing should not replace the steady-state epoch benchmark.
4. Why C2/C3 end-to-end slowdown does not imply slower GPU mathematics.
5. What the stable 217-219 ms CUDA totals imply.
6. Why `cudaLaunchKernel` is relevant to host-side overhead.
7. Why `_to_copy` and DataLoader CPU time matter.
8. Why pinned-memory behavior cannot be diagnosed from GPU VRAM metrics alone.
9. Why multiple CUDA streams do not guarantee concurrency.
10. How stream IDs were extracted from the Nsight SQLite database.
11. How H2D copies were identified from the copy-kind counts.
12. Why 72 H2D copies on stream 13 prove the secondary transfer stream was used.
13. Why all 8,602 kernels on stream 7 identify the compute stream.
14. How temporal overlap was tested using start/end intervals.
15. Why an empty overlap query means no memcpy/kernel temporal overlap was observed.
16. Why that result must be scoped to the captured profiling window.
17. Why C3's +5.77% recovery cannot be attributed to proven H2D/compute overlap.
18. Why C1 remains the stronger simple development execution path.
19. Why C2/C3 should remain in the controlled experiment matrix despite their negative results.
20. Why evidence-driven negative findings strengthen the project.

---

## 51. Phase 13 Evidence Status

| Requirement | Status |
|---|---|
| PyTorch Profiler workflow | Complete |
| C1 PyTorch profile | Complete |
| C2 PyTorch profile | Complete |
| C3 PyTorch profile | Complete |
| C1 Nsight Systems profile | Complete |
| C3 Nsight Systems profile | Complete |
| Nsight kernel summary | Complete |
| Nsight memory-operation summary | Complete |
| Nsight CUDA API summary | Complete |
| Nsight SQLite export | Complete |
| CPU-vs-CUDA comparison | Complete |
| Copy-path overhead analysis | Complete |
| Kernel-launch overhead analysis | Complete |
| DataLoader overhead analysis | Complete |
| C1/C3 kernel comparison | Complete |
| C3 transfer stream identified | Complete |
| C3 compute stream identified | Complete |
| H2D copy-kind mapping | Complete |
| Memcpy/kernel overlap query | Complete |
| H2D/compute overlap observed | No |
| Bottleneck diagnosis | Complete |
| Optimization decision | Complete |
| Profiler evidence preserved | Complete |
| Negative result documented | Complete |

---

## 52. Phase 13 Conclusion

Phase 13 converted the C2/C3 performance regression from an unexplained benchmark result into an evidence-supported systems diagnosis.

The central finding is:

> The slower C2/C3 execution is not explained by slower Transformer CUDA computation.

PyTorch Profiler recorded approximately:

```text
C1 self CPU  = 813.905 ms
C2 self CPU  = 2.217 s
C3 self CPU  = 2.294 s

C1 self CUDA = 217.213 ms
C2 self CUDA = 218.917 ms
C3 self CUDA = 218.931 ms
```

Major CUDA operations such as matrix multiplication and LayerNorm backward remained nearly unchanged.

C2 instead showed large increases in:

- CUDA kernel-launch CPU overhead
- `_to_copy` CPU overhead
- `cudaMemcpyAsync` CPU attribution
- DataLoader CPU time

C3 reduced several C2 copy-path overheads and recovered approximately:

```text
+5.77% throughput
```

relative to C2, but remained approximately:

```text
66.90% below C1 throughput
```

Nsight Systems independently confirmed that the major C1 and C3 GPU kernels had nearly identical execution characteristics.

The C3 SQLite timeline analysis additionally showed:

```text
H2D copies on stream 13 = 72
H2D copies on stream 7  = 31

compute kernels on stream 7 = 8,602
```

The direct memcpy/kernel interval query returned:

```text
NO ROWS
```

Therefore:

> No measurable memcpy/kernel temporal overlap was observed in the captured C3 Nsight profiling window.

The C3 design successfully created and used a separate H2D transfer stream, but the captured execution did not realize the intended copy/compute concurrency.

The final Phase 13 diagnosis is:

> For this small Transformer forecasting workload on the NVIDIA A40, the C2/C3 regression is primarily associated with host/framework orchestration and input-path overhead. The C3 prefetch stream changed scheduling and partially recovered C2 performance, but did not produce measurable H2D/kernel overlap in the captured Nsight trace.

The resulting engineering decision is:

> Preserve the simple C1 AMP path as the stronger development execution path, retain C2 and C3 as controlled experimental conditions, and report their measured behavior transparently rather than adding further pipeline complexity without profiler evidence.

This phase reinforces the central systems principle of the project:

> Do not optimize from intuition alone. Measure the workload, profile the execution, verify the intended mechanism, and let the evidence determine the next engineering decision.

**Phase 13 profiling and bottleneck diagnosis status: COMPLETE.**
