# Phase 12: C3 CUDA Prefetch Stream and Copy/Compute Overlap

## Status

**Phase:** 12 - CUDA Streams and Batch Prefetch Overlap  
**Execution configuration:** C3 - AMP + Pinned Memory + Non-Blocking H2D + CUDA Prefetch Stream  
**Hardware:** NVIDIA A40  
**Development seed:** 42  
**Primary reference configuration:** C2 AMP + Pinned H2D  
**Secondary reference configuration:** C1 AMP  
**Benchmark status:** Completed  
**Profiler interpretation:** Deferred to Phase 13  
**Role in project:** Add explicit transfer-stream scheduling on top of the C2 pinned/non-blocking input path and measure its end-to-end effect

---

## 1. Objective

The purpose of C3 is to investigate whether a dedicated CUDA transfer stream can improve the C2 input pipeline by preloading the next training batch while computation proceeds on the current batch.

C2 already enabled:

- AMP
- pinned host memory
- non-blocking H2D transfer

C3 keeps those features and adds:

- a dedicated CUDA prefetch stream

C3 does not introduce:

- torch.compile

The main question is:

> Does adding a dedicated CUDA prefetch stream improve end-to-end training performance relative to the C2 pinned/non-blocking path?

A second question is:

> Does the stream-based design create actual copy/compute overlap?

The first question can be answered from benchmark timing.

The second requires profiler evidence and is therefore deferred to Phase 13.

---

## 2. Recorded C3 Configuration

The saved C3 configuration is:

```yaml
seed: 42

data:
  path: data/ETTm1.csv
  seq_len: 96
  horizon: 24
  train_ratio: 0.7
  val_ratio: 0.1
  batch_size: 256
  num_workers: 0
  pin_memory: true

model:
  name: transformer
  hidden_size: 92
  num_layers: 2
  dropout: 0.1
  nhead: 4
  tcn_channels:
    - 64
    - 88
    - 88
    - 88
    - 88

train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: true
  prefetch_stream: true
  compile: false
  device: cuda
  non_blocking_h2d: true

output:
  dir: results/cuda_amp_prefetch
```

The configuration source is:

`configs/cuda_amp_prefetch.yaml`

The important systems settings are therefore:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = ON
torch.compile          = OFF
num_workers            = 0
```

C3 differs from C2 primarily by enabling:

```text
prefetch_stream = true
```

---

## 3. Forecasting Workload Held Constant

C3 uses the same Transformer forecasting workload used in C1 and C2.

Input shape:

`[batch_size, 96, 7]`

Target shape:

`[batch_size, 24, 7]`

For the recorded development run:

```text
batch_size = 256
lookback   = 96
horizon    = 24
features   = 7
```

The focal model remains:

`Transformer`

with:

`222,072 parameters`

The forecasting problem, training budget, checkpoint-selection logic, and evaluation path therefore remain unchanged.

The systems variable added in C3 is the prefetch-stream execution path.

---

## 4. Benchmark Environment

The recorded C3 run used:

| Component | Recorded value |
|---|---|
| Python | 3.12.3 |
| Platform | Linux x86_64 |
| GPU | NVIDIA A40 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Model | Transformer |
| Parameters | 222,072 |
| Batch size | 256 |
| Seed | 42 |
| Frozen benchmark Git SHA | `25809b3b2d8fede7eec4e369dc60051c68f016ad` |

The same frozen benchmark engine used for C0-C2 was used for C3.

This preserves a common:

- dataset
- temporal split
- scaler
- model architecture
- optimizer
- batch size
- seed
- epoch budget
- gradient clipping
- checkpoint-selection rule
- evaluation path
- timing protocol
- result schema

---

## 5. Recorded Run and Evidence

The recorded C3 run ID is:

`transformer_seed42_amp1_pin1_prefetch1_compile0`

The primary result artifact is:

`results/cuda_amp_prefetch/transformer_seed42_amp1_pin1_prefetch1_compile0/result.json`

The run directory also contains:

```text
best.pt
status.json
environment.json
config.json
result.json
```

The original GPU evidence was preserved in:

`local_gpu_artifacts/a40_gpu_evidence_complete.tar.gz`

The saved C3 profiling evidence includes:

```text
profiling/c3_amp_pin_prefetch_torch_table.txt
profiling/c3_amp_pin_prefetch_torch_trace.json
profiling/c3_nsys.nsys-rep
profiling/c3_nsys.sqlite
profiling/c3_nsys_stats.txt
```

These profiler artifacts will be formally interpreted in Phase 13.

---

## 6. What a CUDA Stream Is

A CUDA stream is an ordered queue of GPU work.

Operations submitted to the same stream preserve ordering.

Conceptually:

```text
Stream A:

Operation A1
    |
    v
Operation A2
    |
    v
Operation A3
```

The GPU preserves the required order within that stream.

This is important because many dependencies rely on one operation completing before another can safely consume its output.

---

## 7. Default Stream Behavior

Without a dedicated transfer stream, transfer and compute operations may be ordered within the same execution stream.

Conceptually:

```text
H2D COPY
   |
   v
FORWARD
   |
   v
BACKWARD
   |
   v
NEXT H2D COPY
```

Even when `non_blocking=True` is used, same-stream ordering can prevent useful overlap between transfer and compute.

This is why C2 provided asynchronous-transfer capability without explicitly scheduling transfer work on a separate CUDA stream.

---

## 8. Why C3 Adds a Second Stream

C3 introduces a dedicated CUDA stream for prefetching the next batch.

The goal is to allow two categories of work to exist independently:

```text
Compute stream
    |
    +-- forward
    +-- backward
    +-- optimizer work

Transfer stream
    |
    +-- copy next batch to GPU
```

Conceptually, the desired timeline is:

```text
time ---------------------------------------------------->

Compute stream:
TRAIN BATCH 1  ==================
                     TRAIN BATCH 2  ==================
                                         TRAIN BATCH 3  ==================

Transfer stream:
       COPY BATCH 2  =====
                           COPY BATCH 3  =====
                                               COPY BATCH 4  =====
```

If hardware resources and dependencies allow it, some transfer work can occur while current-batch computation is still running.

This is the intended copy/compute overlap.

---

## 9. What Prefetching Means

Prefetching means preparing the next batch before it is immediately needed by the compute path.

Without prefetching:

```text
copy batch 1
train batch 1
copy batch 2
train batch 2
copy batch 3
train batch 3
```

With prefetching, the intended behavior is closer to:

```text
train batch 1
    +
copy batch 2 concurrently

then

train batch 2
    +
copy batch 3 concurrently
```

The goal is to hide part of the transfer latency behind useful computation.

---

## 10. CUDAPrefetcher Concept

The C3 execution path uses a CUDA prefetcher abstraction.

At a conceptual level, the prefetcher:

1. receives the next CPU batch
2. enters a dedicated transfer stream
3. moves the batch to CUDA using non-blocking copies
4. prepares the next batch before the compute loop needs it
5. synchronizes stream dependencies before the compute stream consumes the tensors

A simplified conceptual form is:

```python
transfer_stream = torch.cuda.Stream()

with torch.cuda.stream(transfer_stream):
    next_x = x.to(
        "cuda",
        non_blocking=True,
    )

    next_y = y.to(
        "cuda",
        non_blocking=True,
    )
```

The actual implementation must also handle stream dependencies and tensor lifetime correctly.

---

## 11. Why Stream Synchronization Is Required

Separate CUDA streams may execute independently.

This creates concurrency opportunities, but it also creates dependency risks.

Suppose the transfer stream is still copying batch 2 while the compute stream tries to begin the forward pass for batch 2.

That would be unsafe.

Conceptually:

```text
Transfer stream:
COPY BATCH 2
====================>

Compute stream:
          tries to use BATCH 2 too early
```

The compute stream must therefore wait until the required transfer work has completed.

Conceptually:

```text
Transfer stream:
COPY BATCH 2
====================|
                    |
                    | dependency
                    v
Compute stream:
                 WAIT
                    |
                    v
              FORWARD BATCH 2
```

The synchronization must protect correctness without unnecessarily serializing unrelated work.

---

## 12. `wait_stream()` Concept

PyTorch provides stream coordination mechanisms such as:

```python
current_stream.wait_stream(transfer_stream)
```

The conceptual meaning is:

> Do not allow work on the current stream to consume data that is still being prepared by the transfer stream.

This establishes an execution dependency between the streams.

The purpose is not to globally synchronize the GPU.

The purpose is to enforce only the dependency required for safe tensor consumption.

---

## 13. Tensor Lifetime and `record_stream()`

Asynchronous multi-stream execution also creates a tensor-lifetime problem.

PyTorch uses a CUDA caching allocator.

A tensor may no longer be referenced by normal Python control flow even though GPU work on another stream is still using the tensor's underlying memory.

Without proper lifetime tracking, the allocator could potentially reuse memory before all asynchronous work has finished using it.

Conceptually:

```text
tensor allocated
      |
      v
transfer stream uses tensor
      |
      v
compute stream still needs tensor
      |
      v
Python reference changes
      |
      v
allocator must NOT recycle memory too early
```

`record_stream()` allows the allocator to associate tensor memory with asynchronous use on another stream.

Conceptually:

```python
tensor.record_stream(stream)
```

means:

> This tensor's storage is still being used by work on this stream.

This helps preserve correct tensor lifetime in asynchronous multi-stream execution.

---

## 14. Separate Streams Do Not Guarantee Overlap

A second CUDA stream creates the possibility of overlap.

It does not guarantee overlap.

Useful concurrency still depends on:

- transfer size
- compute duration
- GPU copy-engine availability
- memory bandwidth
- PCIe behavior
- kernel resource usage
- stream dependencies
- synchronization placement
- whether current computation already saturates the GPU
- CPU-side batch-preparation speed

Therefore the presence of two streams is not proof that copy and compute actually overlapped.

That must be verified using profiler evidence.

---

## 15. Why Profiler Evidence Matters

From timing alone, we can determine:

- C3 was faster or slower than C2
- the magnitude of the difference

From timing alone, we cannot prove:

- that H2D copy overlapped with compute
- how much overlap occurred
- whether the GPU was idle
- whether host preparation dominated
- whether synchronization serialized work

These questions require timeline evidence.

This is why the C3 A40 session preserved both:

- PyTorch Profiler traces
- Nsight Systems traces

Formal overlap interpretation belongs to Phase 13.

---

## 16. Why C2 Is the Primary Reference for C3

C2 uses:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = OFF
compile                = OFF
```

C3 uses:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = ON
compile                = OFF
```

Therefore the primary incremental change is:

```text
prefetch_stream:
OFF -> ON
```

C2 is therefore the correct direct systems reference for evaluating the effect of the prefetch stream.

---

## 17. C3 Training Completion

The C3 run completed successfully.

Recorded status:

`completed`

Training budget:

`20 epochs`

The best validation checkpoint occurred at:

`epoch 4`

with:

`best validation RMSE = 2.0354264971`

The selected checkpoint was then evaluated on the held-out test set.

---

## 18. C3 Forecast-Quality Result

The recorded C3 test metrics were:

| Metric | C3 |
|---|---:|
| Test MSE | 5.473136 |
| Test RMSE | 2.339474 |
| Test MAE | 1.258119 |

These values match C1 and C2 exactly in the recorded development runs.

C1:

```text
RMSE = 2.339474
MAE  = 1.258119
```

C2:

```text
RMSE = 2.339474
MAE  = 1.258119
```

C3:

```text
RMSE = 2.339474
MAE  = 1.258119
```

This is strong correctness evidence for the systems path.

The transfer scheduling changed, but the recorded model-quality result did not.

---

## 19. C2 Versus C3 Systems Results

The frozen steady-state comparison is:

| Metric | C2 AMP + Pinned | C3 + Prefetch Stream |
|---|---:|---:|
| Test RMSE | 2.339474 | 2.339474 |
| Test MAE | 1.258119 | 1.258119 |
| Mean steady-state epoch time | 14.353531 s | 13.590975 s |
| Mean throughput | 3,394.871 samples/s | 3,590.779 samples/s |
| Mean per-epoch peak allocated CUDA memory | 265.917 MB | 266.738 MB |
| Mean per-epoch peak reserved CUDA memory | 370.000 MB | 414.000 MB |

C3 therefore improved performance relative to C2, but it did not return to the C1 performance level.

---

## 20. C3 Throughput Improvement Relative to C2

C2 throughput:

`3,394.871 samples/s`

C3 throughput:

`3,590.779 samples/s`

The throughput ratio is:

```text
3,590.779 / 3,394.871
= 1.057707
```

Therefore C3 achieved approximately:

**1.058x C2 throughput**

The percentage improvement is approximately:

**+5.77%**

This means the prefetch-stream path recovered some of the performance lost in C2.

---

## 21. C3 Epoch-Time Improvement Relative to C2

C2 mean steady-state epoch time:

`14.353531 s`

C3 mean steady-state epoch time:

`13.590975 s`

The reduction is:

```text
14.353531 - 13.590975
= 0.762556 s
```

The percentage reduction is approximately:

**5.31%**

Therefore C3 reduced steady-state epoch time by approximately:

**5.31%**

relative to C2.

This is consistent with the throughput improvement.

---

## 22. C3 Relative to C1

Although C3 improved relative to C2, C1 remains an important secondary reference.

C1 throughput:

`10,849.658 samples/s`

C3 throughput:

`3,590.779 samples/s`

The ratio is:

```text
3,590.779 / 10,849.658
= 0.330958
```

Therefore C3 achieved only:

**0.331x C1 throughput**

The corresponding throughput change is approximately:

**-66.90%**

C3 remained dramatically slower than the simpler C1 AMP path.

---

## 23. C3 Epoch Time Relative to C1

C1 steady-state epoch time:

`4.492150 s`

C3 steady-state epoch time:

`13.590975 s`

The ratio is:

```text
13.590975 / 4.492150
= 3.025494
```

Therefore C3 required approximately:

**3.025x the C1 epoch time**

The percentage increase is approximately:

**+202.55%**

Therefore the prefetch-stream path did not recover the large performance loss introduced when moving from C1 to the pinned/non-blocking pipeline.

---

## 24. Interpretation of the C3 Performance Result

The C3 timing result is mixed.

Relative to C2:

```text
throughput improved by 5.77%
epoch time decreased by 5.31%
```

Relative to C1:

```text
throughput remained 66.90% lower
epoch time remained 202.55% higher
```

Therefore the correct interpretation is:

> The dedicated CUDA prefetch stream produced a measurable recovery relative to the C2 path, but the complete C3 pipeline remained substantially slower than the simpler C1 AMP configuration.

This is more accurate than saying:

> CUDA streams made training faster.

The latter statement would ignore the stronger C1 reference.

---

## 25. Does the Timing Result Prove Copy/Compute Overlap?

No.

The fact that C3 was faster than C2 does not prove that actual H2D/compute overlap occurred.

Timing tells us:

```text
C3 > C2 in throughput
```

It does not tell us:

```text
the reason was definitely overlap
```

Other execution effects may also contribute.

The project therefore treats:

```text
real copy/compute overlap
```

as a profiler question.

Nsight Systems and PyTorch Profiler evidence will be inspected in Phase 13.

---

## 26. C3 CUDA Allocated Memory

C2 mean per-epoch peak allocated CUDA memory:

`265.917 MB`

C3 mean per-epoch peak allocated CUDA memory:

`266.738 MB`

The absolute increase is approximately:

`0.820 MB`

The percentage increase is approximately:

`0.31%`

Therefore allocated GPU memory remained almost unchanged.

The prefetch stream did not materially change the core model-memory footprint.

---

## 27. C3 CUDA Reserved Memory

C2 mean per-epoch peak reserved CUDA memory:

`370.000 MB`

C3 mean per-epoch peak reserved CUDA memory:

`414.000 MB`

The increase is:

`44 MB`

or approximately:

**11.89%**

This summary value is the mean of the per-epoch peak-reserved-memory measurements used by the result summarizer.

It is not the maximum raw value seen during the entire run.

---

## 28. Important Reserved-Memory Detail

The raw C3 epoch history shows reserved-memory values increasing over the run.

Examples include:

```text
epoch 1   = 338 MB
epoch 4   = 386 MB
epoch 10  = 410 MB
epoch 13  = 422 MB
epoch 16  = 434 MB
epoch 20  = 450 MB
```

Therefore:

`414 MB`

should be described as the summarized mean per-epoch peak reserved memory.

The maximum raw per-epoch reserved value observed in the saved C3 history was:

`450 MB`

at epoch 20.

This distinction is important for precise reporting.

---

## 29. Why Prefetching Can Increase Memory Pressure

A prefetch pipeline may temporarily keep more than one batch resident or associated with GPU execution.

Conceptually:

```text
current batch
already on GPU
and being computed
        +
next batch
already transferred or being prepared
```

This can increase allocator pressure relative to a strictly sequential pipeline.

That provides a plausible mechanism for the higher reserved-memory behavior.

However, the current report does not claim this is the sole cause of the observed memory difference.

The measured values are reported directly, while deeper allocator interpretation remains secondary.

---

## 30. Raw C3 History Shows Persistent Slowness

The raw C3 training history does not show a return to the C1 timing regime.

Examples include approximately:

```text
epoch 2   = 14.40 s
epoch 4   = 13.80 s
epoch 8   = 12.91 s
epoch 10  = 11.73 s
epoch 13  = 12.91 s
epoch 17  = 12.51 s
epoch 20  = 13.61 s
```

These values remain far above the C1 steady-state mean:

`4.492150 s`

Therefore the C3 regression relative to C1 is not explained by one isolated startup epoch.

It persists throughout training.

---

## 31. What We Can Conclude from C3

The saved C3 evidence supports the following statements:

1. The CUDA prefetch path executed successfully.
2. The full 20-epoch run completed.
3. Forecast-quality metrics remained identical to C1 and C2 in the recorded seed-42 development run.
4. C3 improved throughput relative to C2 by approximately 5.77%.
5. C3 reduced epoch time relative to C2 by approximately 5.31%.
6. C3 remained much slower than C1.
7. C3 allocated CUDA memory remained close to C2.
8. C3 reserved-memory requirements were higher than C2.
9. The timing result alone does not prove that copy and compute overlapped.
10. Formal overlap interpretation requires profiler evidence.

---

## 32. What We Cannot Yet Conclude

The Phase 12 benchmark does not yet justify statements such as:

> H2D transfer successfully overlapped with compute.

or:

> CUDA streams caused the full observed performance recovery.

Those are causal claims.

They require timeline evidence showing the relevant memcpy and kernel regions on separate streams with actual temporal overlap.

The saved profiling artifacts exist specifically to answer this question in Phase 13.

---

## 33. Profiler Evidence Preserved

The C3 A40 session preserved:

```text
profiling/c3_amp_pin_prefetch_torch_table.txt
profiling/c3_amp_pin_prefetch_torch_trace.json
profiling/c3_nsys.nsys-rep
profiling/c3_nsys.sqlite
profiling/c3_nsys_stats.txt
```

The Nsight report is especially important because Phase 13 will inspect:

- CPU launch behavior
- H2D copies
- CUDA API calls
- compute kernels
- stream assignment
- synchronization
- copy/compute overlap
- idle gaps

The profiler evidence is therefore preserved before any causal interpretation is made.

---

## 34. Why the Profiler Is the Correct Next Step

C3 created a systems hypothesis:

> A dedicated transfer stream may hide some H2D latency behind current-batch compute.

The benchmark produced:

```text
+5.77% throughput vs C2
```

but this only establishes correlation with the C3 configuration.

The profiler provides the next level of evidence.

The project therefore follows:

```text
implementation
      |
      v
benchmark
      |
      v
observe timing difference
      |
      v
profile timeline
      |
      v
determine whether intended overlap occurred
```

This is more rigorous than assuming that the intended mechanism actually happened.

---

## 35. Relationship Between C2 and C3

The controlled progression is:

```text
C2
AMP
+
pinned memory
+
non-blocking H2D
+
single/default compute ordering
        |
        v
C3
AMP
+
pinned memory
+
non-blocking H2D
+
dedicated prefetch stream
```

The comparison isolates the stream-prefetch mechanism more cleanly than comparing C3 directly with C0.

---

## 36. Relationship Between C1 and C3

C1 remains important because it represents the simpler AMP path before pinned-memory and prefetch complexity was introduced.

C1:

```text
10,849.658 samples/s
4.492150 s/epoch
```

C3:

```text
3,590.779 samples/s
13.590975 s/epoch
```

Therefore, even though C3 improved relative to C2, the overall C1-to-C3 progression was strongly negative.

This prevents the project from presenting the C2-to-C3 recovery without context.

---

## 37. C1, C2, and C3 Summary

| Metric | C1 AMP | C2 Pinned H2D | C3 Prefetch |
|---|---:|---:|---:|
| Test RMSE | 2.339474 | 2.339474 | 2.339474 |
| Epoch time | 4.492150 s | 14.353531 s | 13.590975 s |
| Throughput | 10,849.658/s | 3,394.871/s | 3,590.779/s |
| Mean peak allocated CUDA memory | 265.917 MB | 265.917 MB | 266.738 MB |
| Mean peak reserved CUDA memory | 370 MB | 370 MB | 414 MB |

The progression shows:

```text
C1 -> C2
large regression

C2 -> C3
partial recovery

C1 -> C3
large net regression remains
```

---

## 38. Quantitative Summary

### C2 -> C3

```text
throughput ratio      = 1.058x
throughput change     = +5.77%
epoch-time reduction  = 5.31%
```

### C1 -> C3

```text
throughput ratio      = 0.331x
throughput change     = -66.90%
epoch-time ratio      = 3.025x
epoch-time change     = +202.55%
```

### C2 -> C3 memory

```text
allocated memory change
= +0.820 MB
= approximately +0.31%

mean reserved memory change
= +44 MB
= approximately +11.89%
```

---

## 39. Why the Negative Overall Result Is Still Useful

C3 demonstrates another important systems principle:

> More sophisticated concurrency machinery does not automatically improve end-to-end training.

A dedicated stream can be implemented correctly and still fail to outperform a simpler pipeline.

The result depends on whether there is meaningful work to overlap and whether the added orchestration overhead is justified.

The project therefore reports:

- the partial C3 recovery relative to C2
- the large remaining regression relative to C1
- the lack of profiler-proven causality at this stage

This is more scientifically useful than reporting only the favorable comparison.

---

## 40. Phase 12 Learning Summary

Phase 12 establishes the following concepts.

### CUDA streams

A CUDA stream is an ordered queue of GPU work.

### Same-stream ordering

Operations within one stream preserve ordering and therefore do not automatically execute concurrently.

### Multiple streams

Different streams can create concurrency opportunities when dependencies and hardware resources permit.

### Prefetching

The next batch can be moved toward the GPU while the current batch is being processed.

### Dependency synchronization

The compute stream must not consume a prefetched batch before its transfer is complete.

### `wait_stream`

Stream dependencies can be enforced without globally synchronizing all GPU work.

### Tensor lifetime

Asynchronous execution means tensor storage may still be in use after normal Python control flow moves on.

### `record_stream`

Tensor storage can be associated with stream usage so the allocator does not recycle it prematurely.

### Overlap is not guaranteed

Separate streams create an opportunity for overlap, not proof of overlap.

### Profiling is required

Nsight Systems and PyTorch Profiler are needed to verify whether H2D transfers and compute kernels actually overlap.

---

## 41. Interview-Level Explanation

A concise technical explanation of C3 is:

> C3 extended the C2 pinned-memory/non-blocking pipeline with a dedicated CUDA transfer stream. The prefetcher moved the next batch on that stream while the compute stream processed the current batch, with stream dependencies and tensor-lifetime handling required for correctness. Relative to C2, the configuration improved throughput by about 5.8% and reduced epoch time by about 5.3%. However, C3 was still about 67% slower in throughput than the simpler C1 AMP path. We therefore reported the partial recovery but did not claim that CUDA streams improved the overall training system. We preserved Nsight and PyTorch Profiler traces to verify whether real copy/compute overlap occurred.

---

## 42. Relation to Phase 13

Phase 12 answers:

> What changed when the prefetch stream was enabled?

The answer is:

```text
C3 improved relative to C2
but remained much slower than C1
```

Phase 13 will answer:

> What does the profiler show about why C1, C2, and C3 behave differently?

The next analysis will therefore inspect:

- C1 profiler evidence
- C2 profiler evidence
- C3 profiler evidence
- H2D memcpy activity
- CPU launch gaps
- stream placement
- synchronization
- kernel activity
- possible overlap

This is the correct next step before making a bottleneck claim.

---

## 43. Phase Gate

Before considering Phase 12 conceptually complete, the project owner should be able to explain:

1. What a CUDA stream is.
2. Why operations within one stream preserve order.
3. Why different streams may execute work concurrently.
4. Why multiple streams do not guarantee overlap.
5. What batch prefetching means.
6. Why pinned memory and non-blocking H2D are prerequisites for useful asynchronous transfer.
7. Why the compute stream must wait before consuming a prefetched batch.
8. What `wait_stream()` accomplishes conceptually.
9. Why asynchronous tensor lifetime matters.
10. What `record_stream()` protects against.
11. Why C3 can improve relative to C2 while still being a poor overall configuration.
12. Why the C2-to-C3 timing improvement does not prove overlap.
13. Why profiler evidence is required before claiming H2D/compute concurrency.
14. Why operator or transfer improvements must still be evaluated using end-to-end epoch time.

---

## 44. Phase 12 Evidence Status

| Requirement | Status |
|---|---|
| Pinned-memory path retained | Complete |
| Non-blocking H2D retained | Complete |
| Dedicated CUDA prefetch stream implemented | Complete |
| Stream-based batch preload path implemented | Complete |
| Stream dependency handling implemented | Complete |
| Tensor-lifetime handling implemented | Complete |
| Configuration switch available | Complete |
| Full 20-epoch C3 development run | Complete |
| Forecast-quality consistency verified | Complete |
| End-to-end throughput measured | Complete |
| Steady-state epoch time measured | Complete |
| CUDA allocated memory measured | Complete |
| CUDA reserved memory measured | Complete |
| C2-vs-C3 comparison calculated | Complete |
| C1-vs-C3 context calculated | Complete |
| PyTorch Profiler trace preserved | Complete |
| PyTorch Profiler table preserved | Complete |
| Nsight Systems report preserved | Complete |
| Nsight SQLite export preserved | Complete |
| Nsight stats preserved | Complete |
| Actual copy/compute overlap proven | Deferred to Phase 13 |
| Root-cause bottleneck interpretation | Deferred to Phase 13 |

---

## 45. Phase 12 Conclusion

C3 successfully implemented and executed the CUDA prefetch-stream path on top of the C2 pinned-memory and non-blocking H2D configuration.

The recorded configuration used:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = ON
torch.compile          = OFF
num_workers            = 0
```

The model completed all 20 epochs and preserved the same recorded seed-42 forecast-quality result as C1 and C2.

Relative to C2:

- throughput increased from `3,394.871` to `3,590.779 samples/s`
- throughput improved by approximately **5.77%**
- epoch time decreased from `14.353531` to `13.590975 s`
- epoch time improved by approximately **5.31%**

However, relative to C1:

- C3 throughput was still approximately **66.90% lower**
- C3 achieved only approximately **0.331x C1 throughput**
- C3 epoch time remained approximately **202.55% higher**
- C3 required approximately **3.025x the C1 epoch time**

The C3 result therefore represents a partial recovery of the C2 regression, not an overall system-speedup success.

The correct development conclusion is:

> The dedicated CUDA prefetch stream improved the C2 pinned/non-blocking path modestly, but the complete C3 pipeline remained substantially slower than the simpler C1 AMP configuration.

The timing evidence does not yet prove that H2D transfer overlapped with compute.

That claim is intentionally deferred until the saved Nsight Systems and PyTorch Profiler traces are analyzed in Phase 13.

This preserves the project rule:

> Do not infer concurrency from configuration names or timing alone. Verify it from the execution timeline.

**Phase 12 C3 development benchmark status: COMPLETE, WITH OVERLAP INTERPRETATION DEFERRED TO PHASE 13.**
