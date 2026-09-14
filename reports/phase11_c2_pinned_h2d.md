# Phase 11: C2 Pinned Memory and Non-Blocking H2D Transfer

## Status

**Phase:** 11 - DataLoader, Pinned Memory, and Asynchronous H2D Transfer  
**Execution configuration:** C2 - AMP + Pinned Memory + Non-Blocking H2D  
**Hardware:** NVIDIA A40  
**Development seed:** 42  
**Primary reference configuration:** C1 AMP  
**Benchmark status:** Completed  
**Profiler interpretation:** Deferred to Phase 13  
**Role in project:** Evaluate the host-memory and host-to-device transfer path before introducing a dedicated CUDA prefetch stream

---

## 1. Objective

The purpose of C2 is to investigate the CPU-to-GPU input boundary.

C0 established the FP32 CUDA reference.

C1 introduced Automatic Mixed Precision.

C2 keeps the C1 AMP execution path but changes the way training batches are prepared in CPU memory and transferred to the GPU.

C2 introduces:

- pinned host memory
- non-blocking host-to-device transfer

C2 does not introduce:

- a dedicated CUDA prefetch stream
- torch.compile

The main question is:

> Does page-locked host memory combined with non-blocking H2D transfer improve end-to-end training performance for the same Transformer workload?

The project does not assume that these mechanisms must improve performance.

The result is determined by measurement.

---

## 2. Recorded C2 Configuration

The saved C2 configuration is:

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
  prefetch_stream: false
  compile: false
  device: cuda
  non_blocking_h2d: true

output:
  dir: results/cuda_amp_pinned
```

The configuration source is:

`configs/cuda_amp_pinned.yaml`

The important systems settings are therefore:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = OFF
torch.compile          = OFF
num_workers            = 0
```

C2 is therefore the first configuration in the controlled development sequence that changes the CPU-memory and H2D-transfer path.

---

## 3. Forecasting Workload Held Constant

C2 uses the same forecasting problem as C0 and C1.

The model receives:

`[batch_size, 96, 7]`

and predicts:

`[batch_size, 24, 7]`

For the recorded development run:

```text
batch_size = 256
lookback   = 96
horizon    = 24
features   = 7
```

The focal model remains the Transformer with:

`222,072 parameters`

The systems experiment therefore changes the input-transfer configuration rather than changing the forecasting task or model architecture.

---

## 4. Benchmark Environment

The recorded C2 run used the following environment:

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

The same frozen benchmark engine used for C0 and C1 was used for C2.

This preserves the experimental contract.

The following were held constant:

- dataset version
- chronological split
- train-only scaler statistics
- lookback length
- forecast horizon
- Transformer architecture
- model hyperparameters
- batch size
- optimizer
- learning rate
- weight decay
- gradient clipping
- training budget
- validation-based checkpoint selection
- test-evaluation procedure
- timing protocol
- CUDA-memory instrumentation

This allows the C1-to-C2 comparison to focus on the change in the host-memory/H2D execution path.

---

## 5. Recorded Run and Evidence

The recorded C2 run ID is:

`transformer_seed42_amp1_pin1_prefetch0_compile0`

The primary result artifact is:

`results/cuda_amp_pinned/transformer_seed42_amp1_pin1_prefetch0_compile0/result.json`

The run directory also contains:

```text
best.pt
status.json
environment.json
config.json
result.json
```

The original GPU evidence was preserved after the NVIDIA A40 session in:

`local_gpu_artifacts/a40_gpu_evidence_complete.tar.gz`

The saved C2 profiler evidence includes:

```text
profiling/c2_amp_pin_torch_table.txt
profiling/c2_amp_pin_torch_trace.json
```

Those profiling artifacts are preserved for formal interpretation in Phase 13.

---

## 6. Host Memory Versus Device Memory

Training data initially exists in CPU memory.

The Transformer executes on the GPU.

A training batch therefore has to cross the CPU-to-GPU boundary before the model can use it.

Conceptually:

```text
ETTm1 dataset
      |
      v
PyTorch Dataset
      |
      v
DataLoader
      |
      v
CPU host memory
      |
      | H2D transfer
      v
GPU device memory
      |
      v
Transformer forward
      |
      v
loss
      |
      v
backward
      |
      v
optimizer
```

The CPU is referred to as the **host**.

The NVIDIA GPU is the **device**.

The transfer:

```text
CPU RAM -> GPU VRAM
```

is therefore called a:

**Host-to-Device transfer**

or:

**H2D transfer**

---

## 7. Pageable Host Memory

Normal CPU memory is generally pageable.

The operating system manages pageable memory through its virtual-memory system.

This gives the operating system flexibility over how memory pages are mapped and managed.

Ordinary pageable memory is therefore not necessarily the ideal source for an asynchronous GPU transfer.

At a high level, CUDA may require additional host-side handling or staging before a transfer can be serviced through the GPU transfer path.

Conceptually:

```text
Pageable CPU memory
        |
        v
host-side handling / possible staging
        |
        v
GPU transfer
        |
        v
GPU VRAM
```

This is one reason GPU input pipelines often consider pinned memory.

---

## 8. Pinned Host Memory

Pinned memory is also called:

**page-locked memory**

When host memory is pinned, the operating system cannot move or swap those pages in the normal way while the memory remains pinned.

Conceptually:

```text
normal pageable memory
        |
        | page-lock
        v
stable host buffer
        |
        v
GPU transfer engine
```

In PyTorch, DataLoader-managed pinning is enabled using:

```python
DataLoader(
    dataset,
    batch_size=256,
    pin_memory=True,
)
```

The recorded C2 configuration uses:

```yaml
pin_memory: true
```

The purpose is to provide host buffers that are suitable for efficient CUDA transfer.

---

## 9. Direct Memory Access

Pinned host memory supports efficient DMA-style transfers.

DMA means:

**Direct Memory Access**

At a high level, DMA allows a hardware transfer engine to move data between host memory and device memory without requiring the CPU to manually copy each value itself.

Conceptually:

```text
Pinned CPU memory
       |
       | DMA-capable transfer
       v
GPU device memory
```

Pinned memory therefore provides an important prerequisite for efficient asynchronous host-to-device transfer.

---

## 10. Why Pinned Memory Is Not Free

Pinned memory is useful, but it is not a free optimization.

Page-locked memory is a constrained host resource.

Because the operating system cannot manage pinned pages as flexibly as ordinary pageable memory, excessive pinning can:

- consume host resources
- reduce memory-management flexibility
- introduce additional host-side work
- become inefficient when batches are small
- provide little benefit when H2D transfer is not a major bottleneck
- increase overhead when there is insufficient computation to hide transfer cost

Therefore:

```text
pin_memory=True
```

does not imply:

```text
training will definitely become faster
```

The effect must be measured end to end.

---

## 11. Why DataLoader-Managed Pinning Is Preferred

The project uses PyTorch DataLoader-managed pinning.

This is preferable to manually calling pinning operations throughout the training hot loop.

The DataLoader already owns the responsibility for:

- retrieving samples
- batching samples
- producing CPU tensors
- handing batches to the trainer

Keeping pinning at this boundary gives the pipeline a clear separation of responsibilities.

Conceptually:

```text
Dataset
   |
   v
DataLoader
   |
   | pin batch if configured
   v
Trainer
```

This is cleaner than repeatedly introducing manual host-memory transformations inside the model-training code.

It also makes the behavior controlled through configuration.

---

## 12. Non-Blocking H2D Transfer

C2 additionally enables:

```yaml
non_blocking_h2d: true
```

The trainer therefore moves batches using the non-blocking CUDA transfer path when configured.

Conceptually:

```python
x = x.to(
    device,
    non_blocking=True,
)
```

and similarly for the target tensor.

The purpose of `non_blocking=True` is to allow the transfer operation to be enqueued without unnecessarily forcing the Python host to wait for completion when the runtime conditions permit asynchronous behavior.

---

## 13. Blocking Versus Non-Blocking Behavior

A simplified blocking conceptual model looks like:

```text
CPU
 |
 | launch copy
 |-------------------- wait --------------------|
                                               |
                                               v
                                            continue
```

while the device receives:

```text
GPU
 |
 | H2D copy
 |==================|
```

A potential non-blocking model looks like:

```text
CPU
 |
 | enqueue copy
 |----> continue host-side work -------------------->
```

while:

```text
GPU
 |
 | H2D copy
 |==================>
```

The important point is that `non_blocking=True` relates to asynchronous execution relative to the host.

It does not by itself prove that GPU transfer and GPU computation overlap.

---

## 14. Why Non-Blocking Does Not Automatically Mean Overlap

CUDA operations are submitted to streams.

Operations within one CUDA stream preserve ordering.

Suppose the same stream contains:

```text
H2D COPY
   |
   v
FORWARD
   |
   v
BACKWARD
```

The forward computation cannot consume the new batch until the required transfer is complete.

Therefore, even if the host did not block while submitting the transfer, the GPU operations may still execute sequentially within the stream.

This leads to a critical Phase 11 lesson:

> `non_blocking=True` enables asynchronous submission under the right conditions, but it does not by itself guarantee useful copy/compute overlap.

C3 introduces a separate CUDA prefetch stream specifically to investigate that next level of concurrency.

---

## 15. Why C1 Is the Correct Reference for C2

The primary reference for C2 is C1 rather than C0.

C1 already introduced AMP.

Its execution configuration is conceptually:

```text
AMP                    = ON
pin_memory             = OFF
non_blocking_h2d       = OFF
prefetch_stream        = OFF
compile                = OFF
```

C2 changes this to:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = OFF
compile                = OFF
```

Therefore the C1-to-C2 comparison is the cleanest development comparison for evaluating the host-memory/H2D path.

---

## 16. C2 Training Completion

The C2 run completed successfully.

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

## 17. C2 Forecast-Quality Result

The recorded C2 test metrics were:

| Metric | C2 |
|---|---:|
| Test MSE | 5.473136 |
| Test RMSE | 2.339474 |
| Test MAE | 1.258119 |

These values are identical to the recorded C1 development result.

C1:

```text
Test RMSE = 2.339474
Test MAE  = 1.258119
```

C2:

```text
Test RMSE = 2.339474
Test MAE  = 1.258119
```

This is useful correctness evidence.

The systems execution path changed, but the resulting training trajectory and selected model-quality result remained consistent in this development run.

The C2 experiment therefore provides no evidence that the input-transfer modification changed model semantics.

---

## 18. Steady-State Measurement Protocol

The result summarizer excludes the first epoch from the steady-state aggregate.

C2 therefore reports:

`19 steady-state epochs`

The first epoch remains in the raw training history, but it is not used in the summarized steady-state performance metric.

This is consistent with the measurement protocol used for C0 and C1.

The purpose is to prevent initialization and early runtime effects from dominating the comparison.

---

## 19. C1 Versus C2 Results

The matched development comparison is:

| Metric | C1 AMP | C2 AMP + Pinned H2D |
|---|---:|---:|
| Test RMSE | 2.339474 | 2.339474 |
| Test MAE | 1.258119 | 1.258119 |
| Mean steady-state epoch time | 4.492150 s | 14.353531 s |
| Mean throughput | 10,849.658 samples/s | 3,394.871 samples/s |
| Mean per-epoch peak allocated CUDA memory | 265.917 MB | 265.917 MB |
| Mean per-epoch peak reserved CUDA memory | 370.000 MB | 370.000 MB |

The accuracy result is unchanged.

The systems-performance result is not.

---

## 20. C2 Throughput Result

C1 throughput:

`10,849.658 samples/s`

C2 throughput:

`3,394.871 samples/s`

The throughput ratio is:

```text
3,394.871 / 10,849.658
= 0.312901
```

Therefore C2 achieved approximately:

**0.313x C1 throughput**

The percentage change is:

```text
(0.312901 - 1) x 100
= -68.71%
```

Therefore C2 throughput was approximately:

**68.71% lower than C1**

This is a severe end-to-end performance regression.

---

## 21. C2 Epoch-Time Result

C1 mean steady-state epoch time:

`4.492150 s`

C2 mean steady-state epoch time:

`14.353531 s`

The ratio is:

```text
14.353531 / 4.492150
= 3.195247
```

Therefore C2 required approximately:

**3.195x the C1 epoch time**

The percentage increase is:

```text
(3.195247 - 1) x 100
= +219.52%
```

Therefore C2 steady-state epoch time increased by approximately:

**219.52%**

relative to C1.

This agrees with the throughput result.

---

## 22. Interpretation of the Performance Result

The C2 experiment produced a clear negative end-to-end result.

The expected mechanism was:

```text
pinned memory
      +
non-blocking H2D
      |
      v
potentially more efficient input transfer
      |
      v
potentially lower data-transfer overhead
      |
      v
potentially higher training throughput
```

The observed result was instead:

```text
C1 throughput
10,849.658 samples/s
        |
        v
C2 throughput
3,394.871 samples/s
```

The C2 path therefore regressed substantially.

This is not hidden or removed from the experiment.

It is retained as an important systems result.

---

## 23. Does This Mean Pinned Memory Is Bad?

No.

The result does not justify the universal statement:

> Pinned memory makes training slower.

The supported statement is narrower:

> For the recorded Transformer development workload on the NVIDIA A40, using the C2 configuration with pinned memory, non-blocking H2D, and `num_workers=0` caused a severe end-to-end performance regression relative to C1.

Pinned memory is a valid CUDA input-pipeline mechanism.

Its benefit depends on the workload.

Factors that can influence whether it helps include:

- batch size
- transfer size
- host-memory behavior
- CPU preparation cost
- DataLoader configuration
- worker configuration
- amount of GPU computation
- ability to overlap transfer and computation
- synchronization behavior
- whether H2D transfer was a meaningful bottleneck to begin with

Therefore the correct response to the C2 regression is to profile the system rather than declare the mechanism universally good or bad.

---

## 24. C2 CUDA Memory Result

C1 and C2 recorded the same steady-state GPU-memory summary:

| Metric | C1 | C2 |
|---|---:|---:|
| Mean per-epoch peak allocated CUDA memory | 265.917 MB | 265.917 MB |
| Mean per-epoch peak reserved CUDA memory | 370.000 MB | 370.000 MB |

This is an important distinction.

Pinned memory refers to:

**CPU host memory**

whereas:

`torch.cuda.max_memory_allocated()`

and CUDA reserved-memory statistics refer to:

**GPU device memory**

Therefore:

```text
Pinned memory
!=
CUDA VRAM
```

Enabling pinned host memory should not be interpreted as an attempt to reduce the model's GPU VRAM footprint.

---

## 25. Why GPU Memory Stayed the Same

C2 does not change the Transformer architecture.

It also does not change:

- batch size
- model width
- model depth
- sequence length
- forecast horizon
- AMP configuration
- optimizer
- major activation structure

Therefore the core GPU training tensors remain effectively the same size as in C1.

The pinned-memory modification occurs on the CPU side of the pipeline.

It is therefore consistent that C1 and C2 show the same summarized CUDA allocated and reserved memory.

---

## 26. What the Raw C2 History Shows

The saved result contains all 20 training epochs.

The first epoch recorded:

```text
train_seconds              = 15.597284
train_samples_per_sec      = 3118.492
peak_allocated_memory_mb   = 265.917
peak_reserved_memory_mb    = 334.0
```

Subsequent epochs typically remained in the approximate range of:

```text
13-15.5 seconds per epoch
```

rather than returning to the roughly:

```text
4.49-second
```

C1 steady-state level.

This confirms that the C2 slowdown was not merely a one-time startup artifact.

The regression persisted throughout the run.

---

## 27. What We Can Conclude from C2

The recorded C2 evidence supports the following conclusions:

1. The pinned-memory path executed successfully.
2. The non-blocking H2D path executed successfully.
3. The model completed the full 20-epoch development run.
4. Forecast-quality results remained consistent with C1.
5. CUDA allocated memory remained effectively unchanged relative to C1.
6. CUDA reserved memory remained effectively unchanged relative to C1.
7. End-to-end throughput regressed dramatically.
8. Steady-state epoch time increased dramatically.
9. The regression persisted across the training run rather than appearing only during startup.
10. The C2 execution path was therefore not beneficial for this recorded development workload.

---

## 28. What We Cannot Yet Conclude

The timing numbers tell us **what happened**.

They do not by themselves prove **why it happened**.

Possible hypotheses include:

- host-side pinning overhead
- DataLoader behavior
- synchronization effects
- host-memory handling cost
- H2D transfer not being the original bottleneck
- insufficient GPU computation to amortize transfer overhead
- small batch-transfer sizes
- interaction with `num_workers=0`
- CPU-side preparation dominating the input pipeline
- implementation-specific scheduling behavior

These remain hypotheses.

None of them should be promoted to the definitive cause without profiler evidence.

This distinction is intentional.

The project follows:

```text
observation
    |
    v
measurement
    |
    v
profiling
    |
    v
diagnosis
    |
    v
optimization decision
```

rather than:

```text
observation
    |
    v
guess
```

---

## 29. Why Profiling Is Deferred to Phase 13

Phase 11 answers:

> What did the pinned/non-blocking path do to end-to-end performance?

Phase 13 answers:

> Where did the time go, and what does the profiler show?

The saved C2 profiling evidence is:

```text
profiling/c2_amp_pin_torch_table.txt
profiling/c2_amp_pin_torch_trace.json
```

Those artifacts will be used in Phase 13 to investigate the regression.

This separation prevents the project from making unsupported causal claims.

---

## 30. DataLoader Worker Scope

The recorded C2 configuration uses:

```yaml
num_workers: 0
```

This is important.

The Phase 11 design originally proposed evaluating a small number of DataLoader worker settings.

That worker-count sweep is **not present in the saved C2 development evidence**.

Therefore this report does not claim that different worker counts were tested.

The current result specifically represents:

```text
pin_memory=True
non_blocking_h2d=True
num_workers=0
```

This is a documented limitation of the development benchmark.

It should not be silently rewritten as a broader worker-optimization study.

---

## 31. Why `num_workers=0` Matters

With:

```yaml
num_workers: 0
```

data loading and much of the input preparation occur without separate DataLoader worker processes.

This may affect how much CPU-side work can be prepared concurrently with GPU activity.

However, the current Phase 11 benchmark does not provide enough evidence to state that `num_workers=0` is the cause of the C2 regression.

It is therefore recorded as an experimental condition and scope limitation rather than a causal explanation.

The profiler evidence will be used before assigning a bottleneck.

---

## 32. Scientific Value of the Negative Result

C2 is a useful result precisely because it did not behave as the optimization folklore might suggest.

It would have been easy to assume:

```text
pinned memory
+
non-blocking copy
=
faster training
```

The experiment shows why that equation is incomplete.

The real relationship is closer to:

```text
optimization mechanism
+
specific workload
+
specific hardware
+
specific pipeline
+
specific scheduling behavior
=
measured end-to-end result
```

For this particular combination, the result was negative.

Recording that result increases the credibility of the project because it demonstrates that performance claims are evidence-driven rather than predetermined.

---

## 33. Relationship Between C1, C2, and C3

The execution progression is:

```text
C1
AMP
pageable host path
ordinary transfer configuration
        |
        v
C2
AMP
pinned host memory
non-blocking H2D
        |
        v
C3
AMP
pinned host memory
non-blocking H2D
dedicated CUDA prefetch stream
```

C2 establishes the input-transfer foundation required before C3 can investigate explicit transfer scheduling on another CUDA stream.

C2 is therefore the direct baseline for C3.

---

## 34. Why C2 Does Not Yet Attempt Dedicated Stream Overlap

C2 enables the prerequisites for efficient asynchronous transfer:

- pinned host memory
- non-blocking H2D

But C2 leaves:

```yaml
prefetch_stream: false
```

Therefore C2 does not create the dedicated transfer stream introduced in Phase 12.

This separation is intentional.

It allows the experiment to distinguish:

```text
C2:
pinned + non-blocking transfer
```

from:

```text
C3:
pinned + non-blocking transfer
+ dedicated prefetch stream
```

Without this separation, it would be difficult to determine whether any difference came from pinning, asynchronous submission, or explicit stream scheduling.

---

## 35. Connection to the Controlled Experiment Matrix

The final systems matrix defines:

```text
C0 = FP32

C1 = AMP

C2 = AMP
     + pin_memory
     + non-blocking H2D

C3 = AMP
     + pin_memory
     + non-blocking H2D
     + CUDA prefetch stream

C4 = AMP
     + pin_memory
     + non-blocking H2D
     + CUDA prefetch stream
     + torch.compile
```

C2 therefore represents one controlled step in the progression from a simple AMP training path toward increasingly complex execution behavior.

---

## 36. C2 Versus the Original Optimization Hypothesis

The original hypothesis was that improving the input path might reduce GPU idle time and improve epoch throughput.

The C2 evidence did not support that hypothesis.

Instead:

```text
C1 epoch time
4.492150 s

C2 epoch time
14.353531 s
```

and:

```text
C1 throughput
10,849.658 samples/s

C2 throughput
3,394.871 samples/s
```

The hypothesis is therefore rejected for this specific development configuration.

The project retains the actual result rather than modifying the experiment until the expected result appears.

---

## 37. C2 Development Summary

### C1 reference

```text
epoch time          = 4.492150 s
throughput          = 10,849.658 samples/s
allocated VRAM      = 265.917 MB
reserved VRAM       = 370.000 MB
test RMSE           = 2.339474
test MAE            = 1.258119
```

### C2 result

```text
epoch time          = 14.353531 s
throughput          = 3,394.871 samples/s
allocated VRAM      = 265.917 MB
reserved VRAM       = 370.000 MB
test RMSE           = 2.339474
test MAE            = 1.258119
```

### C1 -> C2 change

```text
throughput ratio        = 0.313x
throughput change       = -68.71%
epoch-time ratio        = 3.195x
epoch-time change       = +219.52%
allocated VRAM change   = approximately 0%
reserved VRAM change    = approximately 0%
test RMSE change        = 0 in recorded values
```

The result is unambiguous:

> C2 preserved model-quality behavior but severely regressed end-to-end training performance relative to C1.

---

## 38. Interview-Level Explanation

A concise technical explanation of C2 is:

> C2 kept the AMP Transformer workload fixed and enabled DataLoader pinned host memory together with non-blocking host-to-device copies. Pinned memory provides page-locked host buffers suitable for efficient DMA transfer, while non-blocking copies allow asynchronous host submission. However, asynchronous submission does not guarantee copy/compute overlap, particularly when work remains ordered on the same CUDA stream. In the seed-42 A40 development benchmark, C2 preserved the same RMSE as C1 but reduced throughput by about 68.7% and increased epoch time by about 219.5%. We therefore treated the slowdown as a profiling question rather than claiming pinning was inherently beneficial or harmful.

---

## 39. Phase 11 Learning Summary

Phase 11 established the following systems concepts.

### Host versus device memory

Training batches begin in CPU host memory and must be transferred into GPU device memory before CUDA computation.

### Pageable memory

Normal host memory is managed flexibly by the operating system.

### Pinned memory

Pinned or page-locked memory remains fixed for the duration of the pin and is suitable for efficient CUDA transfer.

### DMA

Direct Memory Access allows hardware transfer engines to move data without requiring the CPU to manually copy every element.

### Non-blocking H2D

`non_blocking=True` can allow the host to continue after a compatible CUDA copy has been enqueued.

### Non-blocking is not overlap

An asynchronous host call does not prove that transfer and compute are concurrent on the GPU.

### Pinned memory is not GPU memory

Pinned memory refers to CPU RAM.

CUDA allocated and reserved memory refer to GPU VRAM.

### Optimization overhead matters

An optimization can introduce more overhead than benefit for a particular workload.

### Measurement beats assumption

The end-to-end benchmark determines whether an optimization is useful.

---

## 40. Phase Gate

Before considering Phase 11 conceptually complete, the project owner should be able to explain:

1. What host memory is.
2. What GPU device memory is.
3. What an H2D transfer is.
4. The difference between pageable and pinned host memory.
5. Why pinned memory is called page-locked memory.
6. What DMA means conceptually.
7. Why pinned memory can support efficient GPU transfer.
8. Why pinned memory is a constrained host resource.
9. Why DataLoader-managed pinning is preferable to manual pinning in the hot loop.
10. What `non_blocking=True` means.
11. Why non-blocking submission does not guarantee copy/compute overlap.
12. Why work on one CUDA stream remains ordered.
13. Why pinned memory is different from GPU allocated/reserved memory.
14. Why C2 could preserve model accuracy while changing systems performance dramatically.
15. Why the C2 slowdown must be diagnosed with profiler evidence instead of guessed.
16. Why `num_workers=0` is an important scope condition of the recorded result.

---

## 41. Phase 11 Evidence Status

| Requirement | Status |
|---|---|
| Pinned-memory path implemented | Complete |
| `pin_memory=True` recorded | Complete |
| Non-blocking H2D path implemented | Complete |
| `non_blocking_h2d=True` recorded | Complete |
| AMP retained from C1 | Complete |
| Prefetch stream disabled for isolation | Complete |
| Compile disabled for isolation | Complete |
| Full 20-epoch C2 development run | Complete |
| Validation-based checkpoint selection | Complete |
| Held-out test evaluation | Complete |
| Forecast-quality consistency with C1 | Complete |
| Steady-state epoch timing | Complete |
| End-to-end throughput measurement | Complete |
| CUDA allocated memory measurement | Complete |
| CUDA reserved memory measurement | Complete |
| Environment metadata preserved | Complete |
| Config artifact preserved | Complete |
| Checkpoint preserved in GPU evidence archive | Complete |
| PyTorch Profiler table preserved | Complete |
| PyTorch Profiler trace preserved | Complete |
| Benefit/lack-of-benefit documented | Complete |
| Negative result retained transparently | Complete |
| DataLoader worker sweep | Not evidenced |
| Recorded worker configuration | `num_workers=0` |
| Root-cause profiler interpretation | Deferred to Phase 13 |

---

## 42. Phase 11 Conclusion

C2 successfully implemented and executed the pinned-memory and non-blocking H2D input path.

The recorded configuration used:

```text
AMP                    = ON
pin_memory             = ON
non_blocking_h2d       = ON
prefetch_stream        = OFF
torch.compile          = OFF
num_workers            = 0
```

The model completed all 20 epochs and produced the same recorded seed-42 test RMSE and MAE as C1.

However, the systems-performance result was strongly negative.

Relative to C1:

- throughput decreased from `10,849.658` to `3,394.871 samples/s`
- throughput decreased by approximately **68.71%**
- throughput ratio was approximately **0.313x**
- steady-state epoch time increased from `4.492150` to `14.353531 s`
- epoch time increased by approximately **219.52%**
- C2 required approximately **3.195x** the C1 epoch time
- summarized CUDA allocated memory remained effectively unchanged
- summarized CUDA reserved memory remained effectively unchanged
- recorded test RMSE remained `2.339474`
- recorded test MAE remained `1.258119`

The correct development conclusion is therefore:

> For the recorded Transformer workload on the NVIDIA A40, enabling DataLoader pinned memory and non-blocking H2D with `num_workers=0` preserved forecasting behavior but caused a major end-to-end training-performance regression relative to the simpler C1 AMP path.

The timing result establishes the regression.

It does not yet establish the root cause.

The saved PyTorch Profiler evidence will be analyzed in Phase 13 before assigning a causal bottleneck.

This preserves the central project rule:

> Measure first, profile second, explain third, and only then optimize.

**Phase 11 C2 development benchmark status: COMPLETE WITH DOCUMENTED SCOPE LIMITATION.**
