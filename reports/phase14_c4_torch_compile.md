# Phase 14: PyTorch Compilation, Graph Break Analysis, and C4 GPU Optimization

## Project: GPU-Accelerated Deep Learning for Multivariate Time-Series Forecasting

**Phase:** 14  
**Experiment:** C4 - AMP + Pinned Memory + Non-Blocking H2D + CUDA Prefetch + `torch.compile`  
**GPU:** NVIDIA A40  
**Framework:** PyTorch 2.8.0+cu128  
**CUDA Version:** 12.8  
**Python Version:** 3.12.3  
**Experiment Seed:** 42  
**Training Epochs:** 20  
**Model:** Transformer  
**Model Parameters:** 222,072  
**Experiment Status:** Completed  
**Technical Analysis:** Complete  
**Primary Result:** C4 improved steady-state throughput by 17.91% relative to C3, but remained substantially slower than the simpler C0 and C1 configurations.  
**Compiler Finding:** An evaluation-path graph break occurred at the unsupported Transformer encoder operator `aten._transformer_encoder_layer_fwd.default`.  
**Documentation Status:** Finalized for repository submission.

---

# 1. Executive Summary

Phase 14 investigates PyTorch graph compilation as the fifth GPU execution strategy in the project.

Earlier experiments evaluated:

- C0: FP32 reference execution.
- C1: Automatic Mixed Precision (AMP).
- C2: AMP with pinned host memory and non-blocking host-to-device transfers.
- C3: C2 with a dedicated CUDA prefetch stream.

Phase 13 demonstrated that the severe C2/C3 performance regression was associated primarily with host-side framework and orchestration overhead rather than slower Transformer CUDA computation.

This motivated the Phase 14 question:

> Can PyTorch compilation reduce execution overhead, improve GPU training throughput, and reduce memory usage when applied to the C3 Transformer execution path?

To investigate this, C4 enabled `torch.compile` while retaining the C3 model architecture, dataset, training configuration, AMP, pinned memory, non-blocking transfers, and CUDA prefetch mechanism.

The experiment was conducted on an NVIDIA A40 using PyTorch 2.8.0+cu128.

The completed run contained 20 training epochs.

The main results were:

| Metric | C3 | C4 | Change |
|---|---:|---:|---:|
| Mean steady-state epoch time | 13.590975 s | 11.505417 s | -15.35% |
| Mean steady-state throughput | 3,590.779 samples/s | 4,233.859 samples/s | +17.91% |
| Mean peak allocated GPU memory | 266.738 MB | 229.823 MB | -13.84% |
| Mean peak reserved GPU memory | 414 MB | 386 MB | -6.76% |
| Best validation RMSE | 2.035426 | 2.018186 | -0.85% |
| Test RMSE | 2.339474 | 2.395349 | +2.39% |
| Test MAE | 1.258119 | 1.284288 | +2.08% |

C4 improved throughput and allocated-memory efficiency relative to C3.

However, compilation did not eliminate the overall performance regression associated with the more complex input pipeline.

C4 throughput remained approximately:

- 60.98% below C1.
- 63.41% below C0.

The archived compilation diagnostic additionally recorded a graph break involving:

```text
aten._transformer_encoder_layer_fwd.default
```

The recorded traceback places this graph break during model evaluation.

The evidence therefore supports the conclusion that PyTorch compilation improved the C3 execution path but did not completely capture every observed execution path or eliminate all framework overhead.

The final decision is to retain C4 as a controlled configuration for the official multi-seed campaign while preserving the complete compiler diagnostics and numerical results.

---

# 2. Relationship to Previous Phases

The GPU systems track follows a controlled progression.

```text
Phase 9
C0: FP32 CUDA reference
        |
        v
Phase 10
C1: Automatic Mixed Precision
        |
        v
Phase 11
C2: Pinned memory + non-blocking H2D
        |
        v
Phase 12
C3: CUDA prefetch stream
        |
        v
Phase 13
PyTorch Profiler + Nsight Systems
        |
        v
Phase 14
C4: torch.compile
```

Each configuration tests a distinct execution strategy.

The fundamental research principle is:

> Change one declared execution dimension at a time, preserve the experimental environment, measure outcomes, and investigate unexpected behavior using evidence.

Phase 14 is particularly important because it examines compiler-level optimization rather than introducing another explicit data-transfer mechanism.

---

# 3. Research Question

The primary Phase 14 research question is:

> Does enabling `torch.compile` improve steady-state training throughput and GPU memory efficiency for the existing C3 Transformer workload on the NVIDIA A40?

Secondary questions include:

1. How does compilation affect first-epoch execution time?
2. Does the compiled execution path improve later-epoch throughput?
3. Does compilation reduce allocated or reserved CUDA memory?
4. Does compilation preserve acceptable forecasting performance?
5. Does the captured execution contain graph breaks?
6. Which operation causes the recorded graph break?
7. Does the observed graph break occur in training or evaluation?
8. Does compilation overcome the host-side overhead identified in Phase 13?
9. Is C4 faster than the simpler C0/C1 baselines?
10. What is the appropriate engineering decision for the final experiment campaign?

---

# 4. Hypotheses

## Hypothesis H1: Reduced Execution Overhead

Compiling the model may reduce some PyTorch framework-dispatch and GPU kernel-launch overhead through graph-level optimization.

Expected direction:

```text
C4 throughput > C3 throughput
```

## Hypothesis H2: Reduced Memory Overhead

Compiled execution may eliminate some temporary tensors or fuse compatible operations.

Expected direction:

```text
C4 allocated GPU memory <= C3 allocated GPU memory
```

## Hypothesis H3: Compilation Startup Cost

The first compiled execution may be slower because of tracing, code generation, compilation, initialization, and related warm-up effects.

Expected direction:

```text
Initial execution may be slower
than later steady-state executions.
```

## Hypothesis H4: Incomplete Graph Capture

Unsupported operations or Python behavior may create graph breaks.

Consequently, the entire model execution path may not be captured as one uninterrupted compiled graph.

## Hypothesis H5: Limited End-to-End Recovery

Because C4 retains C3's input-pipeline configuration, model compilation may not eliminate the host-side DataLoader, transfer, and synchronization overhead identified in Phase 13.

This creates the possibility that:

```text
C4 > C3
```

while:

```text
C4 < C1
```

in training throughput.

These hypotheses are evaluated against the saved benchmark and compiler evidence.

---

# 5. Experimental Dataset

The project uses the ETTm1 multivariate time-series dataset.

ETTm1 contains 69,680 observations sampled at 15-minute intervals.

Seven variables are modeled:

```text
HUFL
HULL
MUFL
MULL
LUFL
LULL
OT
```

The forecasting task uses:

```text
Historical lookback = 96 time steps
Forecast horizon    = 24 time steps
Number of features  = 7
```

Because each time step represents 15 minutes:

```text
96 steps = 24 hours of history
24 steps = 6 hours of forecasting
```

The model predicts all seven variables over the forecast horizon.

---

# 6. Leakage-Safe Data Processing

The data pipeline uses chronological splitting:

```text
Training   = 70%
Validation = 10%
Testing    = 20%
```

The split preserves time order.

This is necessary because randomly mixing future observations into training would invalidate a forecasting evaluation.

Standardization is fitted using training data only.

The resulting training statistics are applied to validation and test observations.

Sliding windows are constructed so that target observations belong to the appropriate chronological partition.

Context is preserved where appropriate without allowing future targets to enter earlier training partitions.

The model receives tensors shaped approximately:

```text
Input X:
[batch_size, lookback, features]

[256, 96, 7]
```

and predicts:

```text
Output Y:
[batch_size, horizon, features]

[256, 24, 7]
```

The training engine uses a batch size of 256.

The final batch may have fewer observations.

This is relevant to compilation because changing tensor shapes can affect graph specialization and recompilation behavior.

---

# 7. Transformer Architecture

The Phase 14 benchmark uses the existing Transformer implementation.

The archived model configuration contains:

```yaml
model:
  name: transformer
  hidden_size: 92
  num_layers: 2
  dropout: 0.1
  nhead: 4
  tcn_channels: [64, 88, 88, 88, 88]
```

The `tcn_channels` field is present in the shared configuration schema but is not an active Transformer architecture parameter.

The instantiated Transformer contains:

```text
222,072 parameters
```

The model uses an input projection, positional representation, Transformer encoder, and forecasting output mapping.

The encoder uses:

```python
norm_first=True
```

This is important because it appears in a PyTorch warning captured during Phase 14.

The architecture is held fixed when comparing C3 and C4.

---

# 8. Experimental Environment

The completed C4 `result.json` records:

```text
Python version:
3.12.3

Operating system:
Linux 6.8.0-49-generic x86_64

PyTorch:
2.8.0+cu128

CUDA runtime:
12.8

Device:
cuda

GPU:
NVIDIA A40

Git SHA:
25809b3b2d8fede7eec4e369dc60051c68f016ad
```

The recorded Git SHA identifies the frozen benchmark engine used for development experiments.

The environment metadata is part of the saved evidence.

This is critical because compiler behavior may differ with:

- PyTorch version.
- Python version.
- CUDA toolchain.
- GPU architecture.
- Compiler backend.
- Model implementation.
- Input shapes.
- Precision settings.

The results therefore apply to the recorded environment.

They should not be generalized unconditionally to other hardware or software versions.

---

# 9. Complete C4 Configuration

The saved configuration is:

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
  tcn_channels: [64, 88, 88, 88, 88]

train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: true
  prefetch_stream: true
  compile: true
  device: cuda
  non_blocking_h2d: true

output:
  dir: results/cuda_optimized
```

The configuration enables all C3 mechanisms and additionally enables compilation.

The training engine otherwise retains the controlled development setup.

---

# 10. What Is PyTorch Eager Execution?

PyTorch normally uses eager execution.

Under eager execution, Python operations are executed as they are encountered.

Consider:

```python
import torch
import torch.nn.functional as F

def operation(x):
    a = x + 1
    b = a * 2
    return F.gelu(b)
```

Conceptually:

```text
Python
   |
   v
Addition
   |
   v
Multiplication
   |
   v
GELU
   |
   v
Output
```

Each tensor operation invokes the relevant framework functionality.

For GPU execution, this can involve:

1. Python-level execution.
2. Framework operator dispatch.
3. CUDA API calls.
4. GPU kernel launches.
5. GPU execution.
6. Intermediate tensor allocation.
7. Memory reads and writes.

Eager execution is flexible and generally easier to debug.

However, its framework overhead may become important for workloads that execute many relatively small GPU operations.

---

# 11. Why Eager Execution Can Become Expensive

Suppose one training step contains numerous GPU operations.

Each operation may be individually fast.

However, the CPU still needs to coordinate their execution.

Conceptually:

```text
CPU:
Launch A
Launch B
Launch C
Launch D
Launch E

GPU:
Execute A
Execute B
Execute C
Execute D
Execute E
```

The total cost includes more than GPU arithmetic.

Relevant costs can include:

- Python overhead.
- Operator dispatch.
- CUDA kernel launch overhead.
- Intermediate allocations.
- Tensor conversion.
- Memory traffic.
- Stream synchronization.
- DataLoader operations.

For small models, these costs may represent a substantial fraction of total execution time.

This is precisely why Phase 13 identified host-side overhead as an important concern.

---

# 12. What Is `torch.compile`?

`torch.compile` is PyTorch's compilation interface for optimizing eligible model or function execution.

A basic example is:

```python
model = torch.compile(model)
```

The compiler attempts to capture supported operations into computational graphs.

These graphs can then be optimized and compiled for the target execution environment.

Conceptually:

```text
PyTorch Python program
           |
           v
Graph capture
           |
           v
Graph optimization
           |
           v
Backend compilation
           |
           v
Optimized execution
```

Compilation may improve performance through:

- Reduction of framework overhead.
- Kernel fusion.
- Reduced intermediate memory traffic.
- More efficient generated kernels.
- Improved graph-level execution.

However:

> Compilation is an optimization mechanism, not a guarantee of faster execution.

The result depends on the model, available compiler support, execution environment, and workload characteristics.

---

# 13. TorchDynamo

TorchDynamo is responsible for capturing eligible PyTorch operations from Python execution.

It observes execution and attempts to construct computational graphs while preserving the required behavior.

The captured graphs can be passed to downstream optimization and compilation components.

TorchDynamo also uses guards.

Guards encode assumptions about the execution context, such as tensor properties or other relevant conditions.

If assumptions change, the compiled execution path may need another graph specialization.

TorchDynamo is therefore responsible for more than simply recording a static list of operations.

It must also determine when previously captured graphs remain valid.

---

# 14. TorchInductor

TorchInductor is PyTorch's default compilation backend for many `torch.compile` workloads.

It receives graph representations and attempts to generate efficient executable code.

Depending on the workload and hardware, this may involve:

- Kernel fusion.
- Code generation.
- Memory-use optimization.
- Backend-specific execution strategies.
- Optimized scheduling of supported operations.

The exact transformations are not assumed merely because `torch.compile` was enabled.

The project does not claim a particular number of fused kernels or eliminated operations without corresponding compiler or profiler evidence.

---

# 15. Automatic Differentiation and Compilation

Training requires both forward and backward computations.

PyTorch's compilation stack can incorporate AOTAutograd mechanisms to support optimization of differentiated computation.

Conceptually:

```text
Forward computation
        |
        v
Calculate loss
        |
        v
Backward graph
        |
        v
Gradient computation
        |
        v
Optimizer update
```

Compilation can potentially improve portions of the computational work.

However, the amount of successfully captured or optimized work depends on the specific execution path.

The archived Phase 14 evidence does not provide a complete forward/backward graph-coverage report.

Therefore the project does not claim full training-graph compilation.

---

# 16. Kernel Fusion

Kernel fusion combines compatible operations into fewer generated GPU kernels.

Consider:

```python
a = x + 1
b = a * 2
c = torch.relu(b)
```

An eager implementation may execute multiple separate kernels and materialize intermediate results.

Conceptually:

```text
Kernel 1: Add
        |
Intermediate result
        |
Kernel 2: Multiply
        |
Intermediate result
        |
Kernel 3: ReLU
```

A compiler may fuse compatible operations:

```text
Fused kernel
    |
    +-- Add
    +-- Multiply
    +-- ReLU
```

Potential benefits include:

1. Fewer CUDA kernel launches.
2. Reduced framework-dispatch overhead.
3. Fewer intermediate tensor writes.
4. Reduced memory traffic.
5. Improved execution efficiency.

Fusion opportunities depend on correctness constraints, operator support, memory access patterns, and compiler decisions.

No specific C4 fusion count was captured in the archived summary.

Therefore fusion is discussed as a possible mechanism, not a separately verified cause of the measured speedup.

---

# 17. Relationship to Custom CUDA Kernels

Later project phases implement a manually fused LayerNorm+GELU CUDA kernel.

That is related to, but distinct from, `torch.compile`.

## `torch.compile`

PyTorch analyzes model operations and attempts to generate optimized execution automatically.

## Custom CUDA kernel

The developer explicitly defines GPU thread behavior, memory access, arithmetic, reductions, and synchronization.

The two approaches operate at different levels.

They can both reduce overhead, but they are not equivalent.

The project evaluates these techniques separately.

The custom forward-only CUDA microbenchmark from Phases 15-17 is not included as a training-kernel replacement in C4.

---

# 18. Compilation Startup Cost

Compilation can require additional work before efficient steady-state execution begins.

This may include:

- Tracing.
- Graph construction.
- Shape and guard analysis.
- Code generation.
- Backend compilation.
- Kernel preparation.
- Cache population.
- Runtime warm-up.

Consequently, the first compiled execution may take longer than subsequent executions.

Conceptually:

```text
First execution:
Compile + Initialize + Execute

Later execution:
Reuse compiled execution path
```

For sufficiently long jobs, the one-time cost may be recovered through faster subsequent iterations.

For short jobs, compilation overhead can outweigh steady-state savings.

---

# 19. Compilation Break-Even Analysis

A simplified compilation break-even model is:

$$
N_{\text{break-even}}
=
\frac{T_{\text{additional startup}}}
{T_{\text{eager steady state}}-
T_{\text{compiled steady state}}}
$$

where:

- $T_{\text{additional startup}}$ represents additional initial compilation cost.
- $T_{\text{eager steady state}}$ represents ordinary execution time per unit of work.
- $T_{\text{compiled steady state}}$ represents compiled execution time per equivalent unit of work.

This expression applies when compiled steady-state execution is faster.

For example:

```text
Additional compilation cost = 20 s
Eager time per epoch        = 5 s
Compiled time per epoch     = 3 s
```

Then:

$$
N_{\text{break-even}} = 20 / (5-3) = 10
$$

Approximately ten epochs of steady-state savings would recover the additional startup cost.

This is a conceptual example.

It is not a measured break-even estimate for the Phase 14 Transformer.

The archived C4 evidence does not isolate compilation startup time sufficiently to calculate a defensible project-specific break-even value.

---

# 20. Dynamic Shapes and Recompilation

Compiled graphs may be specialized to assumptions about tensor shapes and other execution conditions.

Our normal training input shape is:

```text
[256, 96, 7]
```

However, the final batch may be smaller.

Different input shapes can cause guards to fail or trigger alternative compiled graph specializations.

When a previously compiled graph cannot be reused under the current conditions, recompilation may occur.

Recompilation can introduce additional overhead.

Important distinctions:

- A graph break interrupts graph capture.
- A guard determines whether a compiled graph remains applicable.
- Recompilation constructs another compatible compiled execution path.

These concepts are related but are not interchangeable.

The saved Phase 14 log identifies a specific graph break.

It does not establish a complete recompilation count.

---

# 21. Graph Breaks

A graph break occurs when TorchDynamo cannot continue capturing an execution region into the current graph.

Conceptually:

```text
Python model
      |
      v
Compiled graph A
      |
      v
GRAPH BREAK
      |
      v
Uncaptured execution region
      |
      v
Compiled graph B
```

Possible causes include:

- Unsupported operators.
- Certain Python constructs.
- Data-dependent execution.
- Unsupported tracing behavior.
- Operations that cannot be represented safely in the current graph.

Graph breaks can reduce optimization opportunities.

However, a graph break does not necessarily cause the entire program to fail.

Under ordinary non-`fullgraph` compilation behavior, PyTorch may continue execution and compile supported regions separately.

---

# 22. What Is a Fake Tensor?

PyTorch compiler components may use fake tensors to reason about tensor operations without performing ordinary computation on actual tensor data.

A fake tensor carries relevant metadata, such as:

- Tensor shape.
- Data type.
- Device representation.
- Stride and layout information.

This can support graph tracing and metadata inference.

Conceptually:

```text
Real tensor computation
        |
        v
Compiler analyzes tensor metadata
        |
        v
Fake tensor representation
        |
        v
Infer operation behavior
        |
        v
Construct graph
```

Not every operator necessarily supports the relevant fake-tensor execution path.

When an unsupported operator is encountered, tracing may stop at that point.

This becomes important in the C4 graph-break diagnostic.

---

# 23. How C4 Was Enabled

The training engine contains conditional model compilation.

The relevant implementation pattern is:

```python
if (
    cfg["train"].get("compile", False)
    and hasattr(torch, "compile")
):
    model = torch.compile(model)
```

This detail matters.

The project compiles the model object.

It does not wrap the complete training program, DataLoader, optimizer, data-transfer pipeline, and evaluation orchestration into one unified compiled function.

Therefore compilation does not automatically eliminate all host-side overhead observed in Phase 13.

This provides an important boundary for interpreting C4 performance.

---

# 24. Controlled C3-to-C4 Comparison

The essential difference between C3 and C4 is compilation.

| Setting | C3 | C4 |
|---|---|---|
| Model | Transformer | Transformer |
| Seed | 42 | 42 |
| Batch size | 256 | 256 |
| Lookback | 96 | 96 |
| Forecast horizon | 24 | 24 |
| Hidden size | 92 | 92 |
| Encoder layers | 2 | 2 |
| AMP | Enabled | Enabled |
| Pinned memory | Enabled | Enabled |
| Non-blocking H2D | Enabled | Enabled |
| CUDA prefetch | Enabled | Enabled |
| `torch.compile` | Disabled | Enabled |
| DataLoader workers | 0 | 0 |
| Epochs | 20 | 20 |

This makes C3 the appropriate immediate comparator for C4.

C0 and C1 remain useful broader performance references.

The C3-to-C4 comparison isolates the declared compilation setting within the frozen development configuration.

---

# 25. C4 Run Identification

The archived run identifier is:

```text
transformer_seed42_amp1_pin1_prefetch1_compile1
```

The saved run status is:

```text
completed
```

The run recorded:

```text
20 epochs
222,072 parameters
```

Its result folder is:

```text
results/cuda_optimized/
  transformer_seed42_amp1_pin1_prefetch1_compile1/
```

This folder contains:

```text
best.pt
config.json
environment.json
result.json
status.json
```

These artifacts were preserved from the completed A40 session.

---

# 26. C4 Training History

The saved `result.json` contains 20 epoch records.

Selected training measurements are:

| Epoch | Training time (s) | Throughput (samples/s) | Training loss |
|---|---:|---:|---:|
| 1 | 12.699724 | 3,830.005 | 0.281540 |
| 2 | 11.560482 | 4,207.437 | 0.162362 |
| 3 | 11.510555 | 4,225.687 | 0.144787 |
| 18 | 11.893978 | 4,089.464 | 0.079600 |
| 19 | 12.128075 | 4,010.529 | 0.078020 |
| 20 | 11.507067 | 4,226.968 | 0.076364 |

The first epoch was slower than the second.

The relative difference was approximately:

```text
(12.699724 / 11.560482 - 1) * 100
≈ 9.85%
```

The subsequent displayed epochs generally remained near 11-12 seconds.

This is consistent with the existence of initial warm-up or compilation-related overhead.

However, the epoch-level measurements do not isolate the precise duration of compilation.

The report therefore does not assign the entire first-epoch difference to compilation.

---

# 27. Separate C4 Smoke-Test Evidence

An earlier C4 smoke test recorded approximately:

```text
Epoch 1 = 25.177 s
Epoch 2 = 11.385 s
```

This shows a larger first-to-second epoch difference in that separate test.

The smoke-test measurements are retained as contextual development evidence.

They are not substituted into the completed 20-epoch result.

The two measurements came from different execution runs.

Therefore:

> The smoke-test first epoch is not used as the official first-epoch time of the completed C4 benchmark.

For the completed run, the saved first-epoch time is:

```text
12.699724 s
```

---

# 28. Steady-State Timing Methodology

The development benchmark uses the warm-up-exclusion convention established for the GPU systems comparisons.

The primary steady-state summary averages epochs 2-20.

Epoch 1 is excluded from the steady-state average.

This is appropriate because the first epoch may include initialization, warm-up, or compilation-related activity not representative of subsequent steady-state training.

The reported C4 mean epoch time is:

```text
11.505416896586356 seconds
```

The reported C4 mean throughput is:

```text
4233.858761451711 samples/second
```

These are the development benchmark summary values.

They are not replaced by the profiler's operator-time totals or by a single selected epoch.

---

# 29. C4 Training Performance

The completed C4 benchmark produced:

```text
Mean epoch time:
11.505417 seconds

Mean throughput:
4,233.859 samples/second
```

The model also completed the expected 20-epoch optimization and validation workflow.

These values demonstrate that compiled execution was operational and produced measurable steady-state training performance.

However, the C4 result must be interpreted against the other configurations.

---

# 30. C3 Versus C4 Performance

The C3 benchmark recorded:

```text
Epoch time:
13.590975 seconds

Throughput:
3,590.779 samples/second
```

C4 recorded:

```text
Epoch time:
11.505417 seconds

Throughput:
4,233.859 samples/second
```

The throughput change is:

$$
100
\left(
\frac{4233.858761}{3590.778655} - 1
\right)
\approx 17.91\%
$$

The epoch-time change is:

$$
100
\left(
\frac{11.505417}{13.590975} - 1
\right)
\approx -15.35\%
$$

Therefore:

> Enabling model compilation improved development throughput by approximately 17.91% relative to the otherwise comparable C3 configuration.

This supports the Phase 14 performance hypothesis.

However, the saved aggregate results do not independently identify which compiler transformations were responsible for the entire improvement.

---

# 31. Comparison Against C1 and C0

C4 must also be evaluated against the simpler reference configurations.

| Configuration | Epoch time (s) | Throughput (samples/s) |
|---|---:|---:|
| C0 FP32 | 4.209343 | 11,570.598 |
| C1 AMP | 4.492150 | 10,849.658 |
| C2 Pinned H2D | 14.353531 | 3,394.871 |
| C3 CUDA Prefetch | 13.590975 | 3,590.779 |
| C4 Compiled | 11.505417 | 4,233.859 |

Relative to C1:

```text
C4 throughput ≈ 60.98% lower
```

Relative to C0:

```text
C4 throughput ≈ 63.41% lower
```

Thus C4 successfully improved upon C3 but remained substantially slower than the simplest development execution path.

The result does not support the conclusion that compilation made the complete pipeline globally optimal.

---

# 32. Allocated CUDA Memory

The recorded mean peak allocated CUDA memory values were:

```text
C0 = 576.901 MB
C1 = 265.917 MB
C3 = 266.738 MB
C4 = 229.823 MB
```

Relative to C3:

$$
100
\left(
1 - \frac{229.823}{266.738}
\right)
\approx 13.84\%
$$

Relative to C0:

$$
100
\left(
1 - \frac{229.823}{576.901}
\right)
\approx 60.16\%
$$

Therefore C4 reduced the recorded allocated-memory requirement relative to C3 and C0.

This supports the memory-efficiency hypothesis for this development run.

The reduction may be consistent with compilation-related execution and temporary-tensor optimizations.

However, the archived evidence does not attribute the full memory reduction to a particular compiler transformation.

---

# 33. Reserved CUDA Memory

Allocated and reserved GPU memory are different measurements.

Allocated memory tracks memory currently occupied by PyTorch tensor allocations.

Reserved memory includes memory managed by PyTorch's CUDA caching allocator.

The development summaries recorded approximately:

```text
C3 mean peak reserved = 414 MB
C4 mean peak reserved = 386 MB
```

The relative change is:

```text
-6.76%
```

The complete C4 epoch history also shows that per-epoch peak reserved memory changed during the run.

Examples include:

```text
Epoch 1  = 258 MB
Epoch 2  = 350 MB
Epoch 3  = 354 MB
Epoch 18 = 414 MB
Epoch 19 = 418 MB
Epoch 20 = 422 MB
```

Therefore:

> The 386 MB summary is a mean of recorded per-epoch peak reserved-memory values, not the maximum reserved-memory value reached anywhere in the run.

The displayed epoch-20 peak is 422 MB.

This distinction prevents a misleading memory claim.

---

# 34. Model Training Behavior

The recorded training loss decreased over the course of the experiment.

Selected values:

```text
Epoch 1  = 0.281540
Epoch 2  = 0.162362
Epoch 3  = 0.144787
Epoch 18 = 0.079600
Epoch 19 = 0.078020
Epoch 20 = 0.076364
```

This shows that the model continued learning during the compiled run.

However, declining training loss does not guarantee better generalization.

The best validation performance occurred earlier in the run.

This distinction is essential when interpreting model quality.

---

# 35. Best Checkpoint Selection

The best validation RMSE was:

```text
2.0181860215692384
```

It occurred at:

```text
Epoch 4
```

The model was not selected using the final epoch automatically.

Instead, validation performance determined the selected checkpoint.

This prevents the training process from simply assuming that the last epoch generalizes best.

The checkpoint is preserved as:

```text
best.pt
```

The held-out test evaluation corresponds to the selected model.

---

# 36. Final C4 Forecasting Metrics

The saved selected-checkpoint test metrics are:

```text
Test MSE:
5.737695693969727

Test RMSE:
2.3953487624915346

Test MAE:
1.2842880487442017
```

The best validation RMSE is:

```text
2.0181860215692384
```

These values are preserved exactly from the C4 `result.json`.

The successful completion of the training run does not by itself establish numerical equivalence between compiled and eager execution.

That is evaluated separately through recorded performance comparisons and the eventual multi-seed campaign.

---

# 37. C3 Versus C4 Forecasting Accuracy

The C3 selected-checkpoint metrics were approximately:

```text
Best validation RMSE = 2.035426
Test RMSE            = 2.339474
Test MAE             = 1.258119
```

C4 produced:

```text
Best validation RMSE = 2.018186
Test RMSE            = 2.395349
Test MAE             = 1.284288
```

C4 achieved a slightly better best validation RMSE.

Its test RMSE and MAE were slightly higher.

The relative test RMSE difference was:

```text
+2.39%
```

The relative test MAE difference was:

```text
+2.08%
```

These are measured single-seed differences.

The evidence does not establish that compilation inherently worsens forecasting accuracy.

Possible differences can arise through numerical execution behavior and training dynamics, but the specific cause was not isolated.

The project therefore reports the observed difference without assigning an unsupported mechanism.

---

# 38. Compiler Diagnostic Evidence

The archived compiler diagnostic file is:

```text
profiling/c4_graph_breaks.log
```

The extracted log contains:

```text
37 lines
```

The relevant contents include:

1. A nested-tensor warning.
2. A TensorFloat32 precision warning.
3. A TorchDynamo graph-break diagnostic.
4. A traceback locating the graph break in evaluation.

The log is preserved in the A40 evidence archive.

It provides direct evidence of an important compiler limitation encountered in the C4 execution environment.

---

# 39. Nested-Tensor Warning

The first warning states:

```text
enable_nested_tensor is True,
but self.use_nested_tensor is False
because encoder_layer.norm_first was True
```

This originates from PyTorch's Transformer module.

The model uses:

```python
norm_first=True
```

This corresponds to a pre-normalization Transformer encoder design.

Conceptually:

```text
Input
  |
  v
LayerNorm
  |
  v
Self-Attention
  |
  v
Residual connection
  |
  v
LayerNorm
  |
  v
Feed-forward network
  |
  v
Residual connection
```

The warning indicates that a particular nested-tensor optimization path was not enabled for the observed configuration.

It is not evidence that the run failed.

The completed result confirms that training and evaluation finished.

No architecture change was made merely to suppress the warning.

Changing `norm_first` would alter the controlled model configuration.

---

# 40. TensorFloat32 Warning

The second warning states:

```text
TensorFloat32 tensor cores for float32
matrix multiplication available but not enabled.
```

The NVIDIA A40 is an Ampere-generation GPU.

TensorFloat32 is a hardware-supported execution mode that may accelerate eligible FP32 matrix multiplication operations.

However, enabling it can change numerical precision behavior.

The warning is therefore recorded as a possible future performance-investigation topic.

For the controlled C4 development experiment:

```text
TF32 was not enabled through the suggested
torch.set_float32_matmul_precision('high') setting.
```

The project does not introduce this new precision variable midway through the frozen configuration comparison.

The warning alone does not establish the size of any potential TF32 speedup.

---

# 41. Exact Graph-Break Diagnostic

The saved log reports:

```text
Graph break in user code at:

torch/nn/modules/transformer.py:902
```

The graph-break reason is:

```text
Operator does not support running with fake tensors
```

The developer debug context identifies:

```text
unsupported operator:

aten._transformer_encoder_layer_fwd.default
```

This is the specific operation recorded as interrupting graph capture.

The diagnostic should not be generalized to all Transformer encoder implementations or all PyTorch versions.

It applies to the saved execution environment and captured path.

---

# 42. Understanding the Unsupported Operator

The recorded operator is:

```python
aten._transformer_encoder_layer_fwd.default
```

It is associated with PyTorch's Transformer encoder-layer forward execution path.

During the compiler's tracing process, this operator was encountered in a context where fake-tensor execution was not supported.

As a result, TorchDynamo reported a graph break.

Conceptually:

```text
Compiled model execution
          |
          v
Transformer encoder
          |
          v
Internal encoder-layer forward operation
          |
          v
Fake-tensor tracing limitation
          |
          v
GRAPH BREAK
```

This indicates incomplete graph capture for the observed path.

It does not establish that all earlier or later operations were necessarily uncompiled.

The archived log also does not quantify the total fraction of computation captured into compiled graphs.

---

# 43. Location of the Graph Break

The saved traceback includes the following project-level call sequence:

```text
train.py:427
    main()

train.py:317
    val = evaluate(...)

train.py:180
    pred = model(x)

models.py:112
    z = self.encoder(
        self.pos(
            self.input_proj(x)
        )
    )
```

The call then reaches PyTorch's Transformer implementation.

The graph-break diagnostic identifies:

```text
torch/nn/modules/transformer.py:902
```

The important observation is:

> The recorded graph break occurred while the model was being evaluated, not in a traceback originating from the training forward/backward loop.

This distinction matters.

Training and evaluation can use different Transformer execution paths.

The log therefore demonstrates an evaluation-path graph break.

It does not prove that the training forward path experienced exactly the same graph break.

---

# 44. Why Training and Evaluation Can Differ

PyTorch modules can behave differently depending on training or evaluation mode.

For example:

```python
model.train()
```

activates training behavior.

```python
model.eval()
```

activates evaluation behavior.

Dropout behavior changes between these modes.

Internal optimizations may also follow different eligible paths depending on model state and execution context.

Therefore:

```text
Training execution path
```

and:

```text
Evaluation execution path
```

are not necessarily identical.

The recorded graph break during `evaluate()` should not be attributed automatically to the complete training computation.

---

# 45. Does the Graph Break Mean Compilation Failed?

No.

The C4 run completed:

```text
20 epochs
```

and produced:

```text
best.pt
result.json
status.json
```

A graph break can interrupt a particular graph-capture region without stopping the entire program.

In ordinary non-`fullgraph` compilation, supported regions may still execute through compiled paths.

Conceptually:

```text
Compiled region A
        |
        v
Graph break
        |
        v
Uncaptured execution
        |
        v
Compiled region B
```

Therefore the correct conclusion is:

> The C4 run completed successfully despite recording an evaluation-path graph break.

It would be incorrect to claim:

> No compilation occurred.

It would also be incorrect to claim:

> The entire model was compiled into one uninterrupted graph.

Neither extreme is supported by the archived evidence.

---

# 46. Does the Graph Break Explain All Remaining Slowness?

No.

The archived log identifies a real compilation limitation.

However, the log does not measure the graph break's isolated contribution to total training time.

Furthermore, the recorded graph break occurs during evaluation.

The development epoch-time benchmark primarily measures training execution.

Therefore attributing the entire C4-versus-C1 training gap to this evaluation-path graph break would be unsupported.

The better interpretation is:

1. Phase 13 demonstrated substantial host-side execution overhead in the C3 input path.
2. C4 compiled the model but retained the C3 input pipeline.
3. C4 improved training throughput relative to C3.
4. C4 remained slower than C0/C1.
5. The diagnostic demonstrates incomplete compilation coverage in at least one observed evaluation path.
6. The available evidence does not isolate the exact contribution of each remaining overhead source.

This preserves the correct distinction between observation and causal explanation.

---

# 47. Interpreting the C4 Performance Improvement

The measured improvement is:

```text
C3 throughput = 3,590.779 samples/s
C4 throughput = 4,233.859 samples/s
```

The result is consistent with the expectation that compilation can improve eligible model execution.

Potential contributing mechanisms include:

- Lower operator-dispatch overhead.
- Kernel fusion.
- Reduced intermediate memory traffic.
- Backend-generated optimized kernels.
- Improved execution scheduling.

However, the archived evidence does not isolate these mechanisms individually.

The report therefore does not claim a measured percentage reduction in kernel launches, compiled graph count, or fusion count.

The supported empirical claim is:

> C4 achieved 17.91% higher steady-state throughput than C3 in the recorded seed-42 development benchmark.

---

# 48. Why C4 Remained Slower Than C1

C1 used a simpler AMP execution path.

C4 retained:

```text
Pinned memory
Non-blocking H2D
CUDA prefetch
Additional stream coordination
```

and added compilation of the model.

The Phase 13 profiler evidence had already identified substantial framework/orchestration overhead in the C2/C3 input paths.

Because the C4 implementation compiles the model rather than the entire input-processing and training pipeline, this previous overhead remains relevant.

The data supports:

```text
C1: 10,849.658 samples/s

C3: 3,590.779 samples/s

C4: 4,233.859 samples/s
```

Thus C4 recovered part of the C3 regression but did not eliminate it.

This is consistent with the limits of applying a model compiler to a broader pipeline that also contains non-model overhead.

The exact decomposition of remaining costs was not separately measured for C4.

---

# 49. Performance Versus Accuracy Trade-Off

The C4 experiment illustrates the need to evaluate multiple objectives.

A GPU optimization should not be judged only by training time.

Relevant dimensions include:

1. Forecasting accuracy.
2. Training throughput.
3. Epoch time.
4. GPU allocated memory.
5. GPU reserved memory.
6. Compilation overhead.
7. Numerical behavior.
8. Implementation complexity.
9. Reproducibility.
10. Robustness across seeds.

C4 improved throughput and allocated-memory efficiency relative to C3.

However, the selected-checkpoint test RMSE was slightly higher.

Without the final multi-seed campaign, the project cannot determine whether this numerical difference is systematic.

The result is therefore presented as a measured single-seed trade-off.

---

# 50. Why the Experiment Was Not Modified After the Result

A common temptation after observing that C4 remained slower than C1 would be to alter settings until C4 became the fastest configuration.

That would weaken the controlled experiment.

Possible changes such as:

```text
Changing num_workers
Changing norm_first
Enabling TF32
Changing model hidden size
Changing batch size
Changing compiler mode
Changing the data-transfer implementation
```

would introduce additional experimental factors.

Such investigations could be valuable as separate follow-up studies.

However, they should not be silently incorporated into the original C4 condition.

The frozen C0-C4 development evidence is therefore preserved.

---

# 51. Reproducibility and Environment Consistency

The original A40 benchmark environment used:

```text
Python 3.12.3
PyTorch 2.8.0+cu128
CUDA 12.8
NVIDIA A40
```

The later local Mac evidence-extraction session used:

```text
Python 3.13.7
```

The local Python version did not alter the archived A40 results.

It was only used to read JSON and log files.

However, this difference demonstrates why environment reproducibility is essential.

The official Phase 18 experiment campaign must not depend on whichever Python environment happens to be active on the local development machine.

---

# 52. Docker Requirement Before Phase 18

The project has adopted Docker containerization as a mandatory prerequisite before launching the final 60-run experiment campaign.

The intended sequence is:

```text
Complete Phase 14
        |
        v
Document and reconcile Phases 15-17
        |
        v
Build GPU-compatible Docker image
        |
        v
Freeze software dependencies
        |
        v
Validate experiment configurations
        |
        v
Validate 60-run orchestration locally
        |
        v
Deploy image to RunPod
        |
        v
Verify GPU/runtime compatibility
        |
        v
Run containerized GPU smoke tests
        |
        v
Launch official 60-run experiment
```

The Docker environment should reproduce the validated Python/PyTorch/CUDA software stack as closely as practical.

It should include support for:

- Model training.
- AMP.
- CUDA prefetch execution.
- `torch.compile`.
- CUDA extension compilation where required.
- Checkpoint persistence.
- Reproducible experiment configuration.
- Results and log collection.

The NVIDIA driver is supplied by the GPU host, so host-driver and container-runtime compatibility must be verified.

The experiment outputs must be written to persistent mounted storage.

The Phase 18 campaign will not begin until the container and recovery workflow pass the required preflight tests.

---

# 53. Archived Evidence Locations

The original saved GPU evidence archive is:

```text
local_gpu_artifacts/
  a40_gpu_evidence_complete.tar.gz
```

The Phase 14 analysis copy was extracted to:

```text
local_gpu_artifacts/
  phase14_analysis/
    profiling/
      c4_graph_breaks.log

    results/
      cuda_optimized/
        transformer_seed42_amp1_pin1_prefetch1_compile1/
          best.pt
          config.json
          environment.json
          result.json
          status.json
```

The complete run metadata is obtained from:

```text
result.json
```

The compiler diagnostic is obtained from:

```text
c4_graph_breaks.log
```

The model checkpoint is obtained from:

```text
best.pt
```

These files form the primary Phase 14 evidence.

---

# 54. Local Evidence-Recovery Procedure

No GPU is required to reproduce the analysis of archived results.

From the repository root:

```bash
cd ~/gpu-multivariate-forecasting
```

Create the analysis directory:

```bash
mkdir -p local_gpu_artifacts/phase14_analysis
```

Extract the saved C4 artifacts:

```bash
tar -xzf \
  local_gpu_artifacts/a40_gpu_evidence_complete.tar.gz \
  -C local_gpu_artifacts/phase14_analysis \
  results/cuda_optimized \
  profiling/c4_graph_breaks.log
```

Inspect the extracted files:

```bash
find local_gpu_artifacts/phase14_analysis \
  -type f \
  | sort
```

Locate the C4 result:

```bash
C4=$(find \
  local_gpu_artifacts/phase14_analysis/results/cuda_optimized \
  -name result.json \
  -print -quit)
```

Read the result using Python:

```bash
python3 -m json.tool "$C4" | less
```

Inspect the graph-break log:

```bash
sed -n '1,180p' \
  local_gpu_artifacts/phase14_analysis/profiling/c4_graph_breaks.log
```

These commands read preserved evidence.

They do not retrain the model or change benchmark results.

---

# 55. Phase 14 Quantitative Summary

The final Phase 14 development metrics are:

| Quantity | Recorded value |
|---|---:|
| GPU | NVIDIA A40 |
| Model parameters | 222,072 |
| Seed | 42 |
| Epochs | 20 |
| Best epoch | 4 |
| Best validation RMSE | 2.018186 |
| Test MSE | 5.737696 |
| Test RMSE | 2.395349 |
| Test MAE | 1.284288 |
| Steady-state epoch time | 11.505417 s |
| Steady-state throughput | 4,233.859 samples/s |
| Mean peak allocated memory | 229.823 MB |
| Mean peak reserved memory | 386 MB |
| Throughput improvement versus C3 | +17.91% |
| Epoch-time reduction versus C3 | 15.35% |
| Allocated-memory reduction versus C3 | 13.84% |
| Throughput difference versus C1 | -60.98% |
| Throughput difference versus C0 | -63.41% |

These results are development measurements.

They do not replace the planned official multi-seed experiment.

---

# 56. Observation -> Evidence -> Diagnosis -> Decision

## Observation

C4 improved performance relative to C3.

```text
C3:
13.590975 s/epoch
3,590.779 samples/s

C4:
11.505417 s/epoch
4,233.859 samples/s
```

C4 also reduced mean peak allocated CUDA memory.

However, C4 remained substantially slower than C0 and C1.

## Evidence

The completed C4 `result.json` records:

```text
Status: completed

Epochs: 20

Parameters: 222,072

Best epoch: 4

Best validation RMSE:
2.0181860215692384

Test RMSE:
2.3953487624915346
```

The compiler diagnostic records:

```text
Graph Break Reason:
Operator does not support running with fake tensors

Unsupported operator:
aten._transformer_encoder_layer_fwd.default
```

The traceback places this graph break in the evaluation path.

## Diagnosis

Compiling the model improved the C3 development execution path.

However, model compilation did not eliminate the total cost of the more complex C3 input pipeline.

The archived evaluation-path graph break additionally demonstrates that the captured compiler execution was not uninterrupted across every observed path.

The exact contributions of fusion, graph specialization, launch reduction, and remaining input-pipeline overhead were not isolated by the archived diagnostics.

## Decision

Retain C4 as the fifth controlled GPU execution configuration.

Preserve:

- Benchmark metrics.
- Compiler warnings.
- Graph-break evidence.
- Numerical results.
- Full configuration.
- Environment metadata.
- Selected checkpoint.

Do not silently change the frozen architecture or precision settings to suppress warnings or improve benchmark outcomes.

---

# 57. Engineering Lessons

Phase 14 reinforces several important systems lessons.

### Lesson 1: Compilation is not automatically faster

`torch.compile` may improve a workload, but speedup depends on execution characteristics and compiler support.

### Lesson 2: Benchmark warm-up matters

Initial execution may include compilation and initialization overhead.

Steady-state timing must be separated from startup behavior.

### Lesson 3: Fusion is a mechanism, not a proven result

Enabling compilation does not establish which operations were fused.

Claims about actual fusion require supporting compiler or profiler evidence.

### Lesson 4: Graph breaks are not necessarily fatal

A graph break can interrupt one compiled region while allowing the program to continue.

### Lesson 5: Training and evaluation may follow different paths

The recorded C4 graph break occurred during evaluation.

It should not be assigned automatically to the complete training computation.

### Lesson 6: Optimizing the model does not optimize the entire pipeline

Data loading, transfers, synchronization, and Python orchestration can remain expensive.

### Lesson 7: Precision changes require control

TF32 settings should not be modified silently during controlled comparisons.

### Lesson 8: Memory and throughput are distinct metrics

C4 reduced memory and improved throughput versus C3, but neither metric alone establishes overall superiority across all objectives.

### Lesson 9: Numerical results must be reported

Forecasting RMSE and MAE remain part of the optimization evaluation.

### Lesson 10: Negative and partial results are valuable

C4 improved one configuration without becoming the fastest overall.

That is a useful engineering finding.

---

# 58. Interview-Level Technical Explanation

A concise professional explanation of Phase 14 is:

> After profiling our AMP, pinned-memory, and CUDA-prefetch configurations, we found that the slower configurations were dominated by increased host-side orchestration rather than slower Transformer GPU computation. We then enabled `torch.compile` on the prefetch configuration to investigate whether graph-level optimization could recover performance. On an NVIDIA A40, the compiled Transformer improved steady-state throughput by approximately 17.9% and reduced allocated CUDA memory by approximately 13.8% relative to the otherwise equivalent prefetch configuration. However, it remained slower than our simpler AMP and FP32 baselines. We inspected the archived TorchDynamo diagnostic and identified an evaluation-path graph break involving `aten._transformer_encoder_layer_fwd.default`, which was unsupported in the observed fake-tensor tracing context. We preserved that limitation, kept the original controlled configuration unchanged, and treated the result as a measured workload-specific optimization rather than claiming universal compilation speedups.

This explanation demonstrates:

- PyTorch systems knowledge.
- Compiler fundamentals.
- Experimental design.
- GPU benchmarking.
- Graph-break diagnosis.
- Numerical awareness.
- Evidence-based optimization decisions.

---

# 59. Limitations

This Phase 14 investigation has several limitations.

1. The primary experiment used one development seed.

2. The benchmark used one Transformer architecture.

3. The workload used a specific seven-variable forecasting dataset.

4. The experiment ran on an NVIDIA A40.

5. The software stack was PyTorch 2.8.0+cu128 with CUDA 12.8.

6. The compiler log does not provide a complete graph-coverage report.

7. The available evidence does not isolate a precise fusion count.

8. The available evidence does not isolate total compilation startup time.

9. No controlled numerical-equivalence test between compiled and eager model outputs is documented in this phase.

10. The recorded graph break is located in evaluation.

11. The graph break's individual wall-clock contribution was not measured.

12. The experiment does not determine whether alternative compiler settings would produce better results.

13. The experiment does not establish performance behavior on other GPUs.

14. The result does not replace the planned multi-seed statistical analysis.

These limitations are documented to prevent overgeneralization.

---

# 60. Phase Gate Checklist

| Requirement | Result |
|---|---|
| Understand eager execution | Complete |
| Understand `torch.compile` | Complete |
| Understand TorchDynamo | Complete |
| Understand TorchInductor | Complete |
| Understand kernel fusion | Complete |
| Understand compilation startup cost | Complete |
| Understand graph breaks | Complete |
| Understand fake-tensor tracing | Complete |
| Understand guards and recompilation | Complete |
| C4 experiment executed | Complete |
| C4 completed 20 epochs | Complete |
| Frozen environment recorded | Complete |
| C4 configuration preserved | Complete |
| C4 training history inspected | Complete |
| Best checkpoint identified | Complete |
| Final forecasting metrics recorded | Complete |
| Steady-state timing recorded | Complete |
| Allocated-memory comparison completed | Complete |
| Reserved-memory comparison completed | Complete |
| C3/C4 performance comparison completed | Complete |
| C0/C1/C4 reference comparison completed | Complete |
| Nested-tensor warning documented | Complete |
| TF32 warning documented | Complete |
| Graph-break operator identified | Complete |
| Graph-break traceback interpreted | Complete |
| Evaluation-path limitation documented | Complete |
| Numerical limitations documented | Complete |
| Optimization decision recorded | Complete |
| Docker-before-Phase-18 requirement preserved | Complete |
| Full Phase 14 report prepared | Complete |

---

# 61. Transition to Phases 15-17

Phase 14 completes the C0-C4 GPU execution strategy investigation at the development-benchmark level.

The next project section moves to lower-level CUDA programming.

The planned progression is:

```text
Phase 15
CUDA foundations and PyTorch extensions
        |
        v
Phase 16
Fused LayerNorm + GELU implementation
        |
        v
Phase 17
CUDA kernel correctness and microbenchmark
```

The A40 development session already produced the relevant CUDA extension and microbenchmark evidence.

The remaining work is to:

1. Review the CUDA concepts.
2. Reconcile the tested CUDA source with the repository.
3. Document the discovered synchronization race condition.
4. Preserve correctness measurements.
5. Preserve kernel benchmark measurements.
6. Commit the completed phase reports.
7. Prepare the reproducible Docker environment before Phase 18.

No new GPU rental is required solely to document the saved Phase 15-17 evidence.

---

# 62. Phase 14 Conclusion

Phase 14 investigated whether PyTorch model compilation could improve the slow C3 Transformer execution path.

The experiment used:

```text
NVIDIA A40
PyTorch 2.8.0+cu128
CUDA 12.8
Python 3.12.3
Transformer with 222,072 parameters
Seed 42
20 epochs
```

The completed C4 benchmark achieved:

```text
Mean epoch time:
11.505417 seconds

Mean throughput:
4,233.859 samples/second

Mean peak allocated CUDA memory:
229.823 MB

Best validation RMSE:
2.018186

Test RMSE:
2.395349
```

Relative to C3, C4 improved throughput by:

```text
17.91%
```

and reduced allocated GPU memory by:

```text
13.84%
```

However, C4 did not outperform the simpler C0/C1 development configurations.

The saved TorchDynamo diagnostic identified an evaluation-path graph break involving:

```text
aten._transformer_encoder_layer_fwd.default
```

with the reason:

```text
Operator does not support running with fake tensors.
```

This demonstrates a captured compiler limitation without implying failure of the entire experiment.

The experiment therefore supports a scoped conclusion:

> PyTorch compilation improved the existing prefetch-based Transformer training configuration in the recorded A40 development benchmark, but did not eliminate the broader execution-path regression relative to simpler reference configurations.

The project retains C4 as a valid, controlled experimental condition.

The original performance measurements, numerical results, warnings, and graph-break diagnostics are preserved.

The next development milestones are CUDA source reconciliation and documentation for Phases 15-17, followed by mandatory Docker containerization and validation before the official 60-run campaign.

The central Phase 14 engineering principle is:

> Compiler optimization must be evaluated through measured performance, numerical correctness, graph diagnostics, and workload-specific evidence rather than assuming that compilation automatically produces the fastest execution path.

**Phase 14 C4 development benchmark and compilation analysis status: COMPLETE.**
