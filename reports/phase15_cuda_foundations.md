# Phase 15: CUDA Programming Foundations and the First PyTorch CUDA Extension

## GPU-Accelerated Deep Learning for Multivariate Time-Series Forecasting

**Phase:** 15  
**Topic:** CUDA Programming Foundations, Kernel Launching, and PyTorch C++/CUDA Extensions  
**Hardware:** NVIDIA A40  
**Compute Capability:** 8.6  
**CUDA Toolkit:** 12.8.93  
**PyTorch:** 2.8.0+cu128  
**Python:** 3.12.3  
**Operation:** Elementwise multiplication by two  
**Source Recovery:** Complete  
**Original Source Verification:** Complete  
**Python Test Syntax Validation:** Passed  
**GPU Runtime Revalidation:** Pending Dockerized NVIDIA GPU execution  
**Documentation:** Complete

---

# 1. Executive Summary

Phase 15 introduces low-level CUDA programming and demonstrates how a custom GPU kernel can be integrated into PyTorch through a C++/CUDA extension.

The project had previously used PyTorch to implement and train four deep learning architectures:

1. LSTM
2. GRU
3. Temporal Convolutional Network (TCN)
4. Transformer

Phases 9-14 investigated GPU execution strategies involving FP32, Automatic Mixed Precision, pinned host memory, non-blocking host-to-device transfers, CUDA prefetch streams, profiling, and `torch.compile`.

Phase 15 transitions from configuring PyTorch execution to understanding how GPU kernels are implemented directly.

The initial CUDA operation is:

```math
y_i=2x_i
```

Although mathematically simple, this operation introduces several important GPU systems concepts:

- CUDA threads, blocks, and grids.
- GPU thread indexing.
- Device memory access.
- CUDA kernel launches.
- Execution streams.
- C++ templates.
- PyTorch tensor integration.
- CUDA extension compilation.
- Numerical correctness testing.
- Asynchronous execution and error checking.

The original CUDA extension source was recovered from the archived NVIDIA A40 development session.

The recovered implementation includes:

```text
binding.cpp
trivial_kernel.cu
setup.py
environment.txt
```

The source was copied without modification into the Git repository under:

```text
cuda/phase15/
```

A reusable correctness-test script was subsequently added:

```text
test_scale2.py
```

The new test script passed Python syntax validation on the local Mac.

However, the archived evidence does not include a separately preserved Phase 15 GPU correctness-test log.

Consequently, this report distinguishes recovered implementation evidence from GPU tests that remain to be re-executed.

The major Phase 15 outcome is a source-preserved, explainable, testable first CUDA extension ready for future validation inside the project's planned Dockerized NVIDIA GPU environment.

---

# 2. Connection to the GPU Optimization Project

The overall project investigates both forecasting quality and GPU execution efficiency.

The forecasting architectures are:

```text
LSTM
GRU
TCN
Transformer
```

The original GPU optimization progression was:

```text
Phase 9
C0: FP32 CUDA baseline
        |
        v
Phase 10
C1: Automatic Mixed Precision
        |
        v
Phase 11
C2: Pinned host memory
    + non-blocking H2D
        |
        v
Phase 12
C3: CUDA prefetch stream
        |
        v
Phase 13
PyTorch Profiler
+ NVIDIA Nsight Systems
        |
        v
Phase 14
C4: torch.compile
        |
        v
Phase 15
Custom CUDA programming foundations
```

Phases 9-14 primarily examined GPU execution through the PyTorch training framework.

Phase 15 examines the mechanics of writing and launching GPU operations directly.

This knowledge is necessary for the more advanced fused LayerNorm+GELU kernel developed in Phases 16-17.

---

# 3. Objective

The main goal of Phase 15 is:

> Understand and preserve a working CUDA extension architecture that connects PyTorch tensors to a custom CUDA kernel.

The specific learning objectives are:

1. Explain CPU and GPU execution differences.
2. Understand CUDA thread-level parallelism.
3. Explain grids, blocks, threads, and warps.
4. Calculate global CUDA thread indices.
5. Understand kernel launch configuration.
6. Identify global-memory reads and writes.
7. Understand CUDA execution streams.
8. Explain the purpose of PyTorch CUDA device guards.
9. Understand C++ template-based dtype dispatch.
10. Understand Python/C++ binding through pybind11.
11. Explain CUDA extension compilation.
12. Design numerical correctness tests.
13. Distinguish source verification from runtime validation.
14. Preserve the original A40 source for reproducibility.
15. Prepare the extension for future Docker-based GPU validation.

---

# 4. CPU Versus GPU Execution

A CPU and a GPU are optimized for different computational workloads.

A CPU is designed for flexible general-purpose computation and efficient execution of complex control flow.

A GPU is designed for high-throughput parallel execution.

A GPU can execute many threads that perform related operations on different data elements.

This makes GPUs especially suitable for operations such as:

- Matrix multiplication.
- Vector operations.
- Neural network layers.
- Convolutions.
- Elementwise activation functions.
- Reductions.
- Attention computation.

Consider a CPU implementation:

```python
x = [1, 2, 3, 4, 5]

y = []

for value in x:
    y.append(value * 2)
```

The result is:

```text
[2, 4, 6, 8, 10]
```

In CUDA, many threads can perform the same elementwise operation on different elements.

Conceptually:

```text
Thread 0 -> y[0] = 2*x[0]
Thread 1 -> y[1] = 2*x[1]
Thread 2 -> y[2] = 2*x[2]
Thread 3 -> y[3] = 2*x[3]
Thread 4 -> y[4] = 2*x[4]
```

This is data parallelism.

CUDA provides a programming model for expressing that parallel work.

---

# 5. CUDA Execution Hierarchy

CUDA organizes thread execution into:

```text
Grid
 |
 +-- Block 0
 |    +-- Thread 0
 |    +-- Thread 1
 |    +-- ...
 |
 +-- Block 1
 |    +-- Thread 0
 |    +-- Thread 1
 |    +-- ...
 |
 +-- Block 2
      +-- Thread 0
      +-- Thread 1
      +-- ...
```

The three main levels are:

## 5.1 Thread

A thread executes the GPU kernel instructions on its assigned data.

## 5.2 Block

A block is a group of CUDA threads.

Threads within a block can cooperate through shared memory and block-level synchronization.

## 5.3 Grid

A grid contains all blocks launched for one CUDA kernel invocation.

The hardware schedules blocks onto GPU streaming multiprocessors.

The CUDA program must not assume a particular block execution order.

---

# 6. CUDA Warps

NVIDIA GPUs organize threads into warps.

A warp normally contains:

```text
32 threads
```

For example, a block containing:

```text
256 threads
```

contains:

```math
256/32=8
```

warps.

The GPU follows a Single Instruction, Multiple Threads execution model.

Threads execute instructions across different data elements.

Divergent conditional branches can introduce additional execution cost.

For example:

```cpp
if (i % 2 == 0) {
    y[i] = x[i] * 2;
} else {
    y[i] = x[i] + 1;
}
```

Different threads in the same warp may follow different branches.

This is called warp divergence.

The introductory Phase 15 kernel has a simple execution path and no substantial branch complexity beyond bounds checking.

---

# 7. CUDA Thread Indexing

Our kernel uses one-dimensional indexing.

The fundamental equation is:

```math
i=\text{blockIdx.x}\times\text{blockDim.x}+\text{threadIdx.x}
```

where:

- `blockIdx.x` identifies the block.
- `blockDim.x` gives the block size.
- `threadIdx.x` identifies the thread within the block.

For example:

```text
blockIdx.x  = 2
blockDim.x  = 256
threadIdx.x = 5
```

Then:

```math
i=2(256)+5=517
```

That thread processes element 517.

The global index must be calculated correctly to avoid duplicate work or unprocessed elements.

---

# 8. The Phase 15 Mathematical Operation

The target operation is:

```math
y_i=2x_i
```

for:

```math
i=0,\ldots,n-1
```

where:

- $x_i$ is an input element.
- $y_i$ is its output.
- $n$ is the number of tensor elements.

The reference PyTorch implementation is:

```python
expected = x * 2
```

The CUDA implementation must produce numerically equivalent output within the chosen dtype-appropriate tolerance.

The operation is:

- Elementwise.
- Independent across elements.
- Suitable for one thread per element.
- Free of inter-thread data dependencies.

Therefore shared-memory synchronization is not required in the kernel.

---

# 9. Recovered CUDA Extension Structure

The archived source was recovered from:

```text
local_gpu_artifacts/
  a40_gpu_evidence_complete.tar.gz
```

The original source directory was:

```text
cuda_ext/phase15/
```

The recovered files were:

```text
binding.cpp
trivial_kernel.cu
setup.py
environment.txt
```

The source was copied into:

```text
cuda/phase15/
```

The resulting repository directory contains:

```text
cuda/phase15/
├── binding.cpp
├── environment.txt
├── setup.py
├── test_scale2.py
└── trivial_kernel.cu
```

The first four files preserve the original development artifacts.

The correctness-test script was added during the documentation and source-integration stage.

---

# 10. Source Integrity Verification

The recovered files were copied from the local evidence-extraction directory into the source tree.

The following comparison was executed:

```bash
diff -r \
  local_gpu_artifacts/phase15_analysis/cuda_ext/phase15 \
  cuda/phase15
```

Before the new test script was added, the command produced no differences.

Therefore the four recovered source/environment files matched their archived copies.

This is important because the project should not silently rewrite the original A40 source during documentation.

The kernel source is preserved as historical development evidence.

---

# 11. Recovered CUDA Kernel Source

The original `trivial_kernel.cu` contains:

```cpp
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void scale2_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t n
) {
    int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i < n) {
        out[i] = static_cast<scalar_t>(2) * x[i];
    }
}

torch::Tensor scale2_cuda(torch::Tensor x) {
    c10::cuda::CUDAGuard device_guard(x.device());

    auto out = torch::empty_like(x);
    const int64_t n = x.numel();

    const int threads = 256;
    const int blocks =
        static_cast<int>((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(),
        "scale2_cuda",
        [&] {
            scale2_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    n
                );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
```

This file contains two distinct execution components:

1. A GPU kernel named `scale2_kernel`.
2. A host-side launcher named `scale2_cuda`.

That distinction is important.

Not every function contained in a `.cu` file runs on the GPU.

---

# 12. CUDA Header Files

The source includes:

```cpp
#include <torch/extension.h>
```

This provides PyTorch C++ extension interfaces and tensor-related functionality.

The source also includes:

```cpp
#include <c10/cuda/CUDAGuard.h>
```

This supports device-context management.

```cpp
#include <ATen/cuda/CUDAContext.h>
```

provides CUDA context-related functions, including access to the current CUDA stream.

```cpp
#include <c10/cuda/CUDAException.h>
```

provides CUDA error-checking facilities.

These headers demonstrate that our kernel is integrated with PyTorch's CUDA runtime environment.

---

# 13. C++ Template-Based Kernel Design

The kernel is declared as:

```cpp
template <typename scalar_t>
__global__ void scale2_kernel(...)
```

The template parameter:

```cpp
scalar_t
```

represents the element type.

This allows the same kernel implementation to be compiled for supported floating-point types.

Rather than maintaining separate source functions for every dtype, the launcher selects the appropriate specialization.

The recovered implementation supports dispatch for:

```text
FP16
FP32
FP64
```

It does not automatically support every PyTorch dtype.

In particular, the dispatch macro used here does not include BF16.

---

# 14. CUDA Pointer Parameters

The kernel receives:

```cpp
const scalar_t* __restrict__ x
```

and:

```cpp
scalar_t* __restrict__ out
```

These are pointers to device-accessible tensor data.

The input is declared `const` because the kernel should not modify it.

The output pointer identifies where the kernel writes results.

The `__restrict__` qualifier expresses non-aliasing assumptions that may permit compiler optimizations.

Because `torch::empty_like(x)` creates a separate output allocation for this operation, the source follows an out-of-place computation design.

---

# 15. The Global Thread Index

The CUDA kernel calculates:

```cpp
int64_t i =
    static_cast<int64_t>(blockIdx.x) * blockDim.x
    + threadIdx.x;
```

This formula maps each CUDA thread to a flat tensor element.

The 64-bit cast is applied before multiplication, avoiding unwanted 32-bit arithmetic in the global index calculation.

This matters when working with large element counts.

The thread index determines which input and output locations the thread accesses.

---

# 16. Bounds Checking

The kernel contains:

```cpp
if (i < n) {
    out[i] = static_cast<scalar_t>(2) * x[i];
}
```

This ensures that threads outside the valid tensor range do not access invalid addresses.

Suppose:

```text
n = 1,000
threads per block = 256
```

The required number of blocks is:

```math
\left\lceil1000/256\right\rceil=4
```

The launch creates:

```text
4 * 256 = 1,024 threads
```

Therefore:

```text
1,000 threads have valid elements.
24 threads are outside the input range.
```

The bounds check prevents the additional threads from reading or writing invalid tensor positions.

---

# 17. Global Memory Access

Each valid thread performs:

```cpp
out[i] = static_cast<scalar_t>(2) * x[i];
```

Conceptually:

1. Read `x[i]` from device memory.
2. Multiply it by two.
3. Write the result to `out[i]`.

The operation has no need for shared-memory cooperation.

Neighboring threads access neighboring elements in contiguous tensors.

That provides a straightforward memory-access pattern that is suitable for efficient global-memory transactions.

However, no measured memory-bandwidth utilization is claimed from the Phase 15 source alone.

---

# 18. PyTorch CUDA Device Guard

The launcher begins with:

```cpp
c10::cuda::CUDAGuard device_guard(x.device());
```

This ensures that the active CUDA device matches the input tensor's device.

This is important for systems with multiple GPUs.

Without correct device selection, an operation could launch in the wrong device context.

The device guard is part of safely integrating external CUDA kernels with PyTorch.

---

# 19. Output Tensor Allocation

The launcher allocates:

```cpp
auto out = torch::empty_like(x);
```

This creates an uninitialized output tensor with properties corresponding to the input.

The kernel is expected to write every valid output element.

Because the output is fully populated for nonempty supported inputs, a prior zero-initialization step is unnecessary.

This avoids extra initialization work.

The operation is out-of-place:

```text
Input x remains unchanged.
Output out contains 2*x.
```

---

# 20. Number of Tensor Elements

The launcher uses:

```cpp
const int64_t n = x.numel();
```

This returns the total number of elements.

For example:

```text
Input shape:
[256, 96, 7]
```

Then:

```math
n=256\times96\times7=172032
```

The kernel processes the tensor through flat indexing.

Therefore the operation can support different tensor shapes, provided the layout requirements are satisfied.

---

# 21. Kernel Launch Configuration

The launcher uses:

```cpp
const int threads = 256;

const int blocks =
    static_cast<int>((n + threads - 1) / threads);
```

This chooses 256 threads per block.

The expression:

```cpp
(n + threads - 1) / threads
```

performs ceiling division for the relevant nonnegative element counts.

For:

```text
n = 172,032
```

the number of blocks is:

```math
172032/256=672
```

Therefore the launch contains:

```text
672 blocks
256 threads per block
172,032 total threads
```

The choice of 256 threads per block is a reasonable starting configuration.

It is not claimed to be universally optimal for every GPU or workload.

---

# 22. Relationship Between Threads and Warps

Each block contains:

```text
256 threads
```

With 32 threads per warp:

```math
256/32=8
```

warps per block.

The NVIDIA A40 schedules work through its streaming multiprocessors.

The runtime and hardware determine how blocks and warps are scheduled.

The kernel source does not explicitly control which streaming multiprocessor executes each block.

---

# 23. Current CUDA Stream

The launcher obtains:

```cpp
cudaStream_t stream =
    at::cuda::getCurrentCUDAStream();
```

This is a particularly important design choice.

It uses the CUDA stream currently associated with PyTorch execution.

The kernel can therefore participate in the caller's stream ordering.

This connects directly to Phase 12, where we investigated CUDA streams and prefetching.

A CUDA stream imposes ordering among operations submitted to that stream.

Using PyTorch's current stream helps the custom extension cooperate with normal tensor operations.

The function does not independently create a new prefetch stream.

It adopts the current execution stream.

---

# 24. Floating-Point Dispatch

The launcher uses:

```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(),
    "scale2_cuda",
    [&] {
        ...
    }
);
```

This selects a kernel specialization based on the input tensor dtype.

The relevant supported types include:

```text
torch.float16
torch.float32
torch.float64
```

The operation is not intended for arbitrary integer or BF16 inputs without extending dispatch support.

The Python-facing binding does not explicitly check dtype before the launcher.

Unsupported dtypes are rejected through the dispatch mechanism.

---

# 25. The Actual CUDA Kernel Launch

The launch expression is:

```cpp
scale2_kernel<scalar_t>
    <<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n
    );
```

The CUDA execution configuration contains four fields:

```text
<<<
  grid size,
  block size,
  dynamic shared-memory bytes,
  stream
>>>
```

For this implementation:

```text
Grid size:
ceil(n/256)

Block size:
256 threads

Dynamic shared memory:
0 bytes

Stream:
Current PyTorch CUDA stream
```

The kernel receives pointers to input and output tensor storage, along with the element count.

This is where the host launches GPU execution.

---

# 26. CUDA Error Checking

After launching the kernel, the source uses:

```cpp
C10_CUDA_KERNEL_LAUNCH_CHECK();
```

This checks the CUDA launch-error state.

It can identify immediate launch-related errors.

However:

> A launch check does not prove numerical correctness.

Furthermore, CUDA execution is generally asynchronous with respect to the host.

A successful launch does not necessarily mean all device instructions have completed.

Runtime tests must account for synchronization when checking results or investigating asynchronous failures.

---

# 27. CUDA Asynchronous Execution

CUDA kernel launches are generally asynchronous from the CPU's perspective.

Conceptually:

```text
CPU:
Launch kernel
Continue execution

GPU:
Execute kernel
Complete output
```

This is why GPU timing requires careful synchronization.

For correctness checks, PyTorch operations can establish dependencies through stream ordering.

Explicit synchronization is also useful when tests need to ensure completion before measuring results or reporting errors.

The Phase 15 test suite includes synchronization for the normal correctness cases and for the separate-stream case.

---

# 28. C++ Binding Source

The original `binding.cpp` contains:

```cpp
#include <torch/extension.h>

torch::Tensor scale2_cuda(torch::Tensor x);

torch::Tensor scale2(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    return scale2_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale2", &scale2, "Scale tensor by 2 (CUDA)");
}
```

This provides the Python-facing interface.

The file performs input validation and calls the CUDA launcher.

---

# 29. CUDA Tensor Validation

The binding checks:

```cpp
TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
```

This rejects CPU tensors.

The kernel expects device-accessible tensor storage.

Without this validation, passing CPU memory to a CUDA kernel could produce invalid behavior.

The binding also checks:

```cpp
TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
```

This ensures that the tensor layout is compatible with the kernel's flat indexing.

The implementation does not support arbitrary strided noncontiguous tensors directly.

---

# 30. Why Tensor Contiguity Matters

A contiguous tensor stores its logical elements in a compatible sequential memory layout.

For example:

```python
x = torch.randn(4, 7, device="cuda")
```

is ordinarily contiguous.

But:

```python
y = x.transpose(0, 1)
```

usually creates a noncontiguous view.

Its logical element order and underlying memory strides differ.

A kernel that assumes flat sequential storage may not preserve the intended element ordering for arbitrary strided views.

The recovered binding therefore rejects noncontiguous inputs.

A future implementation could support noncontiguous tensors by:

1. Converting them to contiguous form.
2. Implementing stride-aware indexing.

Neither approach is part of the preserved Phase 15 kernel.

---

# 31. Python Binding Through pybind11

The source contains:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale2", &scale2, "Scale tensor by 2 (CUDA)");
}
```

This exposes the C++ function to Python.

After compilation, the intended Python interface is:

```python
import trivial_cuda_ext

y = trivial_cuda_ext.scale2(x)
```

The module name is coordinated through the extension build configuration.

This demonstrates the bridge:

```text
Python
   |
   v
pybind11
   |
   v
C++ wrapper
   |
   v
CUDA launcher
   |
   v
GPU kernel
   |
   v
PyTorch output tensor
```

---

# 32. Extension Build Configuration

The original `setup.py` contains:

```python
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

setup(
    name="trivial_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="trivial_cuda_ext",
            sources=[
                "binding.cpp",
                "trivial_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
```

The build uses PyTorch's CUDA extension utilities.

The source files include:

```text
binding.cpp
trivial_kernel.cu
```

The build creates a Python-loadable extension.

---

# 33. Compiler Optimization Flags

The build passes:

```text
-O3
```

to the C++ and CUDA compilation processes.

This enables high-level compiler optimizations.

The CUDA compilation also uses:

```text
-lineinfo
```

This preserves source-line information useful for profiling and diagnostics.

The project does not claim that these flags independently produced a measured speedup.

They are recorded as properties of the original build environment.

---

# 34. Original NVIDIA A40 Environment

The recovered `environment.txt` records:

```text
Python 3.12.3

torch:
2.8.0+cu128

torch cuda:
12.8

GPU:
NVIDIA A40

Compute capability:
(8, 6)

nvcc:
CUDA compilation tools, release 12.8

nvcc build:
V12.8.93
```

Compute capability 8.6 identifies the relevant NVIDIA GPU architecture capability.

The CUDA toolkit version is important for reproducible extension compilation.

The availability of CUDA support inside PyTorch does not alone guarantee that the full CUDA compiler toolchain is installed.

---

# 35. Phase 15 Correctness Requirements

A successfully compiled CUDA kernel is not automatically correct.

Correctness testing should verify:

1. Numerical agreement with PyTorch.
2. Input shape preservation.
3. Input dtype preservation.
4. Output device.
5. Input immutability.
6. Correct indexing at unusual sizes.
7. Correct execution on supported streams.
8. Rejection of unsupported input layouts.
9. Rejection of CPU tensors.
10. Clear reporting of test failures.

The PyTorch reference is:

```python
expected = x * 2
```

The CUDA result is:

```python
actual = trivial_cuda_ext.scale2(x)
```

The results are compared using dtype-appropriate tolerances.

---

# 36. Added Test Suite

The repository now contains:

```text
cuda/phase15/test_scale2.py
```

The test suite is intended to validate the recovered CUDA extension in a compatible NVIDIA GPU environment.

It tests three dtypes:

```text
FP16
FP32
FP64
```

It also tests eight tensor shapes:

```text
(1,)
(17,)
(255,)
(256,)
(257,)
(1000,)
(32, 92)
(256, 96, 7)
```

These shapes include both exact multiples and nonmultiples of the 256-thread block size.

This is important for validating boundary behavior.

---

# 37. Number of Planned Test Cases

The numerical test matrix contains:

```text
3 dtypes × 8 shapes = 24 cases
```

The suite also contains:

```text
1 non-default CUDA stream test
2 input-rejection tests
```

Therefore the total planned cases are:

```math
24+1+2=27
```

The suite contains 27 defined test cases.

This count is a property of the test design.

It does not imply that the tests have already passed on an NVIDIA GPU.

---

# 38. Numerical Tolerances

The test script uses different tolerances depending on dtype.

For FP16:

```python
rtol = 1e-3
atol = 1e-3
```

For FP32:

```python
rtol = 1e-6
atol = 1e-6
```

For FP64:

```python
rtol = 1e-12
atol = 1e-12
```

These tolerances account for differences in floating-point precision.

FP16 has fewer representable significant digits than FP32 or FP64.

Therefore identical tolerances across all three dtypes would not necessarily be appropriate.

The chosen tolerances are part of the test methodology and remain subject to runtime validation.

---

# 39. Testing Input Preservation

The test script saves:

```python
original = x.clone()
```

It then checks:

```python
assert torch.equal(x, original)
```

This verifies that the extension does not mutate the input tensor.

The recovered kernel performs an out-of-place operation.

Therefore preserving the input is part of the expected behavior.

---

# 40. Testing CUDA Stream Compatibility

The test suite creates a non-default CUDA stream:

```python
stream = torch.cuda.Stream()
```

The operation is invoked under:

```python
with torch.cuda.stream(stream):
    ...
```

The test then synchronizes that stream before validating results.

This checks whether the extension behaves correctly when called on a stream other than the ordinary default execution stream.

The design is relevant because the kernel launcher uses:

```cpp
at::cuda::getCurrentCUDAStream()
```

Correct behavior on the current caller stream is an important integration property.

---

# 41. Testing Invalid Inputs

The extension is designed to reject CPU tensors.

The test suite checks that a CPU tensor produces a runtime error.

It also creates a noncontiguous CUDA tensor using a transpose operation.

The test verifies that the extension rejects the noncontiguous input.

These tests exercise the validation conditions in `binding.cpp`.

---

# 42. Local Python Syntax Validation

The new test script was checked on the local Mac using Python's AST parser.

The command was:

```bash
python3 -c '
import ast
from pathlib import Path

path = Path("cuda/phase15/test_scale2.py")

ast.parse(path.read_text())

print("PASS: Phase 15 test script syntax is valid")
'
```

The observed output was:

```text
PASS: Phase 15 test script syntax is valid
```

This confirms that the file contains syntactically valid Python.

However, AST parsing does not:

- Import PyTorch.
- Compile CUDA.
- Load the extension.
- Launch GPU kernels.
- Execute correctness tests.

Therefore the evidence supports syntax validity only.

---

# 43. Archived GPU Correctness Evidence Limitation

The A40 evidence archive was searched for Phase 15 correctness logs.

The search identified:

```text
cuda_ext/phase15/
cuda_ext/phase15/binding.cpp
cuda_ext/phase15/build/
cuda_ext/phase15/environment.txt
cuda_ext/phase15/setup.py
cuda_ext/phase15/trivial_kernel.cu
```

No separate Phase 15 correctness-test output was identified in that search.

The original development work included successful Phase 15 FP16/FP32 validation, but the standalone output was not preserved among the recovered files.

Therefore this report does not invent:

- A numerical maximum error.
- A runtime test duration.
- A new GPU test result.
- A successful 27-case test summary.

The 27-case test suite remains pending execution in the future NVIDIA GPU environment.

---

# 44. Empty-Tensor Edge Case

The recovered launcher calculates:

```cpp
const int blocks =
    static_cast<int>((n + threads - 1) / threads);
```

For an empty tensor:

```text
n = 0
```

the result is:

```text
blocks = 0
```

Launching an ordinary CUDA kernel with a zero-block grid is invalid.

Therefore the recovered implementation does not safely handle zero-element input tensors.

A future hardened implementation should return the empty output without launching a kernel when `n == 0`.

For example:

```cpp
if (n == 0) {
    return out;
}
```

No modification was made to the original archived implementation during Phase 15 source recovery.

The planned correctness test suite intentionally excludes zero-element tensors until a fix can be validated.

---

# 45. Other Implementation Boundaries

The recovered implementation is deliberately small.

Its boundaries include:

### Contiguous tensors only

The binding rejects noncontiguous layouts.

### Supported floating-point types only

Dispatch supports FP16, FP32, and FP64.

### No explicit empty-input handling

A zero-element tensor requires a separate validated handling path.

### No custom backward operator

The extension is a direct Python/C++/CUDA operation and does not provide a custom PyTorch autograd definition.

### No standalone performance campaign

This phase is primarily a CUDA correctness and programming-foundations exercise.

### No fused neural network integration

The extension is not integrated as a replacement for a Transformer training operator.

These boundaries are appropriate for an introductory CUDA extension.

---

# 46. Why Autograd Matters

Our Phase 15 operation performs:

```math
y=2x
```

Mathematically:

```math
\frac{dy}{dx}=2
```

However, mathematical differentiability does not automatically provide a working PyTorch autograd implementation for a custom extension.

A custom operator intended for differentiable training needs an appropriate backward mechanism or autograd registration.

The introductory `scale2` extension does not define one.

Therefore Phase 15 should not be described as a fully differentiable neural network layer.

This becomes especially important when considering fused operations for training in later phases.

---

# 47. Compilation Versus Runtime Correctness

CUDA development requires separating several milestones.

## Source validity

The implementation follows recognizable CUDA/C++ syntax and programming patterns.

## Build success

A compatible CUDA toolchain compiles the extension.

## Import success

Python can load the compiled module.

## Launch success

The kernel can be submitted to the GPU.

## Runtime correctness

The operation produces valid numerical results.

## Performance

The operation meets its intended performance objective.

These stages are not interchangeable.

The archived Phase 15 materials establish the source and original build configuration.

The current local work additionally establishes Python test-script syntax validity.

Future Dockerized GPU validation will provide reproducible current-session evidence for build, import, execution, and numerical correctness.

---

# 48. Relationship to Phase 16

Phase 15 uses a trivial elementwise kernel.

Each thread works independently:

```text
Read one element
Multiply by two
Write one element
```

Phase 16 is more advanced.

The fused LayerNorm+GELU kernel must calculate normalization statistics.

For a feature dimension $D$, LayerNorm uses:

```math
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j
```

and:

```math
\sigma^2=
\frac{1}{D}
\sum_{j=1}^{D}(x_j-\mu)^2
```

The normalized output is:

```math
z_j=
\gamma_j
\frac{x_j-\mu}
{\sqrt{\sigma^2+\epsilon}}
+
\beta_j
```

GELU is then applied:

```math
y_j=\mathrm{GELU}(z_j)
```

Unlike the Phase 15 operation, calculating the mean and variance often requires cooperation among multiple threads.

That introduces:

- Reductions.
- Shared memory.
- Synchronization.
- Memory reuse.
- Potential data races.

The simple indexing and launch concepts learned in Phase 15 are prerequisites for understanding the more advanced fused kernel.

---

# 49. Relationship to the Phase 16 Race Condition

The original fused LayerNorm+GELU development encountered inconsistent numerical results.

The issue was associated with shared-memory synchronization.

Threads were reusing shared-memory storage without adequate synchronization.

This could allow some threads to overwrite values while other threads still needed to read them.

The problem was corrected during the A40 development session.

Phase 16 will inspect the exact recovered implementation and explain the synchronization requirements.

The key distinction is:

> Phase 15 does not require inter-thread synchronization because its elements are independent. Phase 16 requires explicit reasoning about cooperating threads and shared-memory lifetimes.

---

# 50. Why We Are Not Compiling the Extension on the Mac

The original extension was developed in a Linux environment with an NVIDIA A40.

The current local machine is a Mac.

The Mac environment is useful for:

- Source inspection.
- Python syntax validation.
- Documentation.
- Git version control.
- Dockerfile preparation.
- Experiment orchestration preparation.

It is not equivalent to the original NVIDIA CUDA development environment.

Therefore the project does not attempt to reproduce NVIDIA A40 GPU results using an incompatible local execution path.

The extension will be compiled and validated in a compatible Linux/NVIDIA GPU environment.

---

# 51. Future Docker Validation Requirements

Before Phase 18 begins, the project will create a reproducible Docker environment.

That environment must include:

```text
Compatible Linux AMD64 runtime
Pinned Python dependencies
PyTorch with CUDA support
CUDA build toolkit
nvcc
C++ build tools
Compatible NVIDIA GPU runtime
```

The image should reproduce the original A40 software environment as closely as practical.

The original reference versions are:

```text
Python 3.12.3
PyTorch 2.8.0+cu128
CUDA 12.8
nvcc 12.8.93
```

NVIDIA driver compatibility must be verified on the RunPod host.

The same validated container must then be used for the official experiment campaign.

---

# 52. Planned GPU Test Procedure

Once a compatible NVIDIA Docker environment is available, the Phase 15 extension can be compiled.

From the extension directory:

```bash
cd cuda/phase15
```

Build the extension:

```bash
python setup.py build_ext --inplace
```

Run the correctness tests:

```bash
python test_scale2.py
```

The planned test suite contains:

```text
24 numerical cases
1 CUDA stream case
2 invalid-input cases
```

The expected successful summary is:

```text
ALL 27 TEST CASES PASSED
```

This is an expected future output, not an observed current result.

Actual output and failures must be preserved when the tests are executed.

---

# 53. Reproducibility and Artifact Management

The original source and build environment have been preserved.

The source tree intentionally excludes generated build artifacts.

Files such as compiled extension binaries, object files, and temporary build directories should not be committed as primary source artifacts.

The repository should instead preserve:

```text
Source code
Build instructions
Environment requirements
Correctness tests
Documentation
```

This allows another engineer to rebuild and validate the extension in a compatible environment.

The local evidence archive remains separate from the Git-tracked source tree.

---

# 54. Phase 15 Evidence Register

| Evidence item | Status |
|---|---|
| Original CUDA kernel source | Recovered |
| Original C++ binding source | Recovered |
| Original extension build script | Recovered |
| Original A40 environment metadata | Recovered |
| Source copied into repository | Complete |
| Original source comparison | Passed |
| New correctness-test script | Created |
| Test-script Python syntax | Passed |
| FP16 GPU correctness rerun | Pending |
| FP32 GPU correctness rerun | Pending |
| FP64 GPU correctness rerun | Pending |
| Non-default stream GPU test | Pending |
| Invalid-input runtime tests | Pending |
| Zero-element input support | Not implemented |
| New GPU benchmark | Not performed |
| Docker-based GPU validation | Pending |
| Phase 15 technical documentation | Complete |

---

# 55. Observation, Evidence, Diagnosis, Decision

## Observation

The project successfully recovered the source of its first CUDA extension from the original A40 evidence archive.

The recovered kernel implements:

```math
y_i=2x_i
```

using one-dimensional CUDA thread indexing.

## Evidence

The archive contains:

```text
binding.cpp
trivial_kernel.cu
setup.py
environment.txt
```

The source was copied into:

```text
cuda/phase15/
```

The original files matched their archived counterparts during source comparison.

The added test script passed Python AST syntax validation.

No standalone Phase 15 GPU correctness-test log was located in the recovered archive.

## Diagnosis

The recovered implementation provides a clear foundational example of PyTorch/C++/CUDA integration.

It demonstrates:

- GPU thread indexing.
- Bounds checking.
- Dtype dispatch.
- CUDA stream integration.
- Output allocation.
- Input validation.
- Extension compilation.

The source is suitable for documenting foundational CUDA programming concepts.

Current-session GPU correctness revalidation remains outstanding.

## Decision

Preserve the recovered source without modification.

Retain the new correctness-test script.

Document current evidence accurately.

Defer runtime revalidation to the mandatory Dockerized NVIDIA GPU environment.

Do not claim that the 27 planned test cases have already passed.

---

# 56. Interview-Level Explanation

A concise professional explanation of Phase 15 is:

> I developed a PyTorch C++/CUDA extension implementing an elementwise scaling operation. The kernel uses one-dimensional CUDA thread indexing and a 256-thread block configuration, with bounds checking for inputs whose element count is not divisible by the block size. The C++ wrapper validates CUDA residency and contiguity, allocates the output tensor, dispatches supported floating-point dtypes, and launches the kernel on PyTorch's current CUDA stream. I preserved the original source and CUDA build environment from our NVIDIA A40 development session and added a 27-case correctness-test suite covering FP16, FP32, FP64, irregular tensor sizes, stream behavior, and invalid inputs. The new test script passed local syntax validation, with GPU execution scheduled for the reproducible NVIDIA Docker environment.

This demonstrates practical familiarity with the interface between high-level PyTorch training and low-level CUDA kernel execution.

---

# 57. Major Technical Lessons

Phase 15 established the following lessons.

### Lesson 1: Threads need unique indices

Correct CUDA indexing ensures each element is processed exactly as intended.

### Lesson 2: Grid dimensions must cover the workload

The number of blocks must be sufficient to cover all valid elements.

### Lesson 3: Bounds checking prevents invalid memory access

Extra threads in the final block must not access locations outside the tensor.

### Lesson 4: Input layout matters

Flat pointer indexing requires a compatible memory layout.

### Lesson 5: Streams affect execution ordering

Custom kernels should interact correctly with the surrounding PyTorch stream context.

### Lesson 6: Device management matters

CUDA device guards help ensure operations execute on the intended device.

### Lesson 7: Dtype support must be explicit

Supporting FP16, FP32, and FP64 does not imply support for every tensor type.

### Lesson 8: Extension compilation requires a toolchain

CUDA-enabled PyTorch and a complete CUDA compilation environment are not identical.

### Lesson 9: Compilation does not establish correctness

Numerical validation is required.

### Lesson 10: Source preservation supports reproducibility

Validated historical source should be distinguished from newly modified code.

### Lesson 11: Asynchronous launches require careful reasoning

Kernel-launch success is not equivalent to completed correct GPU execution.

### Lesson 12: GPU tests must be distinguished from syntax checks

A successful AST parse does not mean the CUDA extension has executed.

---

# 58. Phase 15 Completion Gate

Before beginning Phase 16 documentation, the project should be able to explain:

1. CPU versus GPU computation.
2. CUDA data parallelism.
3. Grids, blocks, and threads.
4. Warps and SIMT execution.
5. Global thread indexing.
6. Ceiling division for block counts.
7. Bounds checking.
8. Global device memory.
9. CUDA stream ordering.
10. PyTorch device guards.
11. C++ template kernels.
12. Dtype dispatch.
13. Tensor contiguity.
14. pybind11 bindings.
15. CUDA extension compilation.
16. Launch-error checking.
17. Numerical correctness testing.
18. The distinction between syntax and runtime validation.
19. The recovered kernel's limitations.
20. Why shared-memory synchronization becomes important in Phase 16.

The conceptual and source-recovery requirements have been addressed.

GPU runtime revalidation remains deferred until the Docker environment is available.

---

# 59. Transition to Phase 16

The next phase is:

**Phase 16: Fused LayerNorm + GELU CUDA Kernel**

The project will inspect the original fused-kernel source recovered from RunPod.

The key topics will include:

- LayerNorm mathematics.
- GELU activation.
- Kernel fusion.
- Thread-block reductions.
- Shared-memory allocation.
- Thread synchronization.
- The original shared-memory race condition.
- The corrected kernel implementation.
- Numerical correctness.
- FP32 and FP16 behavior.
- PyTorch extension integration.

Phase 17 will then examine the saved kernel benchmark results.

The archived development work already contains measured fused-kernel performance.

The focus will be on recovering, verifying, explaining, and documenting that evidence without inventing new GPU measurements.

---

# 60. Final Conclusion

Phase 15 establishes the project's CUDA programming foundation.

The recovered implementation demonstrates how a PyTorch tensor operation can be implemented through a custom C++/CUDA extension.

The kernel computes:

```math
y_i=2x_i
```

using one-dimensional CUDA indexing and a 256-thread block configuration.

The host launcher manages:

- CUDA device selection.
- Output allocation.
- Tensor element counting.
- Grid dimension calculation.
- Current-stream selection.
- Floating-point dispatch.
- Kernel launching.
- Immediate launch-error checking.

The C++ binding validates CUDA residency and tensor contiguity before exposing the operation to Python.

The original A40 source was successfully recovered and copied into the repository without modification.

A new 27-case correctness-test suite was created, and its Python syntax was successfully validated on the local Mac.

The archived evidence did not contain a standalone Phase 15 correctness-test log.

Therefore no new GPU execution or numerical-correctness success is claimed in this report.

The project will revalidate the extension in the Dockerized NVIDIA GPU environment before the final official experiment campaign.

This phase prepares the conceptual and implementation foundation for the more advanced fused LayerNorm+GELU work.

The central engineering principle is:

> A correct GPU kernel requires more than CUDA syntax: it requires valid thread indexing, memory-access discipline, compatible stream execution, explicit dtype handling, appropriate error checks, and independently verified numerical behavior.

**Phase 15 source recovery, conceptual analysis, and documentation: COMPLETE.**

**Phase 15 Dockerized NVIDIA GPU correctness revalidation: PENDING.**
