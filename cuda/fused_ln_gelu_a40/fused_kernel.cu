#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__
float gelu_tanh(float x) {
    constexpr float k =
        0.7978845608028654f;

    const float x3 = x * x * x;

    return 0.5f * x *
        (
            1.0f
            + tanhf(
                k * (
                    x
                    + 0.044715f * x3
                )
            )
        );
}

__global__ void fused_ln_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int hidden,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shared[];

    const int64_t base =
        static_cast<int64_t>(row) * hidden;

    float local_sum = 0.0f;

    for (
        int col = tid;
        col < hidden;
        col += blockDim.x
    ) {
        local_sum += x[base + col];
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    const float mean =
        shared[0]
        / static_cast<float>(hidden);

    // Every thread must capture the completed mean
    // before shared memory is reused for variance.
    __syncthreads();

    float local_var = 0.0f;

    for (
        int col = tid;
        col < hidden;
        col += blockDim.x
    ) {
        const float diff =
            x[base + col] - mean;

        local_var += diff * diff;
    }

    shared[tid] = local_var;
    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    const float variance =
        shared[0]
        / static_cast<float>(hidden);

    const float inv_std =
        rsqrtf(variance + eps);

    for (
        int col = tid;
        col < hidden;
        col += blockDim.x
    ) {
        const float normalized =
            (x[base + col] - mean)
            * inv_std;

        const float affine =
            normalized * weight[col]
            + bias[col];

        out[base + col] =
            gelu_tanh(affine);
    }
}

torch::Tensor fused_ln_gelu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    c10::cuda::CUDAGuard device_guard(
        x.device()
    );

    auto out = torch::empty_like(x);

    const int hidden =
        static_cast<int>(x.size(-1));

    const int64_t rows =
        x.numel() / hidden;

    constexpr int threads = 128;

    const size_t shared_bytes =
        threads * sizeof(float);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    fused_ln_gelu_kernel
        <<<rows, threads, shared_bytes, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            hidden,
            static_cast<float>(eps)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
