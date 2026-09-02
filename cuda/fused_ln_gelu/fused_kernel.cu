#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
constexpr int THREADS = 256;

__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    float u = kAlpha * (x + kBeta * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

template <typename scalar_t>
__global__ void fused_ln_gelu_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ out,
    int hidden,
    float eps) {

    int row = blockIdx.x;
    const scalar_t* row_x = x + static_cast<long long>(row) * hidden;
    scalar_t* row_out = out + static_cast<long long>(row) * hidden;

    __shared__ float s_sum[THREADS];
    __shared__ float s_sq[THREADS];
    __shared__ float s_mean;
    __shared__ float s_invstd;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = static_cast<float>(row_x[j]);
        local_sum += v;
        local_sq += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sq[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = s_sum[0] / hidden;
        float var = fmaxf(s_sq[0] / hidden - mean * mean, 0.0f);
        s_mean = mean;
        s_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = (static_cast<float>(row_x[j]) - s_mean) * s_invstd;
        v = v * static_cast<float>(gamma[j]) + static_cast<float>(beta[j]);
        row_out[j] = static_cast<scalar_t>(gelu_tanh(v));
    }
}
} // namespace

torch::Tensor fused_ln_gelu_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto out = torch::empty_like(x);
    const int rows = x.size(0);
    const int hidden = x.size(1);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_ln_gelu_cuda", [&] {
        fused_ln_gelu_kernel<scalar_t><<<rows, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), hidden, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
