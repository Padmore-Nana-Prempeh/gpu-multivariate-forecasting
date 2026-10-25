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
