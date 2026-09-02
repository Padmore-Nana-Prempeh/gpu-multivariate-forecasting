#include <torch/extension.h>

// Forward-only microbenchmark operator. Training integration comes after correctness/perf validation.
torch::Tensor fused_ln_gelu_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);

torch::Tensor fused_ln_gelu(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, hidden]");
    TORCH_CHECK(gamma.numel() == x.size(1) && beta.numel() == x.size(1), "bad affine shape");
    return fused_ln_gelu_cuda(x, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ln_gelu, "Fused LayerNorm + GELU forward (CUDA)");
}
