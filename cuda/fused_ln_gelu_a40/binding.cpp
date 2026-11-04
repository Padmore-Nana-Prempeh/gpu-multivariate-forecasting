#include <torch/extension.h>

torch::Tensor fused_ln_gelu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);

torch::Tensor fused_ln_gelu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "x currently supports float32 only"
    );
    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat32,
        "weight must be float32"
    );
    TORCH_CHECK(
        bias.scalar_type() == torch::kFloat32,
        "bias must be float32"
    );

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");

    const auto hidden = x.size(-1);

    TORCH_CHECK(
        weight.numel() == hidden,
        "weight size must match final dimension"
    );

    TORCH_CHECK(
        bias.numel() == hidden,
        "bias size must match final dimension"
    );

    return fused_ln_gelu_cuda(
        x,
        weight,
        bias,
        eps
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &fused_ln_gelu,
        "Fused LayerNorm + GELU forward CUDA"
    );
}
