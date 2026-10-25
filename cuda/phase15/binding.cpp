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
