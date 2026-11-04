from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

setup(
    name="fused_ln_gelu_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_ln_gelu_cuda",
            sources=[
                "binding.cpp",
                "fused_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
