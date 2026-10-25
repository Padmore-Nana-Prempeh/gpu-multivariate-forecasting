from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
