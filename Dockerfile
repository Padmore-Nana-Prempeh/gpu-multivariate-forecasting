# syntax=docker/dockerfile:1

# Pre-18: Reproducible NVIDIA CUDA development environment.
# Build target: linux/amd64.
#
# Multi-platform registry digest verified on 2026-11-05.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=8.6

# Python 3.12, CUDA extension compilation, and build utilities.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

# Use a virtual environment rather than system Python.
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"


# Preserve the exact versions from the validated AMD64 image.
COPY requirements/ /opt/requirements/

# Install the archived CUDA-enabled PyTorch build.
# The constraints also apply to PyTorch's dependencies.


RUN python -m pip install --no-cache-dir \
    --constraint /opt/requirements/gpu-resolved.lock \
    "torch==2.8.0+cu128" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install direct project dependencies while enforcing
# the versions of all resolved transitive dependencies.
RUN python -m pip install --no-cache-dir \
    --constraint /opt/requirements/gpu-resolved.lock \
    -r /opt/requirements/gpu-direct.txt


WORKDIR /workspace

# Install the forecasting package without replacing pinned packages.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install \
    --no-deps \
    --no-build-isolation \
    -e .

# Copy experiment definitions, scripts, and validation code.
COPY configs/ ./configs/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY cuda/ ./cuda/

# Verify build-time software compatibility.
# These checks do not require a physical NVIDIA GPU.
RUN python -c 'import sys; assert sys.version_info[:2] == (3, 12); print(sys.version)'

RUN python -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__; assert torch.version.cuda == "12.8"; print("PyTorch:", torch.__version__)'

RUN nvcc --version

RUN python -m pip check

RUN python scripts/verify_gpu_environment.py \
    /opt/requirements/gpu-resolved.lock

# Preserve installed package versions in the image.
RUN python -m pip freeze \
    > /opt/gpuforecast-installed-packages.txt

# Default behavior is the safe 60-run dry-run.
# Official GPU execution requires separate authorization.
CMD ["python", "scripts/run_official_matrix.py"]
