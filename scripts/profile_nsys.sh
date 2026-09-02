#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/nsight
nsys profile \
  --trace=cuda,nvtx,osrt \
  --pytorch=autograd-nvtx \
  --output=results/nsight/transformer_amp_prefetch \
  --force-overwrite=true \
  python -m gpuforecast.train \
    --model transformer \
    --amp on \
    --pin-memory on \
    --prefetch-stream on \
    --compile off
