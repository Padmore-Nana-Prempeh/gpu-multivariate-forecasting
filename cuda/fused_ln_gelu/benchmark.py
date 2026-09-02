from __future__ import annotations

import argparse
import time

import fused_ln_gelu_cuda
import torch
import torch.nn.functional as F


def eager(x, gamma, beta, eps):
    y = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
    return F.gelu(y, approximate="tanh")


def bench(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=24576)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=dtype)
    gamma = torch.randn(args.hidden, device="cuda", dtype=dtype)
    beta = torch.randn(args.hidden, device="cuda", dtype=dtype)
    eps = 1e-5

    ref = eager(x, gamma, beta, eps)
    out = fused_ln_gelu_cuda.forward(x.contiguous(), gamma.contiguous(), beta.contiguous(), eps)
    max_err = (ref.float() - out.float()).abs().max().item()

    eager_us = bench(lambda: eager(x, gamma, beta, eps))
    fused_us = bench(lambda: fused_ln_gelu_cuda.forward(x, gamma, beta, eps))
    print(f"dtype={args.dtype} rows={args.rows} hidden={args.hidden}")
    print(f"max_abs_error={max_err:.8g}")
    print(f"eager_us={eager_us:.3f}")
    print(f"fused_us={fused_us:.3f}")
    print(f"speedup={eager_us / fused_us:.3f}x")


if __name__ == "__main__":
    main()
