from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fused_ln_gelu_cuda


torch.manual_seed(42)

EPS = 1e-5
WARMUP = 100
REPS = 500

SHAPES = [
    (4096, 92),
    (24576, 92),
    (4096, 127),
    (4096, 256),
]


def measure(fn):
    # Warm-up
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    times = []

    for _ in range(REPS):
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()
        fn()
        end.record()

        end.synchronize()

        times.append(
            start.elapsed_time(end)
        )

    a = np.asarray(
        times,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "p50_ms": float(
            np.percentile(a, 50)
        ),
        "p95_ms": float(
            np.percentile(a, 95)
        ),
        "min_ms": float(a.min()),
    }


results = []

for rows, hidden in SHAPES:
    x = torch.randn(
        rows,
        hidden,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.float32,
    )

    def eager():
        return F.gelu(
            F.layer_norm(
                x,
                (hidden,),
                weight,
                bias,
                EPS,
            ),
            approximate="tanh",
        )

    def fused():
        return fused_ln_gelu_cuda.forward(
            x,
            weight,
            bias,
            EPS,
        )

    reference = eager()
    candidate = fused()

    torch.cuda.synchronize()

    max_abs_error = (
        reference - candidate
    ).abs().max().item()

    eager_stats = measure(eager)
    fused_stats = measure(fused)

    speedup = (
        eager_stats["p50_ms"]
        / fused_stats["p50_ms"]
    )

    row = {
        "rows": rows,
        "hidden": hidden,
        "elements": rows * hidden,
        "max_abs_error": max_abs_error,
        "eager_mean_ms":
            eager_stats["mean_ms"],
        "eager_p50_ms":
            eager_stats["p50_ms"],
        "eager_p95_ms":
            eager_stats["p95_ms"],
        "fused_mean_ms":
            fused_stats["mean_ms"],
        "fused_p50_ms":
            fused_stats["p50_ms"],
        "fused_p95_ms":
            fused_stats["p95_ms"],
        "speedup_x": speedup,
    }

    results.append(row)

    print(json.dumps(row))


outdir = Path(
    "../../results/kernel"
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    outdir / "fused_ln_gelu.json",
    "w",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

with open(
    outdir / "fused_ln_gelu.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(results)

print()
print(
    "PHASE 17 MICROBENCHMARK: COMPLETE"
)
print(
    "Results: results/kernel/fused_ln_gelu.csv"
)
