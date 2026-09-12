from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def system_name(row: dict) -> str:
    if row["compile"]:
        return "c4_optimized"
    if row["prefetch"]:
        return "c3_amp_pin_prefetch"
    if row["pin"]:
        return "c2_amp_pin"
    if row["amp"]:
        return "c1_amp"
    return "c0_fp32"


def parse_run_name(name: str) -> dict:
    # Example:
    # transformer_seed42_amp1_pin1_prefetch0_compile0

    parts = name.split("_")

    out = {
        "model": parts[0],
        "seed": None,
        "amp": False,
        "pin": False,
        "prefetch": False,
        "compile": False,
    }

    for part in parts[1:]:
        if part.startswith("seed"):
            out["seed"] = int(part[4:])
        elif part.startswith("amp"):
            out["amp"] = bool(int(part[3:]))
        elif part.startswith("pin"):
            out["pin"] = bool(int(part[3:]))
        elif part.startswith("prefetch"):
            out["prefetch"] = bool(
                int(part[8:])
            )
        elif part.startswith("compile"):
            out["compile"] = bool(
                int(part[7:])
            )

    out["system"] = system_name(out)

    return out


def mean_metric(
    history: list[dict],
    key: str,
    fallback_key: str | None = None,
) -> float:
    values = []

    for row in history:
        value = row.get(key)

        if value is None and fallback_key:
            value = row.get(fallback_key)

        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if not math.isnan(value):
                values.append(value)

    if not values:
        return float("nan")

    return sum(values) / len(values)


def main() -> None:
    rows = []

    # Phase 8+ experiment format:
    #
    # results/.../<run_name>/result.json
    #
    # We intentionally summarize only the new
    # reproducible run-artifact format.
    for path in sorted(
        RESULTS.rglob("result.json")
    ):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            run = json.load(f)

        if run.get("status") != "completed":
            continue

        history = run.get("history", [])

        if not history:
            continue

        meta = parse_run_name(
            run["run"]
        )

        environment = run.get(
            "environment",
            {},
        )

        config = run.get(
            "config",
            {},
        )

        data_cfg = config.get(
            "data",
            {},
        )

        # Epoch 1 is treated as warm-up for
        # systems summaries when multiple
        # epochs are available.
        steady_history = (
            history[1:]
            if len(history) > 1
            else history
        )

        epoch_seconds = mean_metric(
            steady_history,
            "train_seconds",
        )

        samples_per_sec = mean_metric(
            steady_history,
            "train_samples_per_sec",
        )

        peak_allocated_mb = mean_metric(
            steady_history,
            "train_peak_allocated_memory_mb",
            fallback_key="train_peak_memory_mb",
        )

        peak_reserved_mb = mean_metric(
            steady_history,
            "train_peak_reserved_memory_mb",
        )

        rows.append(
            {
                **meta,
                "result_path": str(path),
                "device": environment.get(
                    "device"
                ),
                "gpu_name": environment.get(
                    "gpu_name"
                ),
                "torch_version": (
                    environment.get(
                        "torch_version"
                    )
                ),
                "cuda_version": (
                    environment.get(
                        "cuda_version"
                    )
                ),
                "git_sha": environment.get(
                    "git_sha"
                ),
                "parameters": run.get(
                    "parameters"
                ),
                "batch_size": data_cfg.get(
                    "batch_size"
                ),
                "steady_state_epochs": len(
                    steady_history
                ),
                "warmup_excluded": (
                    len(history) > 1
                ),
                "best_val_rmse": run.get(
                    "best_val_rmse"
                ),
                "test_rmse": run["test"][
                    "rmse"
                ],
                "test_mae": run["test"][
                    "mae"
                ],
                "epoch_seconds": (
                    epoch_seconds
                ),
                "samples_per_sec": (
                    samples_per_sec
                ),
                "peak_allocated_memory_mb": (
                    peak_allocated_mb
                ),
                "peak_reserved_memory_mb": (
                    peak_reserved_mb
                ),
            }
        )

    if not rows:
        raise SystemExit(
            "No completed Phase 8+ result.json "
            "files found under results/"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        RESULTS / "summary.csv",
        index=False,
    )

    grouped = (
        df.groupby(
            ["model", "system"],
            as_index=False,
        )
        .agg(
            test_rmse_mean=(
                "test_rmse",
                "mean",
            ),
            test_rmse_std=(
                "test_rmse",
                "std",
            ),
            epoch_seconds_mean=(
                "epoch_seconds",
                "mean",
            ),
            epoch_seconds_std=(
                "epoch_seconds",
                "std",
            ),
            samples_per_sec_mean=(
                "samples_per_sec",
                "mean",
            ),
            samples_per_sec_std=(
                "samples_per_sec",
                "std",
            ),
            peak_allocated_memory_mb_mean=(
                "peak_allocated_memory_mb",
                "mean",
            ),
            peak_reserved_memory_mb_mean=(
                "peak_reserved_memory_mb",
                "mean",
            ),
        )
        .sort_values(
            [
                "model",
                "system",
            ]
        )
    )

    grouped.to_csv(
        RESULTS / "grouped_summary.csv",
        index=False,
    )

    comparisons = []

    for (model, seed), sub in df.groupby(
        ["model", "seed"]
    ):
        base = sub[
            sub.system == "c0_fp32"
        ]

        if base.empty:
            continue

        base = base.iloc[0]

        for _, row in sub.iterrows():
            allocated_reduction = (
                100.0
                * (
                    base.peak_allocated_memory_mb
                    - row.peak_allocated_memory_mb
                )
                / base.peak_allocated_memory_mb
                if (
                    pd.notna(
                        base.peak_allocated_memory_mb
                    )
                    and base.peak_allocated_memory_mb
                    > 0
                )
                else float("nan")
            )

            reserved_reduction = (
                100.0
                * (
                    base.peak_reserved_memory_mb
                    - row.peak_reserved_memory_mb
                )
                / base.peak_reserved_memory_mb
                if (
                    pd.notna(
                        base.peak_reserved_memory_mb
                    )
                    and base.peak_reserved_memory_mb
                    > 0
                )
                else float("nan")
            )

            comparisons.append(
                {
                    "model": model,
                    "seed": seed,
                    "system": row.system,
                    "throughput_speedup_x": (
                        row.samples_per_sec
                        / base.samples_per_sec
                    ),
                    "epoch_speedup_x": (
                        base.epoch_seconds
                        / row.epoch_seconds
                    ),
                    "allocated_memory_reduction_pct": (
                        allocated_reduction
                    ),
                    "reserved_memory_reduction_pct": (
                        reserved_reduction
                    ),
                    "rmse_change_pct": (
                        100.0
                        * (
                            row.test_rmse
                            - base.test_rmse
                        )
                        / base.test_rmse
                    ),
                }
            )

    comp = pd.DataFrame(
        comparisons
    )

    comp.to_csv(
        RESULTS / "system_speedups.csv",
        index=False,
    )

    print(
        "\nGrouped benchmark summary\n"
    )

    print(
        grouped.to_string(
            index=False
        )
    )

    print("\nWrote:")
    print("  results/summary.csv")
    print(
        "  results/grouped_summary.csv"
    )
    print(
        "  results/system_speedups.csv"
    )


if __name__ == "__main__":
    main()