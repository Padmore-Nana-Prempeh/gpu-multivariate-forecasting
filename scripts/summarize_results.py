from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def system_name(row: dict) -> str:
    if row["compile"]:
        return "amp_pin_prefetch_compile"
    if row["prefetch"]:
        return "amp_pin_prefetch"
    if row["pin"]:
        return "amp_pin"
    if row["amp"]:
        return "amp"
    return "fp32"


def parse_run_name(name: str) -> dict:
    # <model>_seed42_amp1_pin1_prefetch1_compile0
    parts = name.split("_")
    model = parts[0]
    out = {"model": model}
    for part in parts[1:]:
        if part.startswith("seed"):
            out["seed"] = int(part[4:])
        elif part.startswith("amp"):
            out["amp"] = bool(int(part[3:]))
        elif part.startswith("pin"):
            out["pin"] = bool(int(part[3:]))
        elif part.startswith("prefetch"):
            out["prefetch"] = bool(int(part[8:]))
        elif part.startswith("compile"):
            out["compile"] = bool(int(part[7:]))
    out["system"] = system_name(out)
    return out


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        meta = parse_run_name(run["run"])
        hist = run["history"]
        best = min(hist, key=lambda x: x["val_rmse"])
        rows.append(
            {
                **meta,
                "device": run["device"],
                "torch_version": run["torch_version"],
                "cuda_version": run["cuda_version"],
                "best_val_rmse": run["best_val_rmse"],
                "test_rmse": run["test"]["rmse"],
                "test_mae": run["test"]["mae"],
                "epoch_seconds": best["train_seconds"],
                "samples_per_sec": best["train_samples_per_sec"],
                "peak_memory_mb": best["train_peak_memory_mb"],
            }
        )

    if not rows:
        raise SystemExit("No result JSON files found in results/")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "summary.csv", index=False)

    grouped = (
        df.groupby(["model", "system"], as_index=False)
        .agg(
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            epoch_seconds_mean=("epoch_seconds", "mean"),
            samples_per_sec_mean=("samples_per_sec", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        )
        .sort_values(["model", "test_rmse_mean"])
    )
    grouped.to_csv(RESULTS / "grouped_summary.csv", index=False)

    comparisons = []
    for (model, seed), sub in df.groupby(["model", "seed"]):
        base = sub[sub.system == "fp32"]
        if base.empty:
            continue
        base = base.iloc[0]
        for _, row in sub.iterrows():
            comparisons.append(
                {
                    "model": model,
                    "seed": seed,
                    "system": row.system,
                    "throughput_speedup_x": row.samples_per_sec / base.samples_per_sec,
                    "epoch_speedup_x": base.epoch_seconds / row.epoch_seconds,
                    "memory_reduction_pct": (
                        100.0 * (base.peak_memory_mb - row.peak_memory_mb) / base.peak_memory_mb
                        if pd.notna(base.peak_memory_mb) and base.peak_memory_mb > 0
                        else float("nan")
                    ),
                    "rmse_change_pct": 100.0 * (row.test_rmse - base.test_rmse) / base.test_rmse,
                }
            )
    comp = pd.DataFrame(comparisons)
    comp.to_csv(RESULTS / "system_speedups.csv", index=False)

    print("\nGrouped benchmark summary\n")
    print(grouped.to_string(index=False))
    print("\nWrote:")
    print("  results/summary.csv")
    print("  results/grouped_summary.csv")
    print("  results/system_speedups.csv")


if __name__ == "__main__":
    main()
