#!/usr/bin/env python3
"""Offline Phase 19 audit of frozen Phase 18 NVIDIA A40 runs.

Run from any directory with:
    python3 analysis/phase18/phase19_full_analysis.py

No torch, GPU, or third-party Python packages are required.
This script only reads official results and writes analysis tables/report.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as stats
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "results" / "official_v1"
OUT = BASE / "tables"
MODELS = ("lstm", "gru", "tcn", "transformer")
SEEDS = (17, 42, 101)
CONFIGS = ("C0", "C1", "C2", "C3", "C4")
FLAG_CONFIG = {
    (0, 0, 0, 0): "C0",
    (1, 0, 0, 0): "C1",
    (1, 1, 0, 0): "C2",
    (1, 1, 1, 0): "C3",
    (1, 1, 1, 1): "C4",
}
NAME = re.compile(
    r"(lstm|gru|tcn|transformer)_seed(17|42|101)"
    r"_amp([01])_pin([01])_prefetch([01])_compile([01])"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, context: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{context}: expected a numerical value, found {value!r}")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{context}: value is negative/nonfinite: {value}")
    return value


def write_csv(path: Path, records: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def mean_sd(values):
    return stats.mean(values), stats.stdev(values)


def timestamp_seconds(runner, name):
    start = datetime.fromisoformat(runner["started_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(runner["finished_utc"].replace("Z", "+00:00"))
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        raise ValueError(f"{name}: invalid runner timestamps ({seconds})")
    return seconds


def parse_one(path: Path):
    folder = path.parent
    match = NAME.fullmatch(folder.name)
    if not match:
        raise ValueError(f"Unexpected official directory name: {folder.name}")
    model, seed_text, *flags_text = match.groups()
    seed = int(seed_text)
    flags = tuple(map(int, flags_text))
    if flags not in FLAG_CONFIG:
        raise ValueError(f"{folder.name}: unexpected CUDA flags")
    cuda_config = FLAG_CONFIG[flags]

    result = read_json(folder / "result.json")
    runner = read_json(folder / "runner_status.json")
    status = read_json(folder / "status.json")
    config = read_json(folder / "config.json")
    environment = read_json(folder / "environment.json")
    if not (folder / "best.pt").is_file() or (folder / "best.pt").stat().st_size <= 0:
        raise ValueError(f"{folder.name}: missing/empty checkpoint")
    if result.get("status") != "completed" or status.get("status") != "completed":
        raise ValueError(f"{folder.name}: missing completed status")
    if runner.get("status") != "completed" or runner.get("return_code") != 0:
        raise ValueError(f"{folder.name}: official runner not successful")
    if result.get("run") != folder.name or status.get("run") != folder.name:
        raise ValueError(f"{folder.name}: result/status identity mismatch")
    if result.get("config") != config:
        raise ValueError(f"{folder.name}: saved config mismatch")
    if config["model"]["name"] != model or int(config["seed"]) != seed:
        raise ValueError(f"{folder.name}: model/seed mismatch")
    config_flags = (
        int(config["train"]["amp"]),
        int(config["data"]["pin_memory"]),
        int(config["train"]["prefetch_stream"]),
        int(config["train"].get("compile", False)),
    )
    if config_flags != flags:
        raise ValueError(f"{folder.name}: saved CUDA flag mismatch")
    if result.get("environment") != environment:
        raise ValueError(f"{folder.name}: saved environment mismatch")
    if environment.get("device") != "cuda" or "A40" not in str(environment.get("gpu_name", "")):
        raise ValueError(f"{folder.name}: not recorded as NVIDIA A40 CUDA run")

    history = result.get("history")
    epochs = int(config["train"]["epochs"])
    if not isinstance(history, list) or len(history) != epochs or epochs != 20:
        raise ValueError(f"{folder.name}: expected 20 epoch history records, found {len(history) if isinstance(history, list) else 'none'}")

    epochs_out = []
    for index, entry in enumerate(history, 1):
        if entry.get("epoch") != index:
            raise ValueError(f"{folder.name}: nonsequential epoch at {index}")
        train_seconds = number(entry["train_seconds"], f"{folder.name}/epoch{index}/seconds")
        throughput = number(entry["train_samples_per_sec"], f"{folder.name}/epoch{index}/throughput")
        if train_seconds <= 0 or throughput <= 0:
            raise ValueError(f"{folder.name}: nonpositive time/throughput")
        ep = {
            "model": model,
            "seed": seed,
            "cuda_config": cuda_config,
            "run": folder.name,
            "epoch": index,
            "train_seconds": train_seconds,
            "train_samples_per_sec": throughput,
            "train_peak_allocated_mb": number(entry["train_peak_allocated_memory_mb"], f"{folder.name}/allocated", True),
            "train_peak_reserved_mb": number(entry["train_peak_reserved_memory_mb"], f"{folder.name}/reserved", True),
            "train_loss": number(entry["train_loss"], f"{folder.name}/loss"),
            "val_rmse": number(entry["val_rmse"], f"{folder.name}/val_rmse"),
        }
        epochs_out.append(ep)

    total_train = sum(e["train_seconds"] for e in epochs_out)
    steady_train = sum(e["train_seconds"] for e in epochs_out[1:])
    # Use total samples / total seconds, rather than arithmetic averaging rates.
    total_samples = sum(e["train_samples_per_sec"] * e["train_seconds"] for e in epochs_out)
    steady_samples = sum(e["train_samples_per_sec"] * e["train_seconds"] for e in epochs_out[1:])
    row = {
        "model": model,
        "seed": seed,
        "cuda_config": cuda_config,
        "run": folder.name,
        "test_rmse": number(result["test"]["rmse"], f"{folder.name}/test_rmse"),
        "test_mae": number(result["test"]["mae"], f"{folder.name}/test_mae"),
        "test_mse": number(result["test"]["mse"], f"{folder.name}/test_mse"),
        "best_val_rmse": number(result["best_val_rmse"], f"{folder.name}/best_val_rmse"),
        "best_epoch": result["best_epoch"],
        "parameters": result["parameters"],
        "train_seconds_total": total_train,
        "train_epoch1_seconds": epochs_out[0]["train_seconds"],
        "train_steady_seconds_per_epoch": steady_train / (epochs - 1),
        "train_samples_per_second_total": total_samples / total_train,
        "train_samples_per_second_steady": steady_samples / steady_train,
        "train_peak_allocated_mb": max(e["train_peak_allocated_mb"] for e in epochs_out),
        "train_peak_reserved_mb": max(e["train_peak_reserved_mb"] for e in epochs_out),
        "runner_wall_seconds": timestamp_seconds(runner, folder.name),
        "gpu": environment["gpu_name"],
        "torch": environment.get("torch_version"),
        "cuda": environment.get("cuda_version"),
        "container_image_digest": runner.get("container_image_digest"),
    }
    return row, epochs_out


def make_report(summary, paired):
    index = {(r["model"], r["cuda_config"]): r for r in summary}
    paired_groups = defaultdict(list)
    for row in paired:
        paired_groups[(row["model"], row["cuda_config"])].append(row)

    def fnum(x, digits=3):
        return f"{x:.{digits}f}"

    lines = [
        "# Phase 19: Official A40 experiment results (local audit)",
        "",
        "Four models × three random seeds × five CUDA configurations = 60 official 20-epoch runs.",
        "This report is generated from the archived `result.json` epoch histories and `runner_status.json` timestamps, not from model checkpoints.",
        "",
        "## Test forecasting accuracy",
        "",
        "Mean ± sample SD of test RMSE across three seeds (lower is better):",
        "",
        "| Model | C0 | C1 | C2 | C3 | C4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        cells = [model.upper() if model != "transformer" else "Transformer"]
        for cfg in CONFIGS:
            r = index[(model, cfg)]
            cells.append(f"{r['test_rmse_mean']:.4f} ± {r['test_rmse_sd']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Steady-state training throughput relative to C0",
        "",
        "Geometric mean of matched-seed throughput ratios (Ck / C0), using epochs 2–20. Higher is better.",
        "",
        "| Model | C1 | C2 | C3 | C4 |",
        "|---|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        cells = [model.upper() if model != "transformer" else "Transformer"]
        for cfg in CONFIGS[1:]:
            ratios = [r["steady_throughput_ratio_vs_c0"] for r in paired_groups[(model, cfg)]]
            geo = math.exp(stats.mean(math.log(v) for v in ratios))
            cells.append(f"{geo:.3f}×")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- C0 = FP32; C1 = AMP; C2 adds pinned/nonblocking host-to-device transfer; C3 adds CUDA prefetch; C4 adds `torch.compile`.",
        "- Accuracy is evaluated on inverse-transformed test targets. Smaller RMSE is better.",
        "- The throughput ratio is paired by **model and seed**. It is not evidence of statistical significance with only three seeds.",
        "- Epoch 1 includes potential initialization/JIT compilation effects. Both all-epoch and epochs-2–20 metrics are retained in the CSVs.",
        "- `runner_wall_seconds` includes process startup, training, validation, checkpointing, testing, and compilation; it is not GPU kernel execution time alone.",
        "- GPU memory is PyTorch's peak allocated/reserved memory recorded during training epochs, not total hardware memory occupancy.",
        "- These end-to-end numbers do **not** benchmark the standalone forward-only fused LayerNorm + GELU CUDA extension unless it is explicitly integrated into the training graph.",
        "- Hardware was an NVIDIA A40; per-run environment and image digest should be checked before publication.",
        "- This is a single-model/seed/config run per cell. Running additional replications requires a separate protocol and is not part of this frozen campaign.",
        "",
        "## Generated data files",
        "",
        "- `official_v1_full_runs.csv`: 60 runs with accuracy, total/steady training timing, throughput, peak GPU memory, and total wall time.",
        "- `official_v1_epoch_history.csv`: 1,200 per-epoch records.",
        "- `official_v1_model_config_summary.csv`: 20 model/config group means and sample SDs (three seeds each).",
        "- `official_v1_paired_vs_c0.csv`: 48 matched-seed comparisons against the FP32 baseline.",
    ])
    return "\n".join(lines) + "\n"


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"Missing extracted official runs: {ROOT}")
    paths = sorted(ROOT.glob("*/result.json"))
    if len(paths) != 60:
        raise SystemExit(f"Expected 60 result.json files, found {len(paths)} in {ROOT}")
    runs = []
    epochs = []
    for path in paths:
        row, entries = parse_one(path)
        runs.append(row)
        epochs.extend(entries)
    keys = {(r["model"], r["seed"], r["cuda_config"]) for r in runs}
    expected = {(m, s, c) for m in MODELS for s in SEEDS for c in CONFIGS}
    if keys != expected or len(keys) != len(runs) or len(epochs) != 1200:
        raise SystemExit(f"Expected full 60-cell matrix, missing={expected-keys}, extras={keys-expected}, epoch_count={len(epochs)}")

    run_fields = list(runs[0])
    epoch_fields = list(epochs[0])
    groups = defaultdict(list)
    for row in runs:
        groups[(row["model"], row["cuda_config"])].append(row)
    metrics = (
        "test_rmse", "test_mae", "test_mse", "best_val_rmse",
        "train_seconds_total", "train_steady_seconds_per_epoch",
        "train_samples_per_second_total", "train_samples_per_second_steady",
        "train_peak_allocated_mb", "train_peak_reserved_mb", "runner_wall_seconds",
    )
    summary = []
    for model in MODELS:
        for config in CONFIGS:
            members = sorted(groups[(model, config)], key=lambda x: x["seed"])
            if len(members) != 3:
                raise ValueError(f"{model}/{config}: expected exactly three seeds")
            item = {"model": model, "cuda_config": config, "n_seeds": 3}
            for metric in metrics:
                mean, sd = mean_sd([r[metric] for r in members])
                item[metric + "_mean"] = mean
                item[metric + "_sd"] = sd
            summary.append(item)

    by_key = {(r["model"], r["seed"], r["cuda_config"]): r for r in runs}
    paired = []
    for model in MODELS:
        for seed in SEEDS:
            baseline = by_key[(model, seed, "C0")]
            for config in CONFIGS[1:]:
                current = by_key[(model, seed, config)]
                paired.append({
                    "model": model, "seed": seed, "cuda_config": config,
                    "rmse_percent_change_vs_c0": 100 * (current["test_rmse"] / baseline["test_rmse"] - 1),
                    "total_train_speedup_vs_c0": baseline["train_seconds_total"] / current["train_seconds_total"],
                    "steady_epoch_speedup_vs_c0": baseline["train_steady_seconds_per_epoch"] / current["train_steady_seconds_per_epoch"],
                    "steady_throughput_ratio_vs_c0": current["train_samples_per_second_steady"] / baseline["train_samples_per_second_steady"],
                    "wall_time_speedup_vs_c0": baseline["runner_wall_seconds"] / current["runner_wall_seconds"],
                    "gpu_allocated_memory_percent_change_vs_c0": 100 * (current["train_peak_allocated_mb"] / baseline["train_peak_allocated_mb"] - 1),
                    "gpu_reserved_memory_percent_change_vs_c0": 100 * (current["train_peak_reserved_mb"] / baseline["train_peak_reserved_mb"] - 1),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("official_v1_full_runs.csv", runs, run_fields),
        ("official_v1_epoch_history.csv", epochs, epoch_fields),
        ("official_v1_model_config_summary.csv", summary, list(summary[0])),
        ("official_v1_paired_vs_c0.csv", paired, list(paired[0])),
    ]
    for filename, data, fieldnames in outputs:
        write_csv(OUT / filename, data, fieldnames)
    report = OUT / "official_v1_findings.md"
    report.write_text(make_report(summary, paired), encoding="utf-8")

    print("========== PHASE 19 FULL OFFLINE AUDIT ==========")
    print(f"Verified: {len(runs)} runs; {len(epochs)} epochs; {len(summary)} three-seed groups; {len(paired)} paired vs C0 comparisons")
    print("All 60 runner statuses, checkpoints, per-run config identities and NVIDIA A40 environment: PASS")
    print("\n========== MEAN TEST RMSE (3 SEEDS; LOWER IS BETTER) ==========")
    summary_by_key = {(r['model'], r['cuda_config']): r for r in summary}
    print(f"{'Model':13s}" + "".join(f"{c:>11s}" for c in CONFIGS))
    for model in MODELS:
        print(f"{model:13s}" + "".join(f"{summary_by_key[model,c]['test_rmse_mean']:11.4f}" for c in CONFIGS))
    print("\n========== C4 STEADY-STATE THROUGHPUT VS C0 ==========")
    for model in MODELS:
        values = [p["steady_throughput_ratio_vs_c0"] for p in paired if p["model"] == model and p["cuda_config"] == "C4"]
        print(f"{model:13s} {math.exp(stats.mean(math.log(v) for v in values)):.3f}x  (geometric mean of 3 paired ratios)")
    print("\n========== OUTPUT FILES ==========")
    for filename, _, _ in outputs:
        print(OUT / filename)
    print(report)
    print("\nPASS: Offline analysis complete. No training was launched.")


if __name__ == "__main__":
    main()
