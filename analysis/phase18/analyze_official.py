from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import re
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE / "results" / "official_v1"
OUT = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    (0, 0, 0, 0): "C0",
    (1, 0, 0, 0): "C1",
    (1, 1, 0, 0): "C2",
    (1, 1, 1, 0): "C3",
    (1, 1, 1, 1): "C4",
}

PATTERN = re.compile(
    r"(lstm|gru|tcn|transformer)_seed(17|42|101)"
    r"_amp([01])_pin([01])_prefetch([01])_compile([01])"
)


def flatten_numeric(obj, prefix=""):
    """Extract scalar numerical evidence without flattening configs."""
    found = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "config":
                continue
            path = f"{prefix}.{key}" if prefix else key
            found.update(flatten_numeric(value, path))

    elif (
        isinstance(obj, (int, float))
        and not isinstance(obj, bool)
        and math.isfinite(obj)
    ):
        found[prefix] = obj

    return found


rows = []

for path in sorted(ROOT.glob("*/result.json")):
    folder = path.parent
    match = PATTERN.fullmatch(folder.name)
    assert match, f"Unexpected run directory: {folder.name}"

    model, seed, amp, pin, prefetch, compile_ = match.groups()
    flags = tuple(map(int, (amp, pin, prefetch, compile_)))
    assert flags in CONFIGS, f"Unknown CUDA configuration: {flags}"

    result = json.loads(path.read_text())
    runner = json.loads(
        (folder / "runner_status.json").read_text()
    )
    status = json.loads((folder / "status.json").read_text())

    assert result["status"] == "completed", folder.name
    assert status["status"] == "completed", folder.name
    assert runner["status"] == "completed", folder.name
    assert runner.get("return_code") == 0, folder.name
    assert result["run"] == folder.name, folder.name
    assert (folder / "best.pt").stat().st_size > 0

    row = {
        "model": model,
        "seed": int(seed),
        "cuda_config": CONFIGS[flags],
        "run": folder.name,
    }

    for key, value in flatten_numeric(result).items():
        row[f"result.{key}"] = value

    rows.append(row)

expected = {
    (model, seed, config)
    for model in ("lstm", "gru", "tcn", "transformer")
    for seed in (17, 42, 101)
    for config in ("C0", "C1", "C2", "C3", "C4")
}

actual = {
    (r["model"], r["seed"], r["cuda_config"])
    for r in rows
}

assert len(rows) == 60
assert actual == expected, f"Missing: {expected - actual}"
assert len(actual) == len(rows)

metric_keys = sorted({
    key
    for row in rows
    for key in row
    if key.startswith("result.")
})

# One row for every official experiment.
flat_csv = OUT / "official_v1_all_runs.csv"
columns = ["model", "seed", "cuda_config", "run"] + metric_keys

with flat_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

# Three-seed aggregation for every scalar result field.
groups = defaultdict(list)

for row in rows:
    groups[(row["model"], row["cuda_config"])].append(row)

summary = []

for (model, config), members in sorted(groups.items()):
    assert len(members) == 3

    for metric in metric_keys:
        values = [
            member[metric]
            for member in members
            if metric in member
        ]

        if len(values) != 3:
            continue

        summary.append({
            "model": model,
            "cuda_config": config,
            "metric": metric,
            "n": 3,
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        })

summary_csv = OUT / "official_v1_seed_summary.csv"

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model", "cuda_config", "metric",
            "n", "mean", "sd", "min", "max"
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

print("========== PHASE 19 RESULTS AUDIT ==========")
print(f"Completed records: {len(rows)}/60")
print(f"Unique combinations: {len(actual)}/60")
print("Checkpoints: 60 verified")
print()

print("========== MODEL × CUDA CONFIGURATION ==========")

counts = Counter((r["model"], r["cuda_config"]) for r in rows)

for model in ("lstm", "gru", "tcn", "transformer"):
    print(
        f"{model:12s}",
        " ".join(
            f"{cfg}={counts[(model, cfg)]}"
            for cfg in ("C0", "C1", "C2", "C3", "C4")
        )
    )

print()
print("========== AVAILABLE NUMERICAL FIELDS ==========")

for key in metric_keys:
    print(key)

print()
print("========== OUTPUT FILES ==========")
print(flat_csv)
print(summary_csv)
print(f"Aggregated summary rows: {len(summary)}")
print()
print("PASS: PHASE 19 DATA EXTRACTION COMPLETE")
