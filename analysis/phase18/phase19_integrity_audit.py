#!/usr/bin/env python3
"""Offline verification of archived Phase-18 official experiment records.

Place in analysis/phase18/phase19_integrity_audit.py and run from the project
root: python3 analysis/phase18/phase19_integrity_audit.py

Reads only archived result.json/status.json/runner_status.json/checkpoints;
never trains models, opens a GPU, or modifies the frozen experiment data.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent
RESULT_ROOT = BASE / "results" / "official_v1"
REPORT = BASE / "phase19_integrity_audit.txt"
MODELS = ("lstm", "gru", "tcn", "transformer")
SEEDS = (17, 42, 101)
CONFIGS = {
    "C0": (False, False, False, False, False),
    "C1": (True, False, False, False, False),
    "C2": (True, True, True, False, False),
    "C3": (True, True, True, True, False),
    "C4": (True, True, True, True, True),
}
EXPECTED_DELTAS = {
    ("C0", "C1"): {"train.amp"},
    ("C1", "C2"): {"data.pin_memory", "train.non_blocking_h2d"},
    ("C2", "C3"): {"train.prefetch_stream"},
    ("C3", "C4"): {"train.compile"},
}


def flattened(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flattened(value, name))
        else:
            # Scalars and lists should compare by value rather than identity.
            out[name] = json.dumps(value, sort_keys=True)
    return out


def read_json(path: Path) -> dict:
    assert path.is_file(), f"Missing {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def check() -> list[str]:
    assert RESULT_ROOT.is_dir(), f"Missing results: {RESULT_ROOT}"
    lines: list[str] = []
    records: dict[tuple[str, int, str], dict] = {}
    counts: Counter = Counter()
    result_paths = sorted(RESULT_ROOT.glob("*/result.json"))
    assert len(result_paths) == 60, f"Expected 60 result.json, found {len(result_paths)}"

    for path in result_paths:
        folder = path.parent
        result = read_json(path)
        cfg = result["config"]
        model = cfg["model"]["name"]
        seed = int(cfg["seed"])
        flags = (
            bool(cfg["train"]["amp"]),
            bool(cfg["data"]["pin_memory"]),
            bool(cfg["train"].get("non_blocking_h2d", False)),
            bool(cfg["train"]["prefetch_stream"]),
            bool(cfg["train"]["compile"]),
        )
        labels = [name for name, want in CONFIGS.items() if flags == want]
        assert len(labels) == 1, f"Unexpected flags in {folder.name}: {flags}"
        configuration = labels[0]
        key = model, seed, configuration
        assert key not in records, f"Duplicate model/seed/config: {key}"
        assert model in MODELS and seed in SEEDS, key
        assert result["run"] == folder.name, folder.name
        assert result["status"] == "completed", folder.name
        assert read_json(folder / "status.json")["status"] == "completed", folder.name
        runner = read_json(folder / "runner_status.json")
        assert runner["status"] == "completed" and runner.get("return_code") == 0, folder.name
        checkpoint = folder / "best.pt"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0, folder.name
        assert len(result["history"]) == 20, folder.name
        assert cfg["data"]["batch_size"] == 256, key
        assert cfg["data"].get("num_workers", 0) == 0, key
        assert cfg["train"]["epochs"] == 20, key
        records[key] = result
        counts[(model, configuration)] += 1

    expected = set(itertools.product(MODELS, SEEDS, CONFIGS))
    assert set(records) == expected, f"Missing={expected - set(records)}, extra={set(records) - expected}"
    lines.append("PASS: 60/60 run records, 60 checkpoints, 1,200 epochs, all complete")
    lines.append("PASS: All models/seeds/configurations are present exactly once")
    lines.append("PASS: batch_size=256, num_workers=0, epochs=20 in every saved config")

    for model in MODELS:
        for seed in SEEDS:
            for (before, after), expected_fields in EXPECTED_DELTAS.items():
                a = flattened(records[(model, seed, before)]["config"])
                b = flattened(records[(model, seed, after)]["config"])
                assert a.keys() == b.keys(), f"Config field mismatch {model} {seed} {before}->{after}"
                differences = {k for k in a if a[k] != b[k]}
                assert differences == expected_fields, (
                    f"Unexpected resolved-config differences {model} seed{seed} "
                    f"{before}->{after}: {sorted(differences)} "
                    f"expected {sorted(expected_fields)}"
                )
    lines.append("PASS: All 48 adjacent configuration comparisons change only intended flags")
    lines.append("PASS: C1->C2 modifies ONLY data.pin_memory and train.non_blocking_h2d")

    unchanged = ("C1", "C2", "C3")
    for model in MODELS:
        for seed in SEEDS:
            source = records[(model, seed, "C1")]
            for conf in unchanged[1:]:
                dest = records[(model, seed, conf)]
                assert source["test"] == dest["test"], (model, seed, conf, "test metrics")
                for epoch, (r1, rk) in enumerate(zip(source["history"], dest["history"]), 1):
                    for field in ("train_loss", "val_rmse"):
                        assert r1[field] == rk[field], (model, seed, conf, epoch, field)
    lines.append("PASS: C1/C2/C3 match exactly on test metrics and 20-epoch losses/val RMSE")
    lines.append("NOTE: Correctness equality alone does not prove the transfer/compile path ran as intended")
    lines.append("NOTE: TF32 flags and torch.compile graph breaks cannot be established from these records")
    lines.append("NOTE: C2 slowdown is an OBSERVATION; pinning's independent causal effect is unidentified")
    return lines


if __name__ == "__main__":
    messages = check()
    REPORT.write_text("\n".join(messages) + "\n", encoding="utf-8")
    print("\n".join(messages))
    print(f"Wrote: {REPORT.relative_to(PROJECT)}")
