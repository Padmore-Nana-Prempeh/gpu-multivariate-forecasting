"""
Verify the frozen official 60-run experiment plan.

No training, GPU access, or file modification.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/official_v1"
CONFIGS = PLAN / "configs"

MODELS = ("lstm", "gru", "tcn", "transformer")
SEEDS = (17, 42, 101)

# AMP, pinned memory, nonblocking H2D, prefetch, compile
SYSTEMS = {
    "C0": (False, False, False, False, False),
    "C1": (True, False, False, False, False),
    "C2": (True, True, True, False, False),
    "C3": (True, True, True, True, False),
    "C4": (True, True, True, True, True),
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    manifest = json.loads(
        (PLAN / "manifest.json").read_text(encoding="utf-8")
    )

    runs = manifest["runs"]

    check(len(runs) == 60, "Expected 60 manifest entries")
    check(manifest["planned_runs"] == 60, "Incorrect count")
    check(manifest["models"] == list(MODELS), "Wrong models")
    check(manifest["seeds"] == list(SEEDS), "Wrong seeds")
    check(
        manifest["systems"] == list(SYSTEMS),
        "Wrong execution configurations",
    )

    expected = set(itertools.product(MODELS, SEEDS, SYSTEMS))

    seen = set()
    run_ids = set()
    output_dirs = set()

    model_counts = Counter()
    system_counts = Counter()

    for run in runs:
        model = run["model"]
        seed = run["seed"]
        system = run["system"]

        key = (model, seed, system)

        check(key in expected, f"Unexpected experiment: {key}")
        check(key not in seen, f"Duplicate experiment: {key}")

        seen.add(key)

        run_id = f"{model}_seed{seed}_{system.lower()}"

        check(run["run_id"] == run_id, "Incorrect run ID")
        check(run_id not in run_ids, "Duplicate run ID")
        run_ids.add(run_id)

        expected_path = CONFIGS / f"{run_id}.yaml"

        check(
            run["resolved_config"]
            == f"experiments/official_v1/configs/{run_id}.yaml",
            f"Incorrect configuration path: {run_id}",
        )

        check(expected_path.is_file(), f"Missing: {expected_path}")

        raw = expected_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()

        check(
            actual_sha == run["config_sha256"],
            f"Configuration hash mismatch: {run_id}",
        )

        cfg = yaml.safe_load(raw)

        source = ROOT / f"configs/{model}_baseline.yaml"

        baseline = yaml.safe_load(
            source.read_text(encoding="utf-8")
        )

        check(
            run["source_config"] == f"configs/{model}_baseline.yaml",
            f"Incorrect source configuration: {run_id}",
        )

        check(cfg["seed"] == seed, f"Wrong seed: {run_id}")

        check(
            cfg["model"] == baseline["model"],
            f"Model architecture drift: {run_id}",
        )

        data_keys = (
            "path",
            "seq_len",
            "horizon",
            "train_ratio",
            "val_ratio",
            "batch_size",
            "num_workers",
        )

        for field in data_keys:
            check(
                cfg["data"][field] == baseline["data"][field],
                f"Data configuration drift: {run_id}/{field}",
            )

        training_keys = (
            "epochs",
            "lr",
            "weight_decay",
            "grad_clip",
        )

        for field in training_keys:
            check(
                cfg["train"][field] == baseline["train"][field],
                f"Training configuration drift: {run_id}/{field}",
            )

        check(
            cfg["train"]["device"] == "cuda",
            f"Incorrect device: {run_id}",
        )

        check(
            cfg["output"]["dir"] == "results/official_v1",
            f"Incorrect output directory: {run_id}",
        )

        amp, pin, nonblocking, prefetch, compile_ = SYSTEMS[system]

        expected_flags = {
            "amp": amp,
            "pin_memory": pin,
            "non_blocking_h2d": nonblocking,
            "prefetch_stream": prefetch,
            "compile": compile_,
        }

        actual_flags = {
            "amp": cfg["train"]["amp"],
            "pin_memory": cfg["data"]["pin_memory"],
            "non_blocking_h2d": cfg["train"]["non_blocking_h2d"],
            "prefetch_stream": cfg["train"]["prefetch_stream"],
            "compile": cfg["train"]["compile"],
        }

        check(
            actual_flags == expected_flags,
            f"Execution flag mismatch: {run_id}",
        )

        check(
            run["flags"] == expected_flags,
            f"Manifest flag mismatch: {run_id}",
        )

        engine_name = (
            f"{model}"
            f"_seed{seed}"
            f"_amp{int(amp)}"
            f"_pin{int(pin)}"
            f"_prefetch{int(prefetch)}"
            f"_compile{int(compile_)}"
        )

        expected_output = f"results/official_v1/{engine_name}"

        check(
            run["engine_run_dir"] == expected_output,
            f"Output path mismatch: {run_id}",
        )

        check(
            expected_output not in output_dirs,
            f"Output collision: {run_id}",
        )

        output_dirs.add(expected_output)

        model_counts[model] += 1
        system_counts[system] += 1

    check(seen == expected, "Some planned experiments are missing")

    actual_files = list(CONFIGS.glob("*.yaml"))

    check(
        len(actual_files) == 60,
        f"Expected 60 YAML files; found {len(actual_files)}",
    )

    for model in MODELS:
        check(
            model_counts[model] == 15,
            f"Expected 15 runs for {model}",
        )

    for system in SYSTEMS:
        check(
            system_counts[system] == 12,
            f"Expected 12 runs for {system}",
        )

    print("=" * 60)
    print("OFFICIAL V1 EXPERIMENT PLAN VERIFICATION")
    print("=" * 60)

    print("PASS: 60 exact model/seed/system combinations")
    print("PASS: 60 unique run identifiers")
    print("PASS: 60 unique training-output directories")
    print("PASS: All configuration SHA-256 hashes")
    print("PASS: Model architectures match their baselines")
    print("PASS: Data and training contracts remain fixed")
    print("PASS: All C0-C4 execution flags")
    print("PASS: CUDA device explicitly selected")
    print("PASS: 15 experiments per model")
    print("PASS: 12 experiments per execution configuration")

    print("\nNO GPU TRAINING WAS LAUNCHED.")
    print("OFFICIAL V1 CONFIGURATION VERIFICATION: PASSED")


if __name__ == "__main__":
    main()
