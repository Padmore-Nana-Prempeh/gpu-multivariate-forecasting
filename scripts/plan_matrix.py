"""
Pre-18: Generate the frozen 60-run experiment plan.

4 models x 3 seeds x 5 system configurations = 60 runs.

This script creates configuration files and a manifest.
It never launches model training.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


MODELS = ("lstm", "gru", "tcn", "transformer")
SEEDS = (17, 42, 101)

# AMP, pinned, non-blocking H2D, prefetch, compile
SYSTEMS = {
    "C0": (False, False, False, False, False),
    "C1": (True, False, False, False, False),
    "C2": (True, True, True, False, False),
    "C3": (True, True, True, True, False),
    "C4": (True, True, True, True, True),
}

PLAN_DIR = Path("experiments/official_v1")
CONFIG_DIR = PLAN_DIR / "configs"
RESULT_ROOT = Path("results/official_v1")


def write_if_identical_or_new(path: Path, contents: str):
    """Never silently overwrite a previously generated plan."""
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise RuntimeError(
                f"Existing file differs; refusing overwrite: {path}"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def validate_base(model_name: str, cfg: dict):
    if cfg["model"]["name"] != model_name:
        raise ValueError(f"Model mismatch: {model_name}")

    data = cfg["data"]
    train = cfg["train"]

    expected_data = {
        "seq_len": 96,
        "horizon": 24,
        "batch_size": 256,
        "num_workers": 0,
    }

    for key, expected in expected_data.items():
        if data[key] != expected:
            raise ValueError(
                f"{model_name}: {key}={data[key]}, expected {expected}"
            )

    if abs(data["train_ratio"] - 0.70) > 1e-12:
        raise ValueError("Unexpected training split")

    if abs(data["val_ratio"] - 0.10) > 1e-12:
        raise ValueError("Unexpected validation split")

    if train["epochs"] != 20:
        raise ValueError("Unexpected epoch count")

    if data["path"] != "data/ETTm1.csv":
        raise ValueError("Unexpected dataset path")


def main():
    planned_configs = {}
    runs = []

    for model in MODELS:
        source = Path(f"configs/{model}_baseline.yaml")

        with source.open(encoding="utf-8") as f:
            base = yaml.safe_load(f)

        validate_base(model, base)

        for seed in SEEDS:
            for system, flags in SYSTEMS.items():
                amp, pin, nonblocking, prefetch, compile_ = flags

                cfg = copy.deepcopy(base)

                cfg["seed"] = seed
                cfg["model"]["name"] = model

                cfg["data"]["pin_memory"] = pin
                cfg["data"]["num_workers"] = 0

                cfg["train"]["amp"] = amp
                cfg["train"]["non_blocking_h2d"] = nonblocking
                cfg["train"]["prefetch_stream"] = prefetch
                cfg["train"]["compile"] = compile_
                cfg["train"]["device"] = "cuda"

                cfg["output"]["dir"] = str(RESULT_ROOT)

                run_id = f"{model}_seed{seed}_{system.lower()}"

                config_path = CONFIG_DIR / f"{run_id}.yaml"

                config_text = yaml.safe_dump(
                    cfg,
                    sort_keys=True,
                    allow_unicode=True,
                )

                digest = hashlib.sha256(
                    config_text.encode("utf-8")
                ).hexdigest()

                # Match the current train.py naming convention.
                engine_name = (
                    f"{model}"
                    f"_seed{seed}"
                    f"_amp{int(amp)}"
                    f"_pin{int(pin)}"
                    f"_prefetch{int(prefetch)}"
                    f"_compile{int(compile_)}"
                )

                runs.append({
                    "run_id": run_id,
                    "model": model,
                    "seed": seed,
                    "system": system,
                    "source_config": str(source),
                    "resolved_config": str(config_path),
                    "config_sha256": digest,
                    "engine_run_dir": str(
                        RESULT_ROOT / engine_name
                    ),
                    "flags": {
                        "amp": amp,
                        "pin_memory": pin,
                        "non_blocking_h2d": nonblocking,
                        "prefetch_stream": prefetch,
                        "compile": compile_,
                    },
                })

                planned_configs[config_path] = config_text

    if len(runs) != 60:
        raise RuntimeError(f"Expected 60 runs, got {len(runs)}")

    if len({r["run_id"] for r in runs}) != 60:
        raise RuntimeError("Duplicate experiment identifiers")

    if len({r["engine_run_dir"] for r in runs}) != 60:
        raise RuntimeError("Training output-directory collision")

    manifest = {
        "schema_version": 1,
        "campaign": "official_v1",
        "planned_runs": len(runs),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "systems": list(SYSTEMS),
        "runs": runs,
    }

    manifest_path = PLAN_DIR / "manifest.json"

    manifest_text = (
        json.dumps(manifest, indent=2, sort_keys=True)
        + "\n"
    )

    # Check for conflicts before writing any new files.
    all_files = {
        **planned_configs,
        manifest_path: manifest_text,
    }

    for path, contents in all_files.items():
        if path.exists():
            if path.read_text(encoding="utf-8") != contents:
                raise RuntimeError(
                    f"Conflicting existing file: {path}"
                )

    for path, contents in all_files.items():
        write_if_identical_or_new(path, contents)

    print(f"PASS: {len(runs)} unique experiments")
    print("PASS: No output-directory collisions")
    print("PASS: Explicit C0-C4 transfer and compute flags")
    print(f"PASS: Manifest saved to {manifest_path}")

    print("\nTransformer, seed 42:")
    for run in runs:
        if run["model"] == "transformer" and run["seed"] == 42:
            print(
                f"  {run['run_id']}: "
                f"{run['flags']}"
            )

    print("\nNO TRAINING WAS LAUNCHED.")


if __name__ == "__main__":
    main()
