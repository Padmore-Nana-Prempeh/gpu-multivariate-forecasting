"""
Verify the exact Python package environment against the
GPU lock snapshot.

CPU-safe: does not require NVIDIA GPU access.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/verify_gpu_environment.py "
            "<lock-file>"
        )

    lock_path = Path(sys.argv[1])

    if not lock_path.is_file():
        raise SystemExit(f"Missing lock file: {lock_path}")

    expected = {}

    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.count("==") != 1:
            raise SystemExit(f"Invalid lock entry: {line}")

        name, version = line.split("==", 1)
        key = normalize(name)

        if key in expected:
            raise SystemExit(f"Duplicate locked package: {name}")

        expected[key] = version

    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pip",
            "list",
            "--format=json",
        ],
        text=True,
    )

    installed = {}

    for package in json.loads(output):
        key = normalize(package["name"])

        # The project is installed separately from its own source.
        if key == "gpuforecast":
            continue

        installed[key] = package["version"]

    missing = sorted(set(expected) - set(installed))
    unexpected = sorted(set(installed) - set(expected))

    mismatched = {
        name: {
            "expected": expected[name],
            "installed": installed[name],
        }
        for name in sorted(set(expected) & set(installed))
        if expected[name] != installed[name]
    }

    if missing or unexpected or mismatched:
        print("FAILED: GPU environment differs from lock.")
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        print("Version mismatches:", mismatched)
        raise SystemExit(1)

    assert expected["torch"] == "2.8.0+cu128"
    assert expected["triton"] == "3.4.0"

    import torch

    assert torch.__version__ == "2.8.0+cu128"
    assert torch.version.cuda == "12.8"

    print("=" * 55)
    print("GPU ENVIRONMENT LOCK VERIFICATION")
    print("=" * 55)
    print(f"PASS: {len(expected)} packages match exactly.")
    print("PASS: No missing packages.")
    print("PASS: No unexpected packages.")
    print("PASS: No version mismatches.")
    print(f"PASS: Python {sys.version.split()[0]}")
    print(f"PASS: PyTorch {torch.__version__}")
    print(f"PASS: CUDA runtime {torch.version.cuda}")
    print()
    print("GPU runtime execution is a separate preflight gate.")


if __name__ == "__main__":
    main()
