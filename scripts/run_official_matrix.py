"""
Pre-18: Failure-safe official experiment runner.

Default: dry-run only.

Execution requires:
  --execute
  GPUFORECAST_OFFICIAL_RUNS=YES
  An NVIDIA A40 CUDA device

Incomplete attempts are never silently overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from verify_matrix import main as verify_matrix

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "experiments/official_v1/manifest.json"
RESULT_ROOT = ROOT / "results/official_v1"
HISTORY_ROOT = RESULT_ROOT / "_attempt_history"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan():
    # Independently validates all 60 configs and hashes.
    verify_matrix()

    manifest = read_json(MANIFEST)
    return manifest["runs"]


def get_run_state(run):
    run_dir = ROOT / run["engine_run_dir"]

    if not run_dir.exists():
        return "pending"

    if not run_dir.is_dir():
        raise RuntimeError(
            f"Expected directory, found different path: {run_dir}"
        )

    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "best.pt"

    if not all(
        p.is_file()
        for p in (
            status_path,
            result_path,
            checkpoint_path,
        )
    ):
        return "incomplete"

    try:
        status = read_json(status_path)
        result = read_json(result_path)
    except (OSError, ValueError):
        return "incomplete"

    if (
        status.get("status") != "completed"
        or result.get("status") != "completed"
    ):
        return "incomplete"

    expected_name = run_dir.name

    if (
        status.get("run") != expected_name
        or result.get("run") != expected_name
    ):
        raise RuntimeError(
            f"Run identity mismatch: {run['run_id']}"
        )

    cfg_path = ROOT / run["resolved_config"]

    expected_cfg = yaml.safe_load(
        cfg_path.read_text(encoding="utf-8")
    )

    if result.get("config") != expected_cfg:
        raise RuntimeError(
            f"Saved configuration mismatch: {run['run_id']}"
        )

    return "completed"


@contextmanager
def campaign_lock():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    lock_path = RESULT_ROOT / ".official_runner.lock"

    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another official experiment runner holds the lock."
            ) from exc

        try:
            yield
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


def archive_incomplete_attempt(run):
    run_dir = ROOT / run["engine_run_dir"]

    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    archive = HISTORY_ROOT / f"{run['run_id']}_{stamp}"

    if archive.exists():
        raise RuntimeError(
            f"Refusing archive collision: {archive}"
        )

    run_dir.rename(archive)

    print(f"ARCHIVED prior attempt: {archive.relative_to(ROOT)}")


def check_gpu_execution_allowed():
    if os.environ.get("GPUFORECAST_OFFICIAL_RUNS") != "YES":
        raise RuntimeError(
            "Set GPUFORECAST_OFFICIAL_RUNS=YES to authorize "
            "official campaign execution."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Official training requires an NVIDIA GPU."
        )

    gpu_name = torch.cuda.get_device_name(0)

    if "A40" not in gpu_name:
        raise RuntimeError(
            f"Expected NVIDIA A40, found: {gpu_name}"
        )

    if not (ROOT / "data/ETTm1.csv").is_file():
        raise RuntimeError(
            "Missing data/ETTm1.csv"
        )

    print(f"GPU preflight: {gpu_name}")


def execute_one(run, retry_incomplete):
    run_dir = ROOT / run["engine_run_dir"]

    state = get_run_state(run)

    if state == "completed":
        print(f"SKIP completed: {run['run_id']}")
        return

    if state == "incomplete":
        if not retry_incomplete:
            raise RuntimeError(
                f"Incomplete attempt: {run['run_id']}. "
                "Preserved existing files. "
                "Use --retry-incomplete explicitly to archive "
                "the old attempt before rerunning."
            )

        archive_incomplete_attempt(run)

    # No existing run directory may be reused.
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = ROOT / run["resolved_config"]

    command = [
        sys.executable,
        "-m",
        "gpuforecast.train",
        "--config",
        str(config_path),
    ]

    runner_status = {
        "run_id": run["run_id"],
        "config_sha256": run["config_sha256"],
        "started_utc": now_utc(),
        "status": "running",
        "command": command,
        "container_image_digest": os.environ.get(
            "GPUFORECAST_IMAGE_DIGEST"
        ),
    }

    status_path = run_dir / "runner_status.json"

    write_json(status_path, runner_status)

    print(f"START: {run['run_id']}", flush=True)

    try:
        with (
            (run_dir / "stdout.log").open(
                "w", encoding="utf-8"
            ) as stdout,
            (run_dir / "stderr.log").open(
                "w", encoding="utf-8"
            ) as stderr,
        ):
            proc = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )

        runner_status["return_code"] = proc.returncode

        if proc.returncode != 0:
            raise RuntimeError(
                f"Training exited with code {proc.returncode}"
            )

        if get_run_state(run) != "completed":
            raise RuntimeError(
                "Training returned success but completed "
                "result, status, or checkpoint is missing."
            )

        runner_status["status"] = "completed"

        print(f"COMPLETE: {run['run_id']}", flush=True)

    except BaseException as exc:
        runner_status["status"] = "failed"
        runner_status["error"] = str(exc)

        print(
            f"FAILED: {run['run_id']} | {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise

    finally:
        runner_status["finished_utc"] = now_utc()
        write_json(status_path, runner_status)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run training; default is dry-run",
    )

    parser.add_argument(
        "--run-id",
        help="Select exactly one manifest run",
    )

    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Archive a prior incomplete attempt before retry",
    )

    args = parser.parse_args()

    if args.retry_incomplete and not args.execute:
        parser.error(
            "--retry-incomplete requires --execute"
        )

    runs = load_plan()

    if args.run_id:
        runs = [
            run for run in runs
            if run["run_id"] == args.run_id
        ]

        if len(runs) != 1:
            parser.error(
                f"Unknown run ID: {args.run_id}"
            )

    if not args.execute:
        states = Counter()

        print("\n========== OFFICIAL DRY RUN ==========")

        for run in runs:
            state = get_run_state(run)
            states[state] += 1

            print(
                f"{run['run_id']:30s} "
                f"{state:12s} "
                f"{run['resolved_config']}"
            )

        print("\n========== SUMMARY ==========")

        for name in ("pending", "completed", "incomplete"):
            print(f"{name}: {states[name]}")

        print("\nNO TRAINING WAS LAUNCHED.")
        return

    check_gpu_execution_allowed()

    with campaign_lock():
        for run in runs:
            execute_one(run, args.retry_incomplete)

    print("\nSelected official experiments processed.")


if __name__ == "__main__":
    main()
