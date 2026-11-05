"""
Offline safety tests for the official experiment runner.

No GPU, training, Docker build, or official result files required.
All synthetic experiment artifacts live in temporary directories.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

sys.path.insert(0, str(SCRIPTS))

import run_official_matrix as runner


class OfficialRunnerSafetyTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        result_root = self.root / "results/official_v1"

        replacements = {
            "ROOT": self.root,
            "RESULT_ROOT": result_root,
            "HISTORY_ROOT": result_root / "_attempt_history",
        }

        for name, value in replacements.items():
            patcher = patch.object(runner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.cfg = {
            "seed": 17,
            "model": {"name": "lstm"},
            "data": {"batch_size": 256},
            "train": {"epochs": 20},
        }

        config_rel = (
            "experiments/official_v1/configs/"
            "lstm_seed17_c0.yaml"
        )

        config_path = self.root / config_rel
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_path.write_text(
            yaml.safe_dump(self.cfg),
            encoding="utf-8",
        )

        output_rel = (
            "results/official_v1/"
            "lstm_seed17_amp0_pin0_prefetch0_compile0"
        )

        self.run = {
            "run_id": "lstm_seed17_c0",
            "resolved_config": config_rel,
            "engine_run_dir": output_rel,
            "config_sha256": "synthetic-test-hash",
        }

        self.run_dir = self.root / output_rel

    def write_completed_artifacts(self):
        """Create synthetic completion artifacts, not a real model."""

        self.run_dir.mkdir(parents=True, exist_ok=True)

        name = self.run_dir.name

        runner.write_json(
            self.run_dir / "status.json",
            {
                "run": name,
                "status": "completed",
            },
        )

        runner.write_json(
            self.run_dir / "result.json",
            {
                "run": name,
                "status": "completed",
                "config": self.cfg,
            },
        )

        (self.run_dir / "best.pt").write_bytes(
            b"synthetic-checkpoint-marker"
        )

    def test_01_missing_run_is_pending(self):
        self.assertEqual(
            runner.get_run_state(self.run),
            "pending",
        )

    def test_02_completed_run_is_recognized(self):
        self.write_completed_artifacts()

        self.assertEqual(
            runner.get_run_state(self.run),
            "completed",
        )

    def test_03_completed_run_is_skipped(self):
        self.write_completed_artifacts()

        with patch.object(
            runner.subprocess,
            "run",
            side_effect=AssertionError(
                "Completed run must not launch training"
            ),
        ):
            runner.execute_one(
                self.run,
                retry_incomplete=False,
            )

        self.assertEqual(
            runner.get_run_state(self.run),
            "completed",
        )

    def test_04_incomplete_run_is_preserved(self):
        self.run_dir.mkdir(parents=True)

        marker = self.run_dir / "original_attempt.txt"

        marker.write_text(
            "DO NOT OVERWRITE",
            encoding="utf-8",
        )

        self.assertEqual(
            runner.get_run_state(self.run),
            "incomplete",
        )

        with self.assertRaises(RuntimeError):
            runner.execute_one(
                self.run,
                retry_incomplete=False,
            )

        self.assertEqual(
            marker.read_text(encoding="utf-8"),
            "DO NOT OVERWRITE",
        )

    def test_05_explicit_retry_archives_old_attempt(self):
        self.run_dir.mkdir(parents=True)

        old_marker = self.run_dir / "old_attempt.txt"

        old_marker.write_text(
            "PRESERVED HISTORY",
            encoding="utf-8",
        )

        def pretend_training(command, **kwargs):
            self.write_completed_artifacts()

            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
            )

        with patch.object(
            runner.subprocess,
            "run",
            side_effect=pretend_training,
        ) as fake:
            runner.execute_one(
                self.run,
                retry_incomplete=True,
            )

        self.assertEqual(fake.call_count, 1)

        archives = list(
            runner.HISTORY_ROOT.glob(
                "lstm_seed17_c0_*"
            )
        )

        self.assertEqual(len(archives), 1)

        self.assertEqual(
            (
                archives[0] / "old_attempt.txt"
            ).read_text(encoding="utf-8"),
            "PRESERVED HISTORY",
        )

        self.assertEqual(
            runner.get_run_state(self.run),
            "completed",
        )

    def test_06_failed_subprocess_preserves_evidence(self):
        def pretend_failure(command, **kwargs):
            return subprocess.CompletedProcess(
                args=command,
                returncode=17,
            )

        with patch.object(
            runner.subprocess,
            "run",
            side_effect=pretend_failure,
        ), self.assertRaises(RuntimeError):
            runner.execute_one(
                self.run,
                retry_incomplete=False,
            )

        status = json.loads(
            (
                self.run_dir / "runner_status.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["return_code"], 17)

        self.assertTrue(
            (self.run_dir / "stdout.log").is_file()
        )

        self.assertTrue(
            (self.run_dir / "stderr.log").is_file()
        )

        self.assertEqual(
            runner.get_run_state(self.run),
            "incomplete",
        )

    def test_07_success_without_artifacts_is_rejected(self):
        def pretend_empty_success(command, **kwargs):
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
            )

        with patch.object(
            runner.subprocess,
            "run",
            side_effect=pretend_empty_success,
        ), self.assertRaisesRegex(
            RuntimeError,
            "completed result",
        ):
            runner.execute_one(
                self.run,
                retry_incomplete=False,
            )

        status = runner.read_json(
            self.run_dir / "runner_status.json"
        )

        self.assertEqual(status["status"], "failed")

    def test_08_changed_completed_config_is_rejected(self):
        self.write_completed_artifacts()

        config_path = (
            self.root / self.run["resolved_config"]
        )

        modified = dict(self.cfg)
        modified["seed"] = 101

        config_path.write_text(
            yaml.safe_dump(modified),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Saved configuration mismatch",
        ):
            runner.get_run_state(self.run)

    def test_09_run_identity_mismatch_is_rejected(self):
        self.write_completed_artifacts()

        status_path = self.run_dir / "status.json"
        status = runner.read_json(status_path)

        status["run"] = "wrong_experiment"

        runner.write_json(status_path, status)

        with self.assertRaisesRegex(
            RuntimeError,
            "Run identity mismatch",
        ):
            runner.get_run_state(self.run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
