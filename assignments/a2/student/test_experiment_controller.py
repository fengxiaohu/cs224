#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the bounded experiment controller using a mock trainer."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

STUDENT_DIR = os.path.dirname(os.path.abspath(__file__))
if STUDENT_DIR not in sys.path:
    sys.path.insert(0, STUDENT_DIR)

import experiment_controller as controller


class MockTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.results_root = os.path.join(self.tempdir.name, "results")
        os.makedirs(self.results_root)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_run(self, run_id, metrics, status="completed"):
        run_dir = os.path.join(self.results_root, run_id)
        os.makedirs(run_dir)
        checkpoint = os.path.join(run_dir, "model.weights")
        with open(checkpoint, "w", encoding="utf-8") as handle:
            handle.write("mock")
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle)
        with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": status}, handle)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"git": {"commit": "abc1234567890"}}, handle)
        return run_dir

    def _mock_launch_factory(self, sequence):
        calls = {"index": 0}

        def fake_launch(python_executable, run_spec, timeout_seconds):
            idx = calls["index"]
            calls["index"] += 1
            payload = sequence[idx]
            run_id = "mock_{}".format(idx)
            run_dir = self._write_run(run_id, payload["metrics"], payload.get("status", "completed"))
            return {
                "run_id": run_id,
                "run_dir": run_dir,
                "metrics": payload["metrics"],
                "elapsed": payload.get("elapsed", 1.0),
                "role": run_spec["role"],
                "seed": run_spec["seed"],
                "lr": run_spec["lr"],
                "mode": run_spec.get("mode", "full"),
            }

        return fake_launch

    @mock.patch("experiment_controller.write_conclusion")
    @mock.patch("experiment_controller.launch_run")
    @mock.patch("experiment_controller.RESULTS_ROOT")
    def test_phase_b_runs_in_order_without_phase_c_when_no_winner(
        self, mock_results_root, mock_launch, mock_conclusion
    ):
        mock_results_root = self.results_root
        sequence = [
            {"metrics": {"final_train_loss": 0.15, "best_dev_uas": 0.71}, "elapsed": 10},
            {"metrics": {"final_train_loss": 0.06, "best_dev_uas": 0.886}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.07, "best_dev_uas": 0.880}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.08, "best_dev_uas": 0.885}, "elapsed": 100},
        ]
        mock_launch.side_effect = self._mock_launch_factory(sequence)

        with mock.patch.object(controller, "RESULTS_ROOT", self.results_root):
            outcome = controller.run_controller(python_executable=sys.executable, dry_run=False)

        self.assertEqual(mock_launch.call_count, 4)
        self.assertEqual(len(outcome["phase_b_results"]), 3)
        self.assertIsNone(outcome["winner"])
        self.assertIsNone(outcome["batch_preferred"])
        roles = [call.args[1]["role"] for call in mock_launch.call_args_list]
        self.assertEqual(
            roles,
            ["debug_smoke", "baseline", "candidate_low_lr", "candidate_high_lr"],
        )

    @mock.patch("experiment_controller.write_conclusion")
    @mock.patch("experiment_controller.launch_run")
    def test_phase_c_runs_when_candidate_beats_baseline(self, mock_launch, mock_conclusion):
        sequence = [
            {"metrics": {"final_train_loss": 0.15, "best_dev_uas": 0.71}, "elapsed": 10},
            {"metrics": {"final_train_loss": 0.06, "best_dev_uas": 0.886}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.05, "best_dev_uas": 0.890}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.07, "best_dev_uas": 0.884}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.06, "best_dev_uas": 0.887}, "elapsed": 100},
            {"metrics": {"final_train_loss": 0.05, "best_dev_uas": 0.891}, "elapsed": 100},
        ]
        mock_launch.side_effect = self._mock_launch_factory(sequence)

        with mock.patch.object(controller, "RESULTS_ROOT", self.results_root):
            outcome = controller.run_controller(python_executable=sys.executable, dry_run=False)

        self.assertEqual(mock_launch.call_count, 6)
        self.assertIsNotNone(outcome["winner"])
        self.assertIsNotNone(outcome["batch_preferred"])

    def test_dry_run_prints_planned_runs(self):
        with mock.patch("builtins.print") as mock_print:
            controller.run_controller(dry_run=True)
        printed = repr(mock_print.call_args_list)
        self.assertIn("'role': 'debug_smoke'", printed)
        self.assertIn("'role': 'candidate_high_lr'", printed)
        self.assertEqual(printed.count("[dry-run]"), 4)


class RunCliTests(unittest.TestCase):
    def test_duplicate_run_id_rejected(self):
        run_id = "duplicate_run_id_test"
        run_dir = os.path.join(STUDENT_DIR, "results", run_id)
        os.makedirs(run_dir, exist_ok=True)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(STUDENT_DIR, "run.py"),
                    "--run-id",
                    run_id,
                    "-d",
                    "--skip-test",
                ],
                cwd=STUDENT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("already exists", proc.stdout)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
