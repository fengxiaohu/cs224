#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded Auto Research controller for CS224N A2 learning-rate experiments."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import yaml

STUDENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_PATH = os.path.join(STUDENT_DIR, "program.md")
RESULTS_ROOT = os.path.join(STUDENT_DIR, "results")
SUMMARY_PATH = os.path.join(STUDENT_DIR, "experiment_summary.tsv")
CONCLUSION_PATH = os.path.join(STUDENT_DIR, "experiment_conclusion.md")
RUN_PY = os.path.join(STUDENT_DIR, "run.py")


class ControllerPause(Exception):
    """Raised when the batch must stop and wait for human review."""


def load_protocol():
    with open(PROGRAM_PATH, "r", encoding="utf-8") as handle:
        content = handle.read()
    match = re.search(r"```yaml\n(.*?)```", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find YAML protocol block in program.md")
    return yaml.safe_load(match.group(1))["protocol"]


def short_code_hash(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return ""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    commit = config.get("git", {}).get("commit")
    if commit:
        return commit[:7]
    hashes = config.get("source_hashes", {})
    if "run.py" in hashes:
        return hashes["run.py"][:7]
    return ""


def read_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_status(run_dir):
    status_path = os.path.join(run_dir, "status.json")
    with open(status_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_summary_row(row):
    header = [
        "run_id",
        "code_hash",
        "seed",
        "mode",
        "lr",
        "epochs",
        "batch",
        "elapsed",
        "final_train_loss",
        "best_dev_UAS",
        "best_epoch",
        "test_UAS",
        "checkpoint",
        "status",
    ]
    write_header = not os.path.exists(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in header})


def build_run_id(phase_id, role, seed, lr):
    lr_tag = str(lr).replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "exp_{}_{}_seed{}_lr{}_{}".format(phase_id, role, seed, lr_tag, timestamp)


def launch_run(python_executable, run_spec, timeout_seconds):
    run_id = build_run_id(run_spec["phase_id"], run_spec["role"], run_spec["seed"], run_spec["lr"])
    cmd = [
        python_executable,
        RUN_PY,
        "--run-id",
        run_id,
        "--lr",
        str(run_spec["lr"]),
        "--seed",
        str(run_spec["seed"]),
    ]
    if run_spec.get("mode") == "debug":
        cmd.append("-d")
    if run_spec.get("skip_test", True):
        cmd.append("--skip-test")

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=STUDENT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        run_dir = os.path.join(RESULTS_ROOT, run_id)
        status_path = os.path.join(run_dir, "status.json")
        if os.path.exists(status_path):
            status = read_status(run_dir)
        else:
            status = {"status": "timeout", "failure_reason": "controller timeout"}
        status.update(
            {
                "status": "timeout",
                "failure_reason": "controller timeout after {}s".format(timeout_seconds),
                "finished_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        if os.path.exists(run_dir):
            with open(status_path, "w", encoding="utf-8") as handle:
                json.dump(status, handle, indent=2)
                handle.write("\n")
        append_summary_row(
            {
                "run_id": run_id,
                "seed": run_spec["seed"],
                "mode": run_spec.get("mode", "full"),
                "lr": run_spec["lr"],
                "epochs": 10,
                "batch": 1024,
                "elapsed": round(time.time() - started, 2),
                "status": "timeout",
                "checkpoint": os.path.join(run_dir, "model.weights"),
            }
        )
        raise ControllerPause("Run {} timed out after {} seconds".format(run_id, timeout_seconds))

    run_dir = os.path.join(RESULTS_ROOT, run_id)
    elapsed = round(time.time() - started, 2)
    if proc.returncode != 0:
        append_summary_row(
            {
                "run_id": run_id,
                "seed": run_spec["seed"],
                "mode": run_spec.get("mode", "full"),
                "lr": run_spec["lr"],
                "epochs": 10,
                "batch": 1024,
                "elapsed": elapsed,
                "status": "failed",
                "checkpoint": os.path.join(run_dir, "model.weights"),
            }
        )
        tail = "\n".join(stdout.splitlines()[-20:])
        raise ControllerPause(
            "Run {} failed with exit code {}.\nLast output:\n{}".format(
                run_id, proc.returncode, tail
            )
        )

    metrics = read_metrics(run_dir)
    status = read_status(run_dir)
    checkpoint = metrics.get("checkpoint", os.path.join(run_dir, "model.weights"))
    if not os.path.exists(checkpoint):
        append_summary_row(
            {
                "run_id": run_id,
                "code_hash": short_code_hash(run_dir),
                "seed": run_spec["seed"],
                "mode": run_spec.get("mode", "full"),
                "lr": run_spec["lr"],
                "epochs": 10,
                "batch": 1024,
                "elapsed": elapsed,
                "final_train_loss": metrics.get("final_train_loss", ""),
                "best_dev_UAS": metrics.get("best_dev_uas_percent", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "test_UAS": metrics.get("test_uas_percent", ""),
                "checkpoint": checkpoint,
                "status": "missing_checkpoint",
            }
        )
        raise ControllerPause("Run {} completed without checkpoint".format(run_id))

    append_summary_row(
        {
            "run_id": run_id,
            "code_hash": short_code_hash(run_dir),
            "seed": run_spec["seed"],
            "mode": run_spec.get("mode", "full"),
            "lr": run_spec["lr"],
            "epochs": 10,
            "batch": 1024,
            "elapsed": metrics.get("elapsed_seconds", elapsed),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "best_dev_UAS": metrics.get("best_dev_uas_percent", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "test_UAS": metrics.get("test_uas_percent", ""),
            "checkpoint": checkpoint,
            "status": status.get("status", "completed"),
        }
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "metrics": metrics,
        "elapsed": elapsed,
        "role": run_spec["role"],
        "seed": run_spec["seed"],
        "lr": run_spec["lr"],
        "mode": run_spec.get("mode", "full"),
    }


def qualifies(metrics, mode, protocol):
    final_loss = metrics.get("final_train_loss")
    best_uas = metrics.get("best_dev_uas")
    if final_loss is None or best_uas is None:
        return False
    if not math.isfinite(final_loss) or not math.isfinite(best_uas):
        return False
    if mode == "debug":
        return (
            final_loss < protocol["qualification"]["debug_final_train_loss_lt"]
            and best_uas * 100.0 > protocol["qualification"]["debug_best_dev_uas_percent_gt"]
        )
    return (
        final_loss < protocol["qualification"]["full_final_train_loss_lt"]
        and best_uas * 100.0 > protocol["qualification"]["full_best_dev_uas_percent_gt"]
    )


def compare_runs(baseline, candidate):
    if candidate["metrics"]["best_dev_uas"] > baseline["metrics"]["best_dev_uas"]:
        return candidate
    if candidate["metrics"]["best_dev_uas"] < baseline["metrics"]["best_dev_uas"]:
        return baseline
    if candidate["role"].startswith("candidate"):
        return candidate
    return baseline


def write_conclusion(protocol, phase_b_results, winner, batch_preferred, extra_seconds):
    lines = [
        "# A2 Auto Research Conclusion",
        "",
        "Submission baseline remains the archived pre-seed run: **best dev 88.60%**, **test 89.03%**.",
        "",
        "## Phase B (seed 42)",
        "",
    ]
    for result in phase_b_results:
        metrics = result["metrics"]
        lines.append(
            "- {} lr={} best_dev_UAS={:.4f}% final_train_loss={:.6f} elapsed={:.1f}s".format(
                result["role"],
                result["lr"],
                metrics["best_dev_uas"] * 100.0,
                metrics["final_train_loss"],
                result["elapsed"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if batch_preferred:
        lines.append(
            "- Batch-preferred config: lr={} (beats baseline on seeds 42 and 43).".format(
                batch_preferred["lr"]
            )
        )
        lines.append("- Submission report still uses the archived 88.60 / 89.03 numbers.")
    elif winner and winner["role"] != "baseline":
        lines.append(
            "- Candidate lr={} beat baseline on seed 42 but did not pass two-seed confirmation.".format(
                winner["lr"]
            )
        )
        lines.append("- Keep default lr=0.0005 for this batch.")
    else:
        lines.append("- No configuration beat the seed-42 baseline.")
        lines.append("- Keep default lr=0.0005 for this batch.")
    lines.append("- Additional experiment time: {:.1f} minutes.".format(extra_seconds / 60.0))
    with open(CONCLUSION_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_controller(python_executable=None, dry_run=False):
    protocol = load_protocol()
    python_executable = python_executable or sys.executable
    budgets = protocol["budgets"]
    full_elapsed = 0.0
    batch_started = time.time()
    phase_b_results = []
    baseline_result = None
    winner = None
    batch_preferred = None
    seed43_baseline = None
    seed43_candidate = None

    for phase in protocol["phases"]:
        phase_id = phase["id"]
        if phase_id == "C" and winner is None:
            continue
        for run_template in phase["runs"]:
            run_spec = dict(run_template)
            run_spec["phase_id"] = phase_id
            if run_spec.get("lr") == "winner_lr_from_phase_B":
                if winner is None:
                    continue
                run_spec["lr"] = winner["lr"]
            if dry_run:
                print("[dry-run]", run_spec)
                continue

            timeout = (
                budgets["debug_timeout_seconds"]
                if run_spec.get("mode") == "debug"
                else budgets["full_timeout_seconds"]
            )
            result = launch_run(python_executable, run_spec, timeout)

            if run_spec.get("mode") != "debug":
                full_elapsed += result["elapsed"]
                if full_elapsed > budgets["full_cumulative_timeout_seconds"]:
                    raise ControllerPause(
                        "Full-run cumulative budget exceeded ({}s)".format(full_elapsed)
                    )

            if phase_id == "A":
                if not qualifies(result["metrics"], "debug", protocol):
                    raise ControllerPause(
                        "Debug smoke run failed qualification thresholds"
                    )
            elif phase_id == "B":
                phase_b_results.append(result)
                if result["role"] == "baseline":
                    baseline_result = result
                    if not qualifies(result["metrics"], "full", protocol):
                        raise ControllerPause(
                            "Seed-42 baseline missed full-data reference targets"
                        )
                elif baseline_result is not None and qualifies(result["metrics"], "full", protocol):
                    if (
                        result["metrics"]["best_dev_uas"]
                        > baseline_result["metrics"]["best_dev_uas"]
                    ):
                        if (
                            winner is None
                            or result["metrics"]["best_dev_uas"]
                            > winner["metrics"]["best_dev_uas"]
                        ):
                            winner = result
            elif phase_id == "C":
                if result["role"] == "baseline":
                    seed43_baseline = result
                else:
                    seed43_candidate = result

        if phase_id == "C" and seed43_baseline and seed43_candidate and winner is not None:
            if (
                qualifies(seed43_baseline["metrics"], "full", protocol)
                and qualifies(seed43_candidate["metrics"], "full", protocol)
                and seed43_candidate["metrics"]["best_dev_uas"]
                > seed43_baseline["metrics"]["best_dev_uas"]
                and winner["metrics"]["best_dev_uas"]
                > baseline_result["metrics"]["best_dev_uas"]
            ):
                batch_preferred = seed43_candidate

    if not dry_run:
        write_conclusion(
            protocol,
            phase_b_results,
            winner,
            batch_preferred,
            time.time() - batch_started,
        )
    return {
        "phase_b_results": phase_b_results,
        "winner": winner,
        "batch_preferred": batch_preferred,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run bounded A2 Auto Research batch")
    parser.add_argument("--dry-run", action="store_true", help="print planned runs only")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch run.py",
    )
    args = parser.parse_args(argv)
    try:
        run_controller(python_executable=args.python, dry_run=args.dry_run)
    except ControllerPause as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
