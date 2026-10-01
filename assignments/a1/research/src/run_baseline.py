"""Export and execute baseline notebook from f97ea793."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RESEARCH_DIR.parents[1]
BASELINE_DIR = RESEARCH_DIR / "results" / "baseline"
PYTHON = sys.executable
COMMIT = "f97ea793"
NOTEBOOK_SRC = "assignments/a1/student/exploring_word_vectors.ipynb"


def export_notebook() -> Path:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINE_DIR / "exploring_word_vectors.ipynb"
    result = subprocess.run(
        ["git", "show", f"{COMMIT}:{NOTEBOOK_SRC}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    out_path.write_text(result.stdout, encoding="utf-8")
    return out_path


def execute_notebook(notebook_path: Path) -> dict:
    stdout_path = BASELINE_DIR / "baseline_stdout.txt"
    executed_path = BASELINE_DIR / "exploring_word_vectors_executed.ipynb"

    import shutil
    jupyter_bin = shutil.which("jupyter", path=str(Path(PYTHON).parent))
    if jupyter_bin is None:
        jupyter_bin = str(Path(PYTHON).parent / "jupyter")
    cmd = [
        jupyter_bin,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        str(executed_path.name),
        "--ExecutePreprocessor.timeout=600",
        f"--ExecutePreprocessor.kernel_name=python3",
    ]
    env = dict(**__import__("os").environ)
    conda_bin = str(Path(PYTHON).parent)
    env["PATH"] = conda_bin + ":" + env.get("PATH", "")
    proc = subprocess.run(
        cmd,
        cwd=BASELINE_DIR,
        capture_output=True,
        text=True,
        env=env,
    )
    stdout_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    status = {
        "returncode": proc.returncode,
        "stdout_path": str(stdout_path),
        "executed_notebook": str(executed_path),
        "success": proc.returncode == 0,
    }

    mismatches = {
        "num_samples_vs_text": "NUM_SAMPLES=150 but text says 25,000 train reviews",
        "corpus_size_print": "prints len(imdb_corpus[0]) not document count",
        "glove_sample_size": "text says 40,000 but code loads all ~400,000 vectors",
        "svd_random_state": "TruncatedSVD without random_state depends on global numpy seed",
        "dense_co_occurrence": "uses np.zeros dense matrix; scales poorly",
        "empty_string_tokens": "preprocessing creates empty-string tokens from punctuation",
    }
    mismatches_path = BASELINE_DIR / "notebook_mismatches.json"
    mismatches_path.write_text(json.dumps(mismatches, indent=2), encoding="utf-8")
    status["mismatches_path"] = str(mismatches_path)
    return status


def main() -> None:
    nb = export_notebook()
    print(f"Exported baseline notebook to {nb}")
    status = execute_notebook(nb)
    status_path = BASELINE_DIR / "baseline_run_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if not status["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
