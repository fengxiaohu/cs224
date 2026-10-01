"""Module E: GloVe analogy evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from config import ANALOGY_SET_PATH
from src.analogy import run_analogy_evaluation
from src.glove_utils import load_glove
from src.run_utils import load_json, make_run_dir, save_config, save_json


def main() -> Path:
    run_dir = make_run_dir("module_e")
    analogy_set = load_json(ANALOGY_SET_PATH)
    save_json(run_dir / "analogy_set.json", analogy_set)

    wv = load_glove()
    eval_out = run_analogy_evaluation(wv, analogy_set)
    save_json(run_dir / "module_e_analogy.json", eval_out)

    save_config(
        run_dir,
        {
            "module": "E",
            "analogy_set_path": str(ANALOGY_SET_PATH),
            "glove_model": "glove-wiki-gigaword-200",
        },
    )
    print(f"Module E complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
