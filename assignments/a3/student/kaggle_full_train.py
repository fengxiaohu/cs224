#!/usr/bin/env python3
"""Kaggle GPU full A3 train + test. Looks up student/ under /kaggle/input."""

import os
import shutil
import subprocess
import sys

import torch


def find_student_root():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "run.py" in files and os.path.isdir(os.path.join(root, "zh_en_data")):
            return root
    if os.path.isfile("/kaggle/working/student/run.py"):
        return "/kaggle/working/student"
    raise SystemExit("Could not find run.py + zh_en_data under /kaggle/input")


def run(cmd, cwd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main():
    print("cuda available:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("GPU not available")
    print("device:", torch.cuda.get_device_name(0), flush=True)

    src = find_student_root()
    print("student root:", src, flush=True)
    dst = "/kaggle/working/student"
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))

    os.makedirs(os.path.join(dst, "outputs"), exist_ok=True)
    diag = os.path.join(dst, "beam_search_diagnostics.py")
    if os.path.isfile(diag):
        text = open(diag).read()
        text = text.replace(
            "os.getlogin()",
            '(os.environ.get("USER") or os.environ.get("USERNAME") or "kaggle")',
        )
        open(diag, "w").write(text)
    # Defense in depth: even an older dataset copy of run.py must allow Vocab unpickle.
    run_py = os.path.join(dst, "run.py")
    if os.path.isfile(run_py):
        text = open(run_py).read()
        if "weights_only=False" not in text:
            text = text.replace(
                "params = torch.load(model_save_path, map_location=lambda storage, loc: storage)",
                "params = torch.load(model_save_path, map_location=lambda storage, loc: storage, weights_only=False)",
            )
            open(run_py, "w").write(text)
            print("patched run.py torch.load weights_only=False", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "docopt", "sentencepiece", "sacrebleu", "nltk"]
    )
    run(
        [
            sys.executable,
            "run.py",
            "train",
            "--train-src=./zh_en_data/train.zh",
            "--train-tgt=./zh_en_data/train.en",
            "--dev-src=./zh_en_data/dev.zh",
            "--dev-tgt=./zh_en_data/dev.en",
            "--vocab=vocab.json",
            "--cuda",
            "--lr=5e-4",
            "--patience=1",
            "--valid-niter=200",
            "--batch-size=32",
            "--dropout=.3",
        ],
        cwd=dst,
    )
    run(
        [
            sys.executable,
            "run.py",
            "decode",
            "model.bin",
            "./zh_en_data/test.zh",
            "./zh_en_data/test.en",
            "outputs/test_outputs.txt",
            "--cuda",
        ],
        cwd=dst,
    )
    working = "/kaggle/working"
    for name in ("model.bin", "model.bin.optim"):
        src_path = os.path.join(dst, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(working, name))
            print("copied", name, flush=True)
    test_out = os.path.join(dst, "outputs", "test_outputs.txt")
    if os.path.isfile(test_out):
        shutil.copy2(test_out, os.path.join(working, "test_outputs.txt"))
        print("copied test_outputs.txt", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
