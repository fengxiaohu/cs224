#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024: Homework 2
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

DEFAULT_LR = 0.0005
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 1024
DEFAULT_N_EPOCHS = 10
DEFAULT_HIDDEN_SIZE = 200
DEFAULT_DROPOUT = 0.5
STUDENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(STUDENT_DIR, "results")
SOURCE_FILES = [
    "run.py",
    "parser_model.py",
    "parser_transitions.py",
    "utils/parser_utils.py",
    "utils/general_utils.py",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
    parser.add_argument('-d', '--debug', action='store_true', help='whether to enter debug mode')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='learning rate')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='random seed')
    parser.add_argument('--run-id', type=str, default=None, help='unique run directory name')
    parser.add_argument('--skip-test', action='store_true', help='skip test evaluation after training')
    parser.add_argument(
        '--eval-test',
        type=str,
        default=None,
        metavar='RUN_DIR',
        help='load a saved run and evaluate on test only',
    )
    return parser


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_metadata():
    repo_root = os.path.abspath(os.path.join(STUDENT_DIR, "..", "..", ".."))
    metadata = {
        "commit": None,
        "branch": None,
        "status_porcelain": None,
        "diff_stat": None,
    }
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        metadata["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True
        ).strip()
        metadata["status_porcelain"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        metadata["diff_stat"] = subprocess.check_output(
            ["git", "diff", "--stat"], cwd=repo_root, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return metadata


def collect_source_hashes():
    hashes = {}
    for rel_path in SOURCE_FILES:
        abs_path = os.path.join(STUDENT_DIR, rel_path)
        if os.path.exists(abs_path):
            hashes[rel_path] = sha256_file(abs_path)
    return hashes


def snapshot_source_files(output_dir):
    snapshot_dir = os.path.join(output_dir, "source_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for rel_path in SOURCE_FILES:
        src = os.path.join(STUDENT_DIR, rel_path)
        if not os.path.exists(src):
            continue
        dst = os.path.join(snapshot_dir, rel_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def resolve_run_dir(run_id):
    return os.path.join(RESULTS_ROOT, run_id)


def ensure_unique_run_dir(run_id):
    output_dir = resolve_run_dir(run_id)
    if os.path.exists(output_dir):
        raise FileExistsError("Run directory already exists: {}".format(output_dir))
    os.makedirs(output_dir)
    return output_dir


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_run_artifacts(output_dir, config, metrics, status):
    write_json(os.path.join(output_dir, "config.json"), config)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_json(os.path.join(output_dir, "status.json"), status)


def save_parser_bundle(output_dir, parser_obj, embeddings):
    bundle_path = os.path.join(output_dir, "parser_bundle.pkl")
    with open(bundle_path, "wb") as handle:
        pickle.dump({"parser": parser_obj, "embeddings": embeddings}, handle)


def load_parser_bundle(run_dir):
    bundle_path = os.path.join(run_dir, "parser_bundle.pkl")
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)
    return bundle["parser"], bundle["embeddings"]


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_config(args, run_id, output_dir, debug):
    return {
        "run_id": run_id,
        "mode": "debug" if debug else "full",
        "lr": args.lr,
        "seed": args.seed,
        "batch_size": DEFAULT_BATCH_SIZE,
        "n_epochs": DEFAULT_N_EPOCHS,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "dropout": DEFAULT_DROPOUT,
        "device": "cpu",
        "skip_test": args.skip_test,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "output_dir": output_dir,
        "git": collect_git_metadata(),
        "source_hashes": collect_source_hashes(),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }


# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate

    @return dict: training summary with epoch metrics and best checkpoint info
    """
    best_dev_UAS = 0.0
    best_epoch = None
    epoch_metrics = []

    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func` with `mean`
    ###         reduction (default)
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        train_loss, dev_UAS = train_for_epoch(
            parser, train_data, dev_data, optimizer, loss_func, batch_size
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "dev_uas": dev_UAS,
                "dev_uas_percent": dev_UAS * 100.0,
            }
        )
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            best_epoch = epoch + 1
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")

    final_train_loss = epoch_metrics[-1]["train_loss"] if epoch_metrics else float("nan")
    return {
        "epoch_metrics": epoch_metrics,
        "best_dev_uas": best_dev_UAS,
        "best_dev_uas_percent": best_dev_UAS * 100.0,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "checkpoint": output_path,
    }


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size

    @return tuple: (average train loss, dev UAS)
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return loss_meter.avg, dev_UAS


def evaluate_test(parser_obj, checkpoint_path):
    parser_obj.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    parser_obj.model.eval()
    print("Final evaluation on test set",)
    test_UAS, _ = parser_obj.parse(test_data_holder["data"])
    print("- test UAS: {:.2f}".format(test_UAS * 100.0))
    return test_UAS


test_data_holder = {"data": None}


def run_training(args):
    debug = args.debug
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_unique_run_dir(run_id)
    output_path = os.path.join(output_dir, "model.weights")
    log_path = os.path.join(output_dir, "run.log")

    config = build_config(args, run_id, output_dir, debug)
    status = {
        "run_id": run_id,
        "status": "running",
        "started_at": config["started_at"],
        "finished_at": None,
        "failure_reason": None,
    }
    save_run_artifacts(output_dir, config, {"epoch_metrics": []}, status)
    snapshot_source_files(output_dir)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    log_handle = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)

    metrics = {}
    try:
        assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

        set_global_seed(args.seed)

        print(80 * "=")
        print("INITIALIZING")
        print(80 * "=")
        parser_obj, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
        test_data_holder["data"] = test_data
        save_parser_bundle(output_dir, parser_obj, embeddings)

        start = time.time()
        model = ParserModel(
            embeddings,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout_prob=DEFAULT_DROPOUT,
        )
        parser_obj.model = model
        print("took {:.2f} seconds\n".format(time.time() - start))

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")

        train_start = time.time()
        metrics = train(
            parser_obj,
            train_data,
            dev_data,
            output_path,
            batch_size=DEFAULT_BATCH_SIZE,
            n_epochs=DEFAULT_N_EPOCHS,
            lr=args.lr,
        )
        metrics["elapsed_seconds"] = time.time() - train_start

        if not math.isfinite(metrics["final_train_loss"]):
            raise ValueError("Final train loss is not finite: {}".format(metrics["final_train_loss"]))
        if not os.path.exists(output_path):
            raise FileNotFoundError("Best checkpoint missing: {}".format(output_path))

        test_UAS = None
        if not debug and not args.skip_test:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            test_UAS = evaluate_test(parser_obj, output_path)
            metrics["test_uas"] = test_UAS
            metrics["test_uas_percent"] = test_UAS * 100.0
            print("Done!")

        status.update(
            {
                "status": "completed",
                "finished_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        config["finished_at"] = status["finished_at"]
        save_run_artifacts(output_dir, config, metrics, status)
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": datetime.utcnow().isoformat() + "Z",
                "failure_reason": str(exc),
            }
        )
        save_run_artifacts(output_dir, config, metrics, status)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def run_eval_test(run_dir):
    run_dir = os.path.abspath(run_dir)
    config = load_run_config(run_dir)
    checkpoint_path = os.path.join(run_dir, "model.weights")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint_path))

    set_global_seed(config.get("seed", DEFAULT_SEED))
    parser_obj, embeddings = load_parser_bundle(run_dir)
    debug = config.get("mode") == "debug"
    _, _, _, _, test_data = load_and_preprocess_data(debug)
    test_data_holder["data"] = test_data

    model = ParserModel(
        embeddings,
        hidden_size=config.get("hidden_size", DEFAULT_HIDDEN_SIZE),
        dropout_prob=config.get("dropout", DEFAULT_DROPOUT),
    )
    parser_obj.model = model

    print(80 * "=")
    print("TESTING")
    print(80 * "=")
    print("Restoring the best model weights found on the dev set")
    test_UAS = evaluate_test(parser_obj, checkpoint_path)

    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        metrics = {}
    metrics["test_uas"] = test_UAS
    metrics["test_uas_percent"] = test_UAS * 100.0
    write_json(metrics_path, metrics)
    print("Done!")
    return test_UAS


def main(argv=None):
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args(argv)

    try:
        if args.eval_test:
            run_eval_test(args.eval_test)
            return 0
        return run_training(args)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
