#!/usr/bin/env python
"""Run a grid of ``run_experiment.py`` cells sequentially and report failures.

    .conda-env/bin/python reproduction/scripts/run_grid.py \
        --datasets cora --partitions ccp oslom --models GCN GAT SAGE --seeds 4 5

Cells run one at a time so that the recorded wall-clock timings are not distorted
by contention on the single GPU.  Every cell is attempted even if an earlier one
fails, and the exit status is non-zero if any cell failed -- a partially failed
grid must not look like a clean sweep.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PYTHON = os.path.join(REPO_ROOT, ".conda-env", "bin", "python")
RUNNER = os.path.join(REPO_ROOT, "reproduction", "scripts", "run_experiment.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--partitions", nargs="+", default=["ccp"])
    parser.add_argument("--models", nargs="+", default=["GCN", "GAT", "SAGE"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--unlearn_ratio", type=float, default=0.005)
    parser.add_argument("--agg_feat", default="pca", choices=["pca", "mean"])
    parser.add_argument("--ccp_theta", type=int, default=20)
    parser.add_argument("--ccp_max_community_size", type=int, default=0)
    parser.add_argument("--ccp_tail_min_size", type=int, default=0)
    parser.add_argument("--gae_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--protect", default="none",
                        choices=["none", "test", "test_val"],
                        help="forwarded as --ccp_protect_eval_nodes")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra flags forwarded verbatim to run_experiment.py")
    options = parser.parse_args()

    cells = list(itertools.product(
        options.datasets, options.partitions, options.models, options.seeds
    ))
    print("running {} cell(s)".format(len(cells)), flush=True)

    outcomes = []
    for index, (dataset, partition, model, seed) in enumerate(cells, start=1):
        command = [
            options.python, RUNNER,
            "--dataset", dataset,
            "--partition", partition,
            "--model", model,
            "--seed", str(seed),
            "--unlearn_ratio", str(options.unlearn_ratio),
            "--cuda", str(options.cuda),
            "--ccp_protect_eval_nodes", options.protect,
            "--agg_feat", options.agg_feat,
            "--ccp_theta", str(options.ccp_theta),
            "--ccp_max_community_size", str(options.ccp_max_community_size),
            "--ccp_tail_min_size", str(options.ccp_tail_min_size),
            "--gae_device", options.gae_device,
        ] + list(options.extra)
        if options.timeout:
            command += ["--timeout", str(options.timeout)]
        label = "{}/{}{}{}/{}/seed{}".format(
            dataset, partition,
            "" if options.protect == "none" else "+hold-" + options.protect,
            "" if options.agg_feat == "pca" else "+feat-" + options.agg_feat,
            model, seed)
        print("\n=== [{}/{}] {} ===".format(index, len(cells), label), flush=True)
        started = time.time()
        completed = subprocess.run(command, cwd=REPO_ROOT)
        elapsed = time.time() - started
        outcomes.append({
            "cell": label,
            "returncode": completed.returncode,
            "seconds": elapsed,
        })
        print("--- {} rc={} in {:.1f}s".format(label, completed.returncode, elapsed),
              flush=True)

    failures = [outcome for outcome in outcomes if outcome["returncode"] != 0]
    print("\n{} / {} cells succeeded".format(len(outcomes) - len(failures), len(outcomes)))
    for failure in failures:
        print("FAILED {}".format(failure["cell"]))
    summary_path = os.path.join(
        REPO_ROOT, "reproduction", "results",
        "grid_status_{}_{}_{}.json".format(
            "-".join(options.datasets), options.protect, options.agg_feat))
    with open(summary_path, "w") as handle:
        json.dump(outcomes, handle, indent=2, sort_keys=True)
    print("wrote {}".format(os.path.relpath(summary_path, REPO_ROOT)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
