#!/usr/bin/env python
"""Run the CGE Partition -> Train -> Unlearn pipeline and record everything.

One invocation = one (dataset, partition method, backbone, seed) cell of the
result table.  Each cell gets its own ``CCGU_DATA_ROOT`` so that a second seed
cannot reload the first seed's cached community file (the artifact names in
``lib_dataset/data_store.py`` do not encode the seed).  Downloaded DGL datasets
stay shared through ``CCGU_DGL_DATA``.

    .conda-env/bin/python reproduction/scripts/run_experiment.py \
        --dataset cora --partition ccp --model GCN --seed 4

Every stage's stdout goes to ``reproduction/logs/<run-id>/<stage>.log`` and every
stage's metrics to ``reproduction/results/runs/<run-id>/<stage>.json``, so the
summaries that ``make_tables.py`` produces are assembled from locally generated
outputs.

Paper-faithful values are passed explicitly on the command line rather than by
changing any default in ``parameter_parser.py``:

    --agg_feat pca --agg_label th --agg_edge rubost   Eq. (5) / (9) / (11)
    --test_edge_method 0                             Eq. (11) exactly
    --train_lr 0.01 --train_weight_decay 0.001       Appendix B
    --num_epochs 200                                 Appendix B
    --unlearn_ratio 0.005                            0.5% per unlearning batch
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PYTHON = os.path.join(REPO_ROOT, ".conda-env", "bin", "python")
SHARED_DGL_DATA = os.path.join(REPO_ROOT, "temp_data", "dgl_data")

STAGES = ("Partition", "Train", "Unlearn")


#: Knobs whose non-default value must appear in the run id.  Without this, two
#: ablation cells that differ only by e.g. --ccp_theta would resolve to the same
#: artifact directory and silently overwrite one another.
RUN_ID_KNOBS = (
    ("agg_feat", "pca", "feat"),
    ("test_ratio", 0.2, "tr"),
    ("train_lr", 0.01, "lr"),
    ("train_weight_decay", 0.001, "wd"),
    ("num_epochs", 200, "ne"),
    ("partition_min_communities", 8, "minc"),
    ("partition_max_retries", 0, "retry"),
    ("ccp_resolution", 1.0, "res"),
    ("ccp_fine_resolution", 1.0, "fres"),
    ("ccp_theta", 20, "theta"),
    ("ccp_max_communities_per_node", 3, "k"),
    ("ccp_overlap_threshold", 0.35, "tau"),
    ("ccp_propagation_steps", 2, "prop"),
    ("ccp_max_community_size", 0, "cap"),
    ("ccp_tail_min_size", 0, "tail"),
    ("gae_epochs", 100, "ep"),
    ("gae_hidden_dim", 64, "hid"),
    ("gae_latent_dim", 16, "lat"),
    ("gae_device", "auto", "dev"),
)

#: Options that reach main.py but are already part of the run id, or that cannot
#: change a result.  Everything else forwarded must appear in RUN_ID_KNOBS.
RUN_ID_EXEMPT = frozenset({
    "dataset", "partition", "model", "seed", "unlearn_ratio",
    "ccp_protect_eval_nodes",          # encoded as the _hold<mode> suffix
    "cuda", "num_threads", "python", "stages", "keep_data_root", "timeout",
})


def run_id_for(options):
    protect = getattr(options, "ccp_protect_eval_nodes", "none")
    suffix = "" if protect == "none" else "_hold" + protect
    for name, default, label in RUN_ID_KNOBS:
        value = getattr(options, name, default)
        if value != default:
            suffix += "_{}{}".format(label, str(value).replace(".", "p"))
    return "{}_{}{}_{}_seed{}_ur{}".format(
        options.dataset,
        options.partition,
        suffix,
        options.model,
        options.seed,
        options.unlearn_ratio,
    )


def build_command(python, stage, options, run_dir, metrics_path, log_path):
    command = [
        python,
        os.path.join(REPO_ROOT, "main.py"),
        "--exp", stage,
        "--dataset_name", options.dataset,
        "--dgl_data", "True",
        "--partition", options.partition,
        "--target_model", options.model,
        "--random_seed", str(options.seed),
        "--cuda", str(options.cuda),
        "--num_threads", str(options.num_threads),
        # mapping stage: Eq. (5), Eq. (9), Eq. (11)
        "--agg_feat", options.agg_feat,
        "--agg_label", "th",
        "--agg_edge", "rubost",
        "--test_edge_method", "0",
        "--th_sim2edge", "-1",
        "--use_edge_weight", "False",
        # training: Appendix B
        "--train_lr", str(options.train_lr),
        "--train_weight_decay", str(options.train_weight_decay),
        "--num_epochs", str(options.num_epochs),
        "--test_ratio", str(options.test_ratio),
        # unlearning request
        "--unlearn_task", "node",
        "--unlearn_ratio", str(options.unlearn_ratio),
        # quality gate
        "--partition_min_communities", str(options.partition_min_communities),
        "--partition_max_retries", str(options.partition_max_retries),
        "--metrics_file", metrics_path,
        "--log_file", log_path,
    ]
    if options.partition in ("ccp", "gae"):
        command += [
            "--ccp_resolution", str(options.ccp_resolution),
            "--ccp_fine_resolution", str(options.ccp_fine_resolution),
            "--ccp_theta", str(options.ccp_theta),
            "--ccp_max_communities_per_node", str(options.ccp_max_communities_per_node),
            "--ccp_overlap_threshold", str(options.ccp_overlap_threshold),
            "--ccp_propagation_steps", str(options.ccp_propagation_steps),
            "--ccp_protect_eval_nodes", str(options.ccp_protect_eval_nodes),
            "--ccp_max_community_size", str(options.ccp_max_community_size),
            "--ccp_tail_min_size", str(options.ccp_tail_min_size),
        ]
    if options.partition == "gae":
        command += [
            "--gae_epochs", str(options.gae_epochs),
            "--gae_hidden_dim", str(options.gae_hidden_dim),
            "--gae_latent_dim", str(options.gae_latent_dim),
            "--gae_device", str(options.gae_device),
        ]
    del run_dir
    _assert_run_id_covers_every_knob(options)
    return command


def _assert_run_id_covers_every_knob(options):
    """Fail loudly if a forwarded, result-changing option is not in the run id.

    Two ablation cells that differ only by an uncovered option would resolve to the
    same artifact directory and silently overwrite one another, so this is a hard
    error rather than a comment that can drift out of date.
    """
    covered = {name for name, _default, _label in RUN_ID_KNOBS} | RUN_ID_EXEMPT
    uncovered = sorted(set(vars(options)) - covered)
    if uncovered:
        raise SystemExit(
            "run_id_for() does not encode {}; add them to RUN_ID_KNOBS (or to "
            "RUN_ID_EXEMPT if they cannot change a result) before using them, "
            "otherwise two cells can overwrite each other".format(
                ", ".join(uncovered)
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--partition", default="ccp")
    parser.add_argument("--model", default="GCN", choices=["GCN", "GAT", "SAGE"])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--unlearn_ratio", type=float, default=0.005)
    parser.add_argument("--train_lr", type=float, default=0.01)
    parser.add_argument("--train_weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--agg_feat", default="pca", choices=["pca", "mean"],
                        help="Equation (5) mapped-feature rule; 'pca' is the paper default")
    parser.add_argument("--partition_min_communities", type=int, default=8)
    parser.add_argument("--partition_max_retries", type=int, default=0)
    parser.add_argument("--ccp_resolution", type=float, default=1.0)
    parser.add_argument("--ccp_fine_resolution", type=float, default=1.0)
    parser.add_argument("--ccp_theta", type=int, default=20)
    parser.add_argument("--ccp_max_communities_per_node", type=int, default=3)
    parser.add_argument("--ccp_overlap_threshold", type=float, default=0.35)
    parser.add_argument("--ccp_propagation_steps", type=int, default=2)
    parser.add_argument("--ccp_protect_eval_nodes", default="none",
                        choices=["none", "test", "test_val"])
    parser.add_argument("--ccp_max_community_size", type=int, default=0)
    parser.add_argument("--ccp_tail_min_size", type=int, default=0)
    parser.add_argument("--gae_epochs", type=int, default=100)
    parser.add_argument("--gae_hidden_dim", type=int, default=64)
    parser.add_argument("--gae_latent_dim", type=int, default=16)
    parser.add_argument("--gae_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument(
        "--keep-data-root",
        action="store_true",
        help="reuse an existing per-run artifact directory instead of failing",
    )
    parser.add_argument("--timeout", type=int, default=None, help="per-stage timeout (s)")
    options = parser.parse_args()

    run_id = run_id_for(options)
    data_root = os.path.join(REPO_ROOT, "reproduction", "runs", run_id)
    log_dir = os.path.join(REPO_ROOT, "reproduction", "logs", run_id)
    result_dir = os.path.join(REPO_ROOT, "reproduction", "results", "runs", run_id)

    if os.path.exists(data_root) and not options.keep_data_root:
        # Silently reusing a previous run's community file is exactly the failure
        # this driver exists to prevent, so refuse instead of guessing.
        raise SystemExit(
            "artifact directory {} already exists.  Remove it for a clean run, or "
            "pass --keep-data-root to continue an interrupted one.".format(data_root)
        )
    for directory in (data_root, log_dir, result_dir):
        os.makedirs(directory, exist_ok=True)
    os.makedirs(SHARED_DGL_DATA, exist_ok=True)

    environment = dict(os.environ)
    environment["CCGU_DATA_ROOT"] = data_root
    environment["CCGU_DGL_DATA"] = SHARED_DGL_DATA
    environment["PYTHONHASHSEED"] = str(options.seed)
    environment["OMP_NUM_THREADS"] = str(options.num_threads)

    summary = {
        "run_id": run_id,
        "dataset": options.dataset,
        "partition": options.partition,
        "model": options.model,
        "seed": options.seed,
        "unlearn_ratio": options.unlearn_ratio,
        "data_root": data_root,
        "stages": {},
    }
    # Merge into any existing summary instead of replacing it: running a subset of
    # --stages (e.g. only Partition, to re-check a cached partition) must not delete
    # the Train/Unlearn metrics that the tables are built from.
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as handle:
                previous = json.load(handle)
        except ValueError:
            previous = {}
        if previous.get("run_id") == run_id:
            summary["stages"] = previous.get("stages", {})
            print("[driver] merging into the existing summary ({} stage(s) kept)".format(
                len(summary["stages"])))

    for stage in options.stages:
        metrics_path = os.path.join(result_dir, stage.lower() + ".json")
        log_path = os.path.join(log_dir, stage.lower() + ".log")
        command = build_command(
            options.python, stage, options, data_root, metrics_path, log_path
        )
        print("\n[{}] {}".format(stage, " ".join(command)), flush=True)
        started = time.time()
        with open(log_path, "w") as handle:
            handle.write("# " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=options.timeout,
            )
        elapsed = time.time() - started
        stage_record = {
            "command": command,
            "returncode": completed.returncode,
            "wall_clock_seconds": elapsed,
            "log": os.path.relpath(log_path, REPO_ROOT),
        }
        if os.path.exists(metrics_path):
            with open(metrics_path) as handle:
                stage_record["metrics"] = json.load(handle)
            stage_record["metrics_path"] = os.path.relpath(metrics_path, REPO_ROOT)
        summary["stages"][stage] = stage_record

        print("[{}] rc={} in {:.1f}s -> {}".format(
            stage, completed.returncode, elapsed,
            os.path.relpath(log_path, REPO_ROOT)), flush=True)
        if completed.returncode != 0:
            summary["failed_stage"] = stage
            with open(summary_path, "w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            with open(log_path) as handle:
                tail = handle.readlines()[-25:]
            sys.stderr.write("".join(tail))
            raise SystemExit(
                "stage {} failed with rc={}; see {}".format(
                    stage, completed.returncode, log_path
                )
            )

    summary.pop("failed_stage", None)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("\nwrote {}".format(os.path.relpath(summary_path, REPO_ROOT)))

    train = summary["stages"].get("Train", {}).get("metrics", {})
    unlearn = summary["stages"].get("Unlearn", {}).get("metrics", {})
    if train:
        print("deployed  : macro-F1 {:.4f}  acc {:.4f}  ({} mapped nodes)".format(
            train.get("test_macro_f1", float("nan")),
            train.get("test_accuracy", float("nan")),
            train.get("mapped_graph_nodes")))
    if unlearn:
        print("unlearned : macro-F1 {:.4f}  acc {:.4f}  ({} nodes removed, {:.2f}s)".format(
            unlearn.get("test_macro_f1", float("nan")),
            unlearn.get("test_accuracy", float("nan")),
            unlearn.get("num_unlearned_nodes"),
            unlearn.get("unlearn_seconds", float("nan"))))


if __name__ == "__main__":
    main()
