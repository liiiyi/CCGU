#!/usr/bin/env python
"""Aggregate every recorded run into the markdown tables used in the write-up.

    .conda-env/bin/python reproduction/scripts/make_tables.py \
        --out reproduction/results/tables.md

Reads ``reproduction/results/runs/*/summary.json`` -- i.e. only numbers that were
actually produced by a completed pipeline -- and reports mean +/- population std
across seeds.  Runs are grouped by (dataset, partition, holdout mode, backbone);
the seeds that went into each cell are printed so a reader can see that nothing
was dropped.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mean_std(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return float("nan"), float("nan"), 0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance), len(values)


def fmt(values, digits=4):
    mean, std, count = mean_std(values)
    if count == 0:
        return "n/a"
    if count == 1:
        return "{:.{d}f}".format(mean, d=digits)
    return "{:.{d}f} ± {:.{d}f}".format(mean, std, d=digits)


def fmt_int(values):
    mean, std, count = mean_std([float(v) for v in values if v is not None])
    if count == 0:
        return "n/a"
    if count == 1:
        return "{:.0f}".format(mean)
    return "{:.0f} ± {:.0f}".format(mean, std)


#: Run-id fragments that mark a post-paper ablation.  Each must produce its own
#: table row: averaging an ablation into the baseline would report a configuration
#: nobody ran.  `run_experiment.py` guarantees the run id encodes every forwarded
#: knob whose value differs from the driver default.
VARIANT_LABELS = (
    ("_tail", "tail"),
    ("_theta", "theta"),
    ("_cap", "cap"),
    ("_fres", "fine-res"),
    ("_res", "res"),
    ("_tau", "tau"),
    ("_prop", "prop"),
    ("_hid", "gae-hid"),
    ("_lat", "gae-lat"),
    ("_ep", "gae-ep"),
    ("_tr", "test-ratio"),
    ("_wd", "wd"),
    ("_lr", "lr"),
    ("_ne", "epochs"),
    ("_minc", "min-comm"),
    ("_retry", "retry"),
)


def variant_of(run_id, partition):
    """Short label naming the configuration; ``baseline`` when nothing is off-default."""
    parts = []
    remaining = run_id
    for fragment, label in VARIANT_LABELS:
        index = remaining.find(fragment)
        if index == -1:
            continue
        value = remaining[index + len(fragment):].split("_")[0]
        if value and value[0].isdigit():
            parts.append("{}={}".format(label, value.replace("p", ".")))
            remaining = remaining[:index] + remaining[index + len(fragment) + len(value):]
    if partition == "gae":
        parts.append("detector=gae")
    return ", ".join(parts) if parts else "baseline"


def load_runs(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            summary = json.load(handle)
        stages = summary.get("stages", {})
        if summary.get("failed_stage"):
            continue
        train = stages.get("Train", {}).get("metrics") or {}
        unlearn = stages.get("Unlearn", {}).get("metrics") or {}
        partition = stages.get("Partition", {}).get("metrics") or {}
        if not train:
            continue
        diagnostics = partition.get("partition_diagnostics") or {}
        run_id = summary["run_id"]
        holdout = "none"
        for candidate in ("holdtest_val", "holdtest"):
            if "_" + candidate in run_id:
                holdout = candidate.replace("hold", "")
                break
        agg_feat = "mean" if "_featmean" in run_id else "pca"
        device = "cpu"
        for candidate in ("devcuda", "devcpu"):
            if candidate in run_id:
                device = candidate.replace("dev", "")
                break
        variant = variant_of(run_id, summary["partition"])
        runs.append({
            "run_id": run_id,
            "dataset": summary["dataset"],
            "partition": summary["partition"],
            "holdout": holdout,
            "agg_feat": agg_feat,
            "device": device,
            "variant": variant,
            "model": summary["model"],
            "seed": summary["seed"],
            "train": train,
            "unlearn": unlearn,
            "partition_metrics": partition,
            "diagnostics": diagnostics,
        })
    return runs


def utility_table(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["partition"], run["device"], run["holdout"], run["agg_feat"], run["variant"], run["model"])].append(run)

    lines = [
        "| Dataset | Partition | Device | Held-out | Eq.5 | Variant | Backbone | Seeds | Mapped nodes | Test nodes "
        "| Deployed Macro-F1 | Deployed Acc | Unlearned Macro-F1 | Unlearned Acc |",
        "|---" * 14 + "|",
    ]
    for key in sorted(groups):
        dataset, partition, device, holdout, agg_feat, variant, model = key
        cells = sorted(groups[key], key=lambda run: run["seed"])
        seeds = ",".join(str(run["seed"]) for run in cells)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                dataset, partition, device, holdout, agg_feat, variant, model, seeds,
                fmt_int([run["train"].get("mapped_graph_nodes") for run in cells]),
                fmt_int([run["train"].get("test_nodes") for run in cells]),
                fmt([run["train"].get("test_macro_f1") for run in cells]),
                fmt([run["train"].get("test_accuracy") for run in cells]),
                fmt([run["unlearn"].get("test_macro_f1") for run in cells]),
                fmt([run["unlearn"].get("test_accuracy") for run in cells]),
            )
        )
    return "\n".join(lines)


def efficiency_table(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["partition"], run["device"], run["holdout"],
                run["agg_feat"], run["variant"])].append(run)

    lines = [
        "| Dataset | Partition | Device | Held-out | Eq.5 | Variant | Cells | Detector (s) | Deploy train (s) "
        "| Community update (s) | Unlearn total (s) | Unlearned nodes |",
        "|---" * 12 + "|",
    ]
    for key in sorted(groups):
        dataset, partition, device, holdout, agg_feat, variant = key
        cells = groups[key]

        def numeric(records, field, stage):
            out = []
            for run in records:
                value = run[stage].get(field)
                if isinstance(value, (int, float)):
                    out.append(float(value))
            return out

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                dataset, partition, device, holdout, agg_feat, variant, len(cells),
                fmt(numeric(cells, "partition_seconds", "partition_metrics"), 2),
                fmt(numeric(cells, "train_seconds", "train"), 2),
                fmt(numeric(cells, "community_update_seconds", "unlearn"), 3),
                fmt(numeric(cells, "unlearn_seconds", "unlearn"), 2),
                fmt_int([run["unlearn"].get("num_unlearned_nodes") for run in cells]),
            )
        )
    return "\n".join(lines)


def partition_table(runs):
    groups = defaultdict(list)
    for run in runs:
        if run["diagnostics"]:
            groups[(run["dataset"], run["partition"], run["device"], run["holdout"], run["agg_feat"], run["variant"], run["seed"])].append(run)

    lines = [
        "| Dataset | Partition | Device | Held-out | Eq.5 | Variant | Seed | Communities | Compression | Coverage "
        "| Overlapping nodes | Memberships/node | Singletons | Size median | Size max "
        "| Label purity | Modularity (primary) |",
        "|---" * 17 + "|",
    ]
    for key in sorted(groups):
        dataset, partition, device, holdout, agg_feat, variant, seed = key
        diagnostics = groups[key][0]["diagnostics"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {} | {:.1f} | {} "
            "| {:.4f} | {:.4f} |".format(
                dataset, partition, device, holdout, agg_feat, variant, seed,
                diagnostics["num_communities"],
                diagnostics["compression_ratio"],
                diagnostics["coverage"],
                diagnostics["overlapping_node_fraction"],
                diagnostics["memberships_per_node_mean"],
                diagnostics["singleton_communities"],
                diagnostics["community_size_median"],
                diagnostics["community_size_max"],
                diagnostics.get("label_purity_mean", float("nan")),
                diagnostics.get("modularity_primary", float("nan")),
            )
        )
    return "\n".join(lines)


def seed_spread_table(runs):
    """Per-seed downstream numbers, so partition-driven variance stays visible."""
    groups = defaultdict(dict)
    for run in runs:
        groups[(run["dataset"], run["partition"], run["device"], run["holdout"],
                run["agg_feat"], run["variant"], run["model"])][run["seed"]] = run
    seeds = sorted({run["seed"] for run in runs})

    header = (
        "| Dataset | Partition | Device | Held-out | Eq.5 | Variant | Backbone | "
        + " | ".join("seed {} F1".format(seed) for seed in seeds)
        + " | spread |"
    )
    lines = [header, "|---" * (7 + len(seeds) + 1) + "|"]
    for key in sorted(groups):
        dataset, partition, device, holdout, agg_feat, variant, model = key
        row = [dataset, partition, device, holdout, agg_feat, variant, model]
        values = []
        for seed in seeds:
            run = groups[key].get(seed)
            if run is None:
                row.append("-")
                continue
            value = run["train"].get("test_macro_f1")
            row.append("{:.4f}".format(value) if value is not None else "-")
            if value is not None and not math.isnan(value):
                values.append(value)
        row.append("{:.4f}".format(max(values) - min(values)) if len(values) > 1 else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs",
        default=os.path.join(REPO_ROOT, "reproduction", "results", "runs", "*", "summary.json"),
    )
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "reproduction", "results", "tables.md"))
    options = parser.parse_args()

    runs = load_runs(options.runs)
    if not runs:
        raise SystemExit("no completed runs found under {}".format(options.runs))

    sections = [
        "<!-- generated by reproduction/scripts/make_tables.py -- do not edit by hand -->",
        "# CCGU / CGE reproduction tables",
        "",
        "{} completed run(s).  Values are mean ± population std over the listed seeds."
        .format(len(runs)),
        "",
        "## Model utility (mapped-graph node classification)",
        "",
        utility_table(runs),
        "",
        "## Per-seed spread of deployed Macro-F1",
        "",
        seed_spread_table(runs),
        "",
        "## Efficiency",
        "",
        efficiency_table(runs),
        "",
        "## Partition quality",
        "",
        partition_table(runs),
        "",
    ]
    text = "\n".join(sections)
    with open(options.out, "w") as handle:
        handle.write(text)
    print(text)
    print("\nwrote {}".format(os.path.relpath(options.out, REPO_ROOT)))


if __name__ == "__main__":
    main()
