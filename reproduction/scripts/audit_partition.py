#!/usr/bin/env python
"""Measure community-detection quality for one dataset/method/seed.

Reports the structural diagnostics of a single partition.  It bypasses
``main.py`` on purpose: it runs *only*
the partition stage so that partition cost and partition quality are measured
without any training noise, and it never writes into ``temp_data/`` so it cannot
disturb a pipeline run.

    .conda-env/bin/python reproduction/scripts/audit_partition.py \
        --dataset cora --method ccp --seed 4 \
        --json reproduction/results/partition/cora_ccp_seed4.json

``--method`` accepts ``ccp`` (the new default), ``oslom`` and ``slpa`` (the
legacy label-propagation implementations, kept verbatim) and ``infomap``.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from exp.methods.partition_diagnostics import (  # noqa: E402
    format_summary,
    summarise_partition,
)

METHODS = ("ccp", "gae", "oslom", "slpa", "infomap")


def load_graph(dataset_name):
    import dgl

    loaders = {
        "cora": dgl.data.CoraGraphDataset,
        "citeseer": dgl.data.CiteseerGraphDataset,
        "pubmed": dgl.data.PubmedGraphDataset,
        "cs": dgl.data.CoauthorCSDataset,
        "reddit": dgl.data.RedditDataset,
    }
    if dataset_name not in loaders:
        raise SystemExit("unsupported dataset %r" % dataset_name)
    return loaders[dataset_name](raw_dir=config.DGL_PATH)[0]


def run_partition(method, graph, args, max_iterations):
    if method == "ccp":
        from exp.methods.CCP import CommunityCentricPartition

        detector = CommunityCentricPartition(graph, args=args)
        result = detector.partition()
        return result + (detector,)
    if method == "gae":
        from exp.methods.GAEPartition import GAEPartition

        detector = GAEPartition(graph, args=args)
        result = detector.partition()
        return result + (detector,)
    if method == "oslom":
        from exp.methods.OSLOM import OSLOM

        oslom = OSLOM(graph, args=args)
        if max_iterations is not None:
            oslom.max_iterations = max_iterations
        return oslom.OSLOM_partition()
    if method == "slpa":
        from exp.methods.SLPA import SLPA

        iterations = 50 if max_iterations is None else max_iterations
        slpa = SLPA(
            graph,
            max_iterations=iterations,
            threshold=0.2,
            max_communities_per_node=5,
        )
        return slpa.SLPA_partition()
    if method == "infomap":
        from exp.methods.Infomap import Infomap

        return Infomap(graph, args=args).partition()
    raise SystemExit("unsupported method %r" % method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--method", default="ccp", choices=METHODS)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="override the legacy label-propagation iteration count (recorded "
        "in the report, because it changes the measured cost)",
    )
    parser.add_argument("--ccp-resolution", type=float, default=1.0)
    parser.add_argument("--ccp-fine-resolution", type=float, default=1.0)
    parser.add_argument("--ccp-theta", type=int, default=20)
    parser.add_argument("--ccp-max-communities-per-node", type=int, default=3)
    parser.add_argument("--ccp-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--ccp-propagation-steps", type=int, default=2)
    parser.add_argument("--ccp-max-community-size", type=int, default=0)
    parser.add_argument("--ccp-tail-min-size", type=int, default=0)
    parser.add_argument("--ccp-protect-eval-nodes", default="none",
                        choices=["none", "test", "test_val"])
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--gae-epochs", type=int, default=100)
    parser.add_argument("--gae-hidden-dim", type=int, default=64)
    parser.add_argument("--gae-latent-dim", type=int, default=16)
    parser.add_argument("--gae-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--with-edge-mapping", action="store_true",
                        help="also build and time the mapped edges (Equation 11), so "
                             "detector cost and mapped-edge cost are reported apart")
    parser.add_argument("--json", help="write the report to this path")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8,
                        help="torch thread count; the legacy label-propagation "
                             "methods are strongly affected by it, so the audit "
                             "table must hold it fixed across methods")
    options = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s: - %(name)s - : %(message)s",
        stream=sys.stdout,
    )

    random.seed(options.seed)
    np.random.seed(options.seed)
    import torch

    torch.manual_seed(options.seed)

    args = {
        "cuda": options.cuda,
        "num_threads": options.num_threads,
        "random_seed": options.seed,
        "dataset_name": options.dataset,
        "agg_feat": "pca",
        "agg_label": "th",
        "agg_edge": "rubost",
        "partition": options.method,
        "th_sim2edge": -1,
        "test_edge_method": 0,
        "use_edge_weight": False,
        "ccp_resolution": options.ccp_resolution,
        "ccp_fine_resolution": options.ccp_fine_resolution,
        "ccp_theta": options.ccp_theta,
        "ccp_max_communities_per_node": options.ccp_max_communities_per_node,
        "ccp_overlap_threshold": options.ccp_overlap_threshold,
        "ccp_propagation_steps": options.ccp_propagation_steps,
        "ccp_conductance_threshold": None,
        "ccp_max_community_size": options.ccp_max_community_size,
        "ccp_tail_min_size": options.ccp_tail_min_size,
        "ccp_protect_eval_nodes": options.ccp_protect_eval_nodes,
        "test_ratio": options.test_ratio,
        "gae_epochs": options.gae_epochs,
        "gae_hidden_dim": options.gae_hidden_dim,
        "gae_latent_dim": options.gae_latent_dim,
        "gae_device": options.gae_device,
    }

    graph = load_graph(options.dataset)
    num_nodes = graph.number_of_nodes()
    labels = graph.ndata["label"].cpu().numpy() if "label" in graph.ndata else None
    src, dst = graph.edges()
    edges = (src.cpu().numpy(), dst.cpu().numpy())

    started = time.time()
    result = run_partition(options.method, graph, args, options.max_iterations)
    detector = result[4] if len(result) > 4 else None
    _n2c, c2n, num_communities, elapsed = result[:4]
    wall_clock = time.time() - started

    edge_mapping = None
    if options.with_edge_mapping:
        from exp.unlearning_core import (
            build_node_to_communities,
            calculate_edge_counts,
            calculate_robustness_similarity,
        )

        edge_start = time.time()
        n2c_full = build_node_to_communities(c2n)
        counts = calculate_edge_counts(edges[0], edges[1], n2c_full)
        similarity = calculate_robustness_similarity(
            c2n, counts, test_edge_method=0, include_jaccard=True,
            pair_reduction="source",
        )
        edge_mapping = {
            "edge_mapping_seconds": time.time() - edge_start,
            "community_pairs_scored": len(similarity),
            "directed_edge_count_keys": len(counts),
        }

    summary = summarise_partition(
        c2n,
        num_nodes=num_nodes,
        labels=labels,
        edges=edges,
        elapsed_seconds=elapsed,
    )
    summary.update(
        {
            "dataset": options.dataset,
            "method": options.method,
            "seed": options.seed,
            "wall_clock_seconds": wall_clock,
            "reported_num_communities": int(num_communities),
            "legacy_max_iterations": options.max_iterations,
            "num_threads": options.num_threads,
        }
    )
    if options.method in ("ccp", "gae"):
        summary["ccp_config"] = {
            key: args[key]
            for key in sorted(args)
            if key.startswith("ccp_") or key.startswith("gae_")
        }
    if detector is not None:
        summary["scale_report"] = getattr(detector, "scale_report", None)
        summary["community_hash"] = hashlib.sha256(
            json.dumps(
                {int(k): sorted(int(n) for n in v) for k, v in c2n.items()},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        if hasattr(detector, "gae_report"):
            summary["gae_report"] = dict(detector.gae_report)
    if edge_mapping is not None:
        summary.update(edge_mapping)

    print()
    print("dataset {} | method {} | seed {}".format(
        options.dataset, options.method, options.seed))
    print(format_summary(summary))
    if summary.get("community_hash"):
        print("  community hash         : {}".format(summary["community_hash"]))
    if edge_mapping is not None:
        print("  mapped-edge build      : {:.2f} s for {} scored pair(s)".format(
            edge_mapping["edge_mapping_seconds"],
            edge_mapping["community_pairs_scored"]))
    if summary.get("gae_report"):
        report = summary["gae_report"]
        print("  GAE encoder            : {:.2f} s on {} (loss {:.4f} -> {:.4f})".format(
            report.get("encoder_seconds", float("nan")), report.get("device"),
            report.get("first_loss", float("nan")), report.get("final_loss", float("nan"))))
        print("  GAE k-means            : {:.2f} s for {} cluster(s)".format(
            report.get("kmeans_seconds", float("nan")),
            report.get("num_clusters_returned")))

    if options.json:
        os.makedirs(os.path.dirname(os.path.abspath(options.json)), exist_ok=True)
        with open(options.json, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print("wrote {}".format(options.json))


if __name__ == "__main__":
    main()
