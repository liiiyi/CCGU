"""Lightweight quality diagnostics for a (possibly overlapping) partition.

Community detection is stochastic, and CGE's whole pipeline -- the mapped
features (Eq. 5), the mapped labels (Eq. 9), the mapped edges (Eq. 11) and the
train/val/test split over mapped nodes -- is built on top of whatever partition
comes out.  A partition that happens to be badly aligned with the data therefore
shows up as a large swing in downstream Macro F1, and no amount of downstream
tuning recovers it.

These diagnostics are cheap (pure numpy, one pass over the edge list) and are
logged for *every* partition, so a reader can tell a good draw from a bad one
without re-running anything, and so a bad draw is visible instead of silent.

Nothing here looks at model predictions: the health check may only ever be a
function of the partition itself, otherwise "retry until healthy" degenerates
into cherry-picking on the test set.
"""

from collections import Counter

import numpy as np

#: Membership-count buckets reported in the overlap histogram.
_OVERLAP_BUCKETS = (1, 2, 3, 4, 5)


def _as_int_array(values):
    return np.asarray(list(values), dtype=np.int64)


def _primary_assignment(c2n, num_nodes):
    """Lowest-id community per node, or -1 for nodes in no community."""
    primary = np.full(num_nodes, -1, dtype=np.int64)
    for community in sorted(c2n, reverse=True):
        nodes = _as_int_array(c2n[community])
        if nodes.size:
            primary[nodes] = int(community)
    return primary


def disjoint_modularity(primary, src, dst, num_nodes):
    """Newman modularity of the primary (disjoint) projection of a partition.

    The graph is treated as undirected and simple: duplicate and reversed edges
    collapse, self-loops are dropped.  Nodes with no community are excluded.
    """
    del num_nodes  # inferred from `primary`
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return float("nan")

    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    unique_edges = np.unique(np.stack([low, high], axis=1), axis=0)
    low, high = unique_edges[:, 0], unique_edges[:, 1]

    community_low = primary[low]
    community_high = primary[high]
    valid = (community_low >= 0) & (community_high >= 0)
    low, high = low[valid], high[valid]
    community_low, community_high = community_low[valid], community_high[valid]
    num_edges = low.size
    if num_edges == 0:
        return float("nan")

    degrees = np.bincount(np.concatenate([low, high]), minlength=primary.size)
    internal = np.bincount(
        community_low[community_low == community_high],
        minlength=int(primary.max()) + 1,
    )
    volume = np.bincount(
        primary[primary >= 0],
        weights=degrees[primary >= 0],
        minlength=int(primary.max()) + 1,
    )
    return float(
        np.sum(internal / num_edges - (volume / (2.0 * num_edges)) ** 2)
    )


def summarise_partition(
    c2n,
    num_nodes,
    labels=None,
    edges=None,
    elapsed_seconds=None,
):
    """Return a JSON-serialisable summary of partition quality.

    Args:
        c2n: mapping community id -> list of original node ids (may overlap).
        num_nodes: number of nodes in the original graph.
        labels: optional original node labels, used for label purity.
        edges: optional ``(src, dst)`` arrays, used for modularity.
        elapsed_seconds: optional wall-clock cost of the partition.
    """
    num_communities = len(c2n)
    membership = Counter()
    sizes = []
    for nodes in c2n.values():
        unique_nodes = set(int(node) for node in nodes)
        sizes.append(len(unique_nodes))
        membership.update(unique_nodes)

    sizes = np.asarray(sizes if sizes else [0], dtype=np.int64)
    counts = np.asarray(
        list(membership.values()) if membership else [0], dtype=np.int64
    )
    covered = len(membership)

    histogram = {
        str(bucket): int(np.count_nonzero(counts == bucket))
        for bucket in _OVERLAP_BUCKETS
    }
    histogram["6+"] = int(np.count_nonzero(counts > _OVERLAP_BUCKETS[-1]))
    histogram["0"] = int(num_nodes - covered)

    summary = {
        "num_nodes": int(num_nodes),
        "num_communities": int(num_communities),
        "compression_ratio": (
            float(num_communities) / num_nodes if num_nodes else float("nan")
        ),
        "coverage": float(covered) / num_nodes if num_nodes else float("nan"),
        "uncovered_nodes": int(num_nodes - covered),
        "memberships_per_node_mean": float(counts.mean()) if covered else 0.0,
        "memberships_per_node_max": int(counts.max()) if covered else 0,
        "overlapping_node_fraction": (
            float(np.count_nonzero(counts >= 2)) / num_nodes if num_nodes else 0.0
        ),
        "overlap_histogram": histogram,
        "community_size_min": int(sizes.min()),
        "community_size_median": float(np.median(sizes)),
        "community_size_mean": float(sizes.mean()),
        "community_size_max": int(sizes.max()),
        "singleton_communities": int(np.count_nonzero(sizes == 1)),
        "tiny_communities_le2": int(np.count_nonzero(sizes <= 2)),
    }
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = float(elapsed_seconds)

    if labels is not None and num_communities:
        labels = np.asarray(labels).reshape(-1)
        purities = []
        for nodes in c2n.values():
            nodes = _as_int_array(nodes)
            if nodes.size == 0:
                continue
            _, class_counts = np.unique(labels[nodes], return_counts=True)
            purities.append(class_counts.max() / float(nodes.size))
        summary["label_purity_mean"] = float(np.mean(purities)) if purities else float("nan")

    if edges is not None and num_communities:
        primary = _primary_assignment(c2n, num_nodes)
        summary["modularity_primary"] = disjoint_modularity(
            primary, edges[0], edges[1], num_nodes
        )

    return summary


def format_summary(summary):
    """One multi-line human-readable block, for the run log."""
    lines = [
        "partition diagnostics:",
        "  communities            : {}  (compression {:.4f} of {} nodes)".format(
            summary["num_communities"],
            summary["compression_ratio"],
            summary["num_nodes"],
        ),
        "  coverage               : {:.4f}  ({} uncovered nodes)".format(
            summary["coverage"], summary["uncovered_nodes"]
        ),
        "  memberships per node   : mean {:.3f}, max {}".format(
            summary["memberships_per_node_mean"],
            summary["memberships_per_node_max"],
        ),
        "  overlapping nodes      : {:.4f} of all nodes".format(
            summary["overlapping_node_fraction"]
        ),
        "  overlap histogram      : {}".format(summary["overlap_histogram"]),
        "  community sizes        : min {}, median {:.1f}, mean {:.2f}, max {}".format(
            summary["community_size_min"],
            summary["community_size_median"],
            summary["community_size_mean"],
            summary["community_size_max"],
        ),
        "  singleton communities  : {} (<=2 nodes: {})".format(
            summary["singleton_communities"], summary["tiny_communities_le2"]
        ),
    ]
    if "label_purity_mean" in summary:
        lines.append(
            "  mean label purity      : {:.4f}".format(summary["label_purity_mean"])
        )
    if "modularity_primary" in summary:
        lines.append(
            "  modularity (primary)   : {:.4f}".format(summary["modularity_primary"])
        )
    if "elapsed_seconds" in summary:
        lines.append(
            "  partition wall clock   : {:.2f} s".format(summary["elapsed_seconds"])
        )
    return "\n".join(lines)


def check_partition_health(summary, min_communities=8, min_coverage=1.0):
    """Return a list of human-readable reasons this partition is unusable.

    Deliberately conservative: it only fires on partitions that make the rest of
    the pipeline meaningless (too few mapped nodes to split into train/val/test,
    or original nodes that no mapped node represents).  It never inspects model
    quality -- see the module docstring.
    """
    problems = []
    if summary["num_communities"] < min_communities:
        problems.append(
            "only {} communities (< {}); the mapped graph cannot be split into "
            "train/val/test".format(summary["num_communities"], min_communities)
        )
    if summary["coverage"] < min_coverage:
        problems.append(
            "coverage {:.4f} < {:.4f}; {} original nodes are represented by no "
            "mapped node".format(
                summary["coverage"], min_coverage, summary["uncovered_nodes"]
            )
        )
    return problems
