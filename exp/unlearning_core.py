"""Backend-independent helpers for community-centric node unlearning.

The public pipeline remains DGL-based.  These helpers contain only the
community bookkeeping and edge-score calculations so they can be smoke-tested
on a tiny graph without downloading a dataset or training a GNN.
"""

from collections import defaultdict
from math import log1p, sqrt


def build_node_to_communities(c2n):
    """Build an overlap-aware node-to-community mapping."""
    n2c = defaultdict(list)
    for community, nodes in c2n.items():
        for node in nodes:
            n2c[int(node)].append(int(community))
    return dict(n2c)


def remove_nodes_and_reindex(c2n, unlearning_nodes):
    """Delete nodes from every membership and compact non-empty communities.

    Returns the updated ``c2n``, an old-to-new community id mapping, and the
    affected community ids before and after compaction.
    """
    deleted = {int(node) for node in unlearning_nodes}
    affected_old = {
        int(community)
        for community, nodes in c2n.items()
        if deleted.intersection(int(node) for node in nodes)
    }

    updated_c2n = {}
    old_to_new = {}
    for old_community in sorted(c2n):
        remaining = [
            int(node) for node in c2n[old_community] if int(node) not in deleted
        ]
        if not remaining:
            continue
        new_community = len(updated_c2n)
        old_to_new[int(old_community)] = new_community
        updated_c2n[new_community] = remaining

    affected_new = {
        old_to_new[community]
        for community in affected_old
        if community in old_to_new
    }
    return updated_c2n, old_to_new, affected_old, affected_new


def remap_rows(values, old_to_new):
    """Remap a community-indexed sequence after empty communities are removed."""
    remapped = [None] * len(old_to_new)
    for old_community, new_community in old_to_new.items():
        remapped[new_community] = values[old_community]
    return remapped


def calculate_edge_counts(src, dst, n2c):
    """Count directed original-graph edges between overlapping communities."""
    edge_counts = defaultdict(int)
    for source, target in zip(src, dst):
        source_communities = n2c.get(int(source), ())
        target_communities = n2c.get(int(target), ())
        for source_community in source_communities:
            for target_community in target_communities:
                if source_community != target_community:
                    edge_counts[(source_community, target_community)] += 1
    return dict(edge_counts)


def calculate_robustness_similarity(
    c2n,
    edge_counts,
    test_edge_method=2,
    include_jaccard=True,
):
    """Recalculate Equation (11)-style mapped-edge scores.

    Both directions are returned because the DGL mapped graph is directed.
    """
    if test_edge_method not in (0, 1, 2, 3):
        raise ValueError("test_edge_method must be one of 0, 1, 2, or 3")

    num_communities = len(c2n)
    community_sets = {
        community: set(nodes) for community, nodes in c2n.items()
    }
    similarity = {}

    for source in range(num_communities):
        for target in range(source + 1, num_communities):
            edge_count = edge_counts.get((source, target), 0)
            reverse_edge_count = edge_counts.get((target, source), 0)
            pair_edge_count = max(edge_count, reverse_edge_count)
            if pair_edge_count <= 0:
                continue

            out_degree = sum(
                edge_counts.get((source, other), 0)
                for other in range(num_communities)
                if other != source
            )
            in_degree = sum(
                edge_counts.get((other, target), 0)
                for other in range(num_communities)
                if other != target
            )

            robustness = 0.0
            if out_degree > 0 and in_degree > 0:
                if test_edge_method == 0:
                    robustness = (
                        pair_edge_count / sqrt(out_degree)
                    ) * (
                        pair_edge_count / sqrt(in_degree)
                    )
                elif test_edge_method == 1:
                    robustness = log1p(pair_edge_count) / (
                        log1p(out_degree) + log1p(in_degree)
                    )
                elif test_edge_method == 2:
                    robustness = (
                        log1p(pair_edge_count) / sqrt(log1p(out_degree))
                    ) * (
                        log1p(pair_edge_count) / sqrt(log1p(in_degree))
                    )

            union_size = len(
                community_sets[source].union(community_sets[target])
            )
            jaccard = pair_edge_count / union_size if include_jaccard and union_size else 0.0
            score = robustness + jaccard
            if score:
                similarity[(source, target)] = score
                similarity[(target, source)] = score

    return similarity
