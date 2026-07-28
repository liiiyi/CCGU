"""Backend-independent helpers for community-centric node unlearning and mapping.

The public pipeline remains DGL-based.  These helpers contain only the
community bookkeeping and edge-score calculations so they can be smoke-tested
on a tiny graph without downloading a dataset or training a GNN.

Both ``exp/exp_partition.py`` and ``exp/exp_unlearn.py`` build the mapped edges
from here, so the deployed mapped graph and the post-deletion mapped graph are
guaranteed to come out of the same code.

Everything is **sparse**: the cost scales with the number of community pairs that
actually carry an original edge, not with the square of the number of
communities.  The earlier dense formulation pre-allocated every ordered community
pair and then rescanned all of them, which on Coauthor-CS with evaluation nodes
held out (~5,600 communities) means ~31 million dictionary entries and a pair
scan that ran for minutes; the degree normalisers in Equation (11) were also
recomputed with an inner loop over all communities for every connected pair.
"""

from collections import defaultdict
from math import log1p, sqrt

#: How a community pair's edge count is read out of the directed counts.
#:
#: ``"source"``  -- ``edge_counts[(low, high)]`` only.  This is what
#:                  ``exp/exp_partition.py`` has always used, so it defines the
#:                  deployed mapped graph.
#: ``"max"``     -- ``max(forward, reverse)``, the historical behaviour of this
#:                  module.
#:
#: For every dataset in this project the original DGL graphs are bidirectional, so
#: ``edge_counts`` is symmetric and the two rules coincide; see
#: ``tests/test_mapping_rules.py``.
PAIR_REDUCTIONS = ("source", "max")


def build_node_to_communities(c2n):
    """Build an overlap-aware node-to-community mapping."""
    n2c = defaultdict(list)
    for community, nodes in c2n.items():
        for node in nodes:
            n2c[int(node)].append(int(community))
    return dict(n2c)


def memberships_of(n2c, node):
    """Communities of ``node``, tolerating scalar-valued ``n2c``.

    ``exp/methods/{Infomap,CCD}.py`` return ``n2c[node] = community`` rather than a
    list, and ``--partition lpa`` returns lists; both must work here.
    """
    memberships = n2c.get(int(node))
    if memberships is None:
        return ()
    if isinstance(memberships, (list, tuple, set, frozenset)):
        return memberships
    return (memberships,)


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
    """Count directed original-graph edges between overlapping communities.

    Sparse: only pairs that carry at least one edge appear in the result.
    """
    edge_counts = defaultdict(int)
    for source, target in zip(src, dst):
        source_communities = memberships_of(n2c, source)
        target_communities = memberships_of(n2c, target)
        for source_community in source_communities:
            for target_community in target_communities:
                if source_community != target_community:
                    edge_counts[(int(source_community), int(target_community))] += 1
    return dict(edge_counts)


def community_degrees(edge_counts):
    """Total out- and in-edges per community, over one pass of ``edge_counts``.

    ``out_degree[c]`` is ``sum_k edge_counts[(c, k)]`` for ``k != c`` and
    ``in_degree[c]`` is ``sum_k edge_counts[(k, c)]`` for ``k != c``, which is
    exactly what Equation (11) normalises by.  ``edge_counts`` never holds a
    ``(c, c)`` key, but the guard is kept explicit.
    """
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    for (source, target), count in edge_counts.items():
        if source == target:
            continue
        out_degree[source] += count
        in_degree[target] += count
    return dict(out_degree), dict(in_degree)


def observed_community_pairs(edge_counts):
    """Unordered ``(low, high)`` pairs that carry at least one directed edge."""
    pairs = set()
    for source, target in edge_counts:
        if source == target:
            continue
        pairs.add((source, target) if source < target else (target, source))
    return pairs


def robustness_term(edge_count, out_degree, in_degree, test_edge_method):
    """The non-Jaccard part of the mapped-edge score.

    ``test_edge_method``:
        0 -- Equation (11) exactly: ``(s/sqrt(D_out)) * (s/sqrt(D_in))``
        1 -- ``log1p(s) / (log1p(D_out) + log1p(D_in))``
        2 -- the log-normalised analogue of 0
        3 -- no robustness term at all
    """
    if out_degree <= 0 or in_degree <= 0:
        return 0.0
    if test_edge_method == 0:
        return (edge_count / sqrt(out_degree)) * (edge_count / sqrt(in_degree))
    if test_edge_method == 1:
        return log1p(edge_count) / (log1p(out_degree) + log1p(in_degree))
    if test_edge_method == 2:
        return (log1p(edge_count) / sqrt(log1p(out_degree))) * (
            log1p(edge_count) / sqrt(log1p(in_degree))
        )
    return 0.0


def calculate_robustness_similarity(
    c2n,
    edge_counts,
    test_edge_method=2,
    include_jaccard=True,
    pair_reduction="max",
):
    """Equation (11)-style mapped-edge scores, over observed pairs only.

    Both directions are returned because the DGL mapped graph is directed.
    """
    if test_edge_method not in (0, 1, 2, 3):
        raise ValueError("test_edge_method must be one of 0, 1, 2, or 3")
    if pair_reduction not in PAIR_REDUCTIONS:
        raise ValueError(
            "pair_reduction must be one of {}".format(", ".join(PAIR_REDUCTIONS))
        )

    community_sets = {
        int(community): set(int(node) for node in nodes)
        for community, nodes in c2n.items()
    }
    out_degree, in_degree = community_degrees(edge_counts)
    similarity = {}

    for source, target in sorted(observed_community_pairs(edge_counts)):
        if source not in community_sets or target not in community_sets:
            continue
        forward = edge_counts.get((source, target), 0)
        if pair_reduction == "max":
            pair_edge_count = max(forward, edge_counts.get((target, source), 0))
        else:
            pair_edge_count = forward
        if pair_edge_count <= 0:
            continue

        robustness = robustness_term(
            pair_edge_count,
            out_degree.get(source, 0),
            in_degree.get(target, 0),
            test_edge_method,
        )

        union_size = len(community_sets[source].union(community_sets[target]))
        jaccard = (
            pair_edge_count / union_size if include_jaccard and union_size else 0.0
        )
        score = robustness + jaccard
        if score:
            similarity[(source, target)] = score
            similarity[(target, source)] = score

    return similarity
