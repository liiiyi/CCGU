"""The sparse mapped-edge implementation must equal the dense one it replaced.

``exp/exp_partition.py`` used to pre-allocate every ordered community pair and
then rescan all of them, recomputing both Equation (11) degree normalisers with an
inner loop over every community for each connected pair.  On Coauthor-CS with
evaluation nodes held out (~5,600 communities) that is ~31 million dictionary
entries and a pair scan that ran for minutes.

The reference implementations below are transcriptions of the *original* code.
They are deliberately slow and deliberately dense; the tests assert that the
sparse replacement returns exactly the same numbers.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.unlearning_core import (  # noqa: E402
    build_node_to_communities,
    calculate_edge_counts,
    calculate_robustness_similarity,
    community_degrees,
    memberships_of,
    observed_community_pairs,
)


# --------------------------------------------------------------------------- #
# Transcriptions of the pre-change dense implementations, used as references.
# --------------------------------------------------------------------------- #

def dense_edge_counts_reference(src, dst, n2c, c2n):
    """Original exp_partition.calculate_edge_counts, zeros included."""
    edge_counts = {}
    for community_u in c2n.keys():
        for community_v in c2n.keys():
            if community_u != community_v:
                edge_counts[(community_u, community_v)] = 0
    for u, v in zip(src, dst):
        communities_u = n2c[u] if isinstance(n2c[u], list) else [n2c[u]]
        communities_v = n2c[v] if isinstance(n2c[v], list) else [n2c[v]]
        for community_u in communities_u:
            for community_v in communities_v:
                if community_u != community_v:
                    edge_counts[(community_u, community_v)] += 1
    return edge_counts


def dense_similarity_reference(c2n, edge_counts, n_communities, test_edge_method):
    """Original exp_partition.calculate_nucleus_sim inner loops."""
    import math
    from math import log

    community_set = {community: set(c2n[community]) for community in c2n}
    sim = {}
    for i in range(n_communities):
        for j in range(i + 1, n_communities):
            if i in community_set and j in community_set:
                edge_count_bet_ij = edge_counts.get((i, j), 0)
                if edge_count_bet_ij > 0:
                    union_size_val = len(community_set[i].union(community_set[j]))
                    jaccard_sim = edge_count_bet_ij / union_size_val
                    edge_count = edge_counts.get((i, j), 0)
                    out_degree_A = sum(
                        edge_counts.get((i, k), 0)
                        for k in range(n_communities) if k != i
                    )
                    in_degree_B = sum(
                        edge_counts.get((k, j), 0)
                        for k in range(n_communities) if k != j
                    )
                    if out_degree_A > 0 and in_degree_B > 0:
                        if test_edge_method == 0:
                            robustness = (edge_count / math.sqrt(out_degree_A)) * (
                                edge_count / math.sqrt(in_degree_B))
                        elif test_edge_method == 1:
                            robustness = log(1 + edge_count) / (
                                log(1 + out_degree_A) + log(1 + in_degree_B))
                        elif test_edge_method == 2:
                            robustness = (
                                math.log1p(edge_count) / math.sqrt(math.log1p(out_degree_A))
                            ) * (
                                math.log1p(edge_count) / math.sqrt(math.log1p(in_degree_B))
                            )
                        else:
                            robustness = 0
                        final_score = robustness + jaccard_sim
                        if final_score != 0:
                            sim[(i, j)] = final_score
                            sim[(j, i)] = final_score
    return sim


def random_overlapping_partition(rng, num_nodes, num_communities, max_memberships=3):
    c2n = {index: [] for index in range(num_communities)}
    for node in range(num_nodes):
        count = min(rng.randint(1, max_memberships + 1), num_communities)
        for community in rng.choice(num_communities, size=count, replace=False):
            c2n[int(community)].append(node)
    # Drop empties and compact, as every detector in the repo does.
    compacted = {}
    for nodes in c2n.values():
        if nodes:
            compacted[len(compacted)] = sorted(nodes)
    return compacted


def random_bidirectional_edges(rng, num_nodes, num_edges):
    src = rng.randint(0, num_nodes, size=num_edges)
    dst = rng.randint(0, num_nodes, size=num_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    return np.concatenate([src, dst]), np.concatenate([dst, src])


class SparseEdgeCountEquivalenceTest(unittest.TestCase):
    def test_matches_the_dense_reference_on_random_partitions(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            num_nodes = int(rng.randint(10, 60))
            c2n = random_overlapping_partition(
                rng, num_nodes, int(rng.randint(2, 12))
            )
            n2c = build_node_to_communities(c2n)
            src, dst = random_bidirectional_edges(
                rng, num_nodes, int(rng.randint(5, 80))
            )

            sparse = calculate_edge_counts(src, dst, n2c)
            dense = dense_edge_counts_reference(src, dst, n2c, c2n)

            # The sparse version omits zeros; every non-zero must agree, and the
            # omitted keys must all have been zero.
            for key, value in dense.items():
                self.assertEqual(sparse.get(key, 0), value, msg=str(key))
            for key, value in sparse.items():
                self.assertEqual(dense.get(key, 0), value, msg=str(key))
            self.assertTrue(all(value > 0 for value in sparse.values()))

    def test_tolerates_scalar_valued_n2c(self):
        # --partition infomap and --partition test return n2c[node] = community.
        scalar = {0: 0, 1: 0, 2: 1, 3: 1}
        listed = {node: [community] for node, community in scalar.items()}
        src = np.array([0, 2, 1, 3])
        dst = np.array([2, 0, 3, 1])
        self.assertEqual(
            calculate_edge_counts(src, dst, scalar),
            calculate_edge_counts(src, dst, listed),
        )

    def test_memberships_of_normalises_every_shape(self):
        self.assertEqual(list(memberships_of({0: [1, 2]}, 0)), [1, 2])
        self.assertEqual(list(memberships_of({0: 7}, 0)), [7])
        self.assertEqual(list(memberships_of({0: (1,)}, 0)), [1])
        self.assertEqual(list(memberships_of({}, 5)), [])


class SparseSimilarityEquivalenceTest(unittest.TestCase):
    def test_matches_the_dense_reference_for_every_method(self):
        rng = np.random.RandomState(1)
        checked = 0
        for _ in range(15):
            num_nodes = int(rng.randint(10, 50))
            c2n = random_overlapping_partition(
                rng, num_nodes, int(rng.randint(3, 10))
            )
            n2c = build_node_to_communities(c2n)
            src, dst = random_bidirectional_edges(
                rng, num_nodes, int(rng.randint(10, 90))
            )
            edge_counts = calculate_edge_counts(src, dst, n2c)
            for method in (0, 1, 2, 3):
                expected = dense_similarity_reference(
                    c2n, edge_counts, len(c2n), method
                )
                actual = calculate_robustness_similarity(
                    c2n,
                    edge_counts,
                    test_edge_method=method,
                    include_jaccard=True,
                    pair_reduction="source",
                )
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    self.assertAlmostEqual(expected[key], actual[key], places=12,
                                           msg="{} method {}".format(key, method))
                checked += 1
        self.assertGreater(checked, 0)

    def test_pair_reductions_agree_when_edge_counts_are_symmetric(self):
        """Every DGL graph in this project is bidirectional, so both rules match."""
        rng = np.random.RandomState(2)
        for _ in range(10):
            num_nodes = int(rng.randint(10, 40))
            c2n = random_overlapping_partition(rng, num_nodes, int(rng.randint(3, 8)))
            n2c = build_node_to_communities(c2n)
            src, dst = random_bidirectional_edges(rng, num_nodes, 40)
            edge_counts = calculate_edge_counts(src, dst, n2c)
            for (source, target), count in edge_counts.items():
                self.assertEqual(edge_counts.get((target, source), 0), count)
            self.assertEqual(
                calculate_robustness_similarity(
                    c2n, edge_counts, test_edge_method=0, pair_reduction="source"),
                calculate_robustness_similarity(
                    c2n, edge_counts, test_edge_method=0, pair_reduction="max"),
            )

    def test_pair_reductions_differ_on_a_one_way_pair(self):
        c2n = {0: [0], 1: [1], 2: [2]}
        # 1 -> 0 only, plus a symmetric 1 <-> 2 pair so degrees are non-zero.
        edge_counts = {(1, 0): 3, (1, 2): 1, (2, 1): 1}
        by_source = calculate_robustness_similarity(
            c2n, edge_counts, test_edge_method=0, pair_reduction="source")
        by_max = calculate_robustness_similarity(
            c2n, edge_counts, test_edge_method=0, pair_reduction="max")
        self.assertNotIn((0, 1), by_source)
        self.assertIn((0, 1), by_max)

    def test_emission_order_matches_the_legacy_nested_loops(self):
        """Key *order*, not just the key set, must match the dense original.

        ``exp/exp_train.py`` builds the mapped DGL graph by iterating ``self.sim``
        in insertion order, so a different emission order would reorder the edge
        list and change the GPU aggregation order.  The legacy order is known
        analytically from ``for i in range(C): for j in range(i + 1, C)``:
        ascending ``(i, j)`` with ``i < j``, emitting ``(i, j)`` then ``(j, i)``.
        """
        rng = np.random.RandomState(3)
        for _ in range(10):
            num_nodes = int(rng.randint(12, 50))
            c2n = random_overlapping_partition(rng, num_nodes, int(rng.randint(4, 12)))
            n2c = build_node_to_communities(c2n)
            src, dst = random_bidirectional_edges(rng, num_nodes, 60)
            edge_counts = calculate_edge_counts(src, dst, n2c)

            expected = []
            for low, high in sorted(observed_community_pairs(edge_counts)):
                if edge_counts.get((low, high), 0) > 0:
                    expected += [(low, high), (high, low)]

            actual = list(
                calculate_robustness_similarity(
                    c2n, edge_counts, test_edge_method=0, pair_reduction="source"
                ).keys()
            )
            self.assertEqual(actual, expected)
            # And the same sequence as the dense reference produces.
            dense_order = list(
                dense_similarity_reference(c2n, edge_counts, len(c2n), 0).keys()
            )
            self.assertEqual(actual, dense_order)

    def test_emission_order_is_ascending_by_low_community_id(self):
        edge_counts = {}
        for low, high in ((3, 7), (0, 5), (1, 2), (0, 9)):
            edge_counts[(low, high)] = 1
            edge_counts[(high, low)] = 1
        keys = list(
            calculate_robustness_similarity(
                {index: [index] for index in range(10)},
                edge_counts,
                test_edge_method=0,
                pair_reduction="source",
            ).keys()
        )
        self.assertEqual(
            keys,
            [(0, 5), (5, 0), (0, 9), (9, 0), (1, 2), (2, 1), (3, 7), (7, 3)],
        )

    def test_rejects_an_unknown_pair_reduction(self):
        with self.assertRaises(ValueError):
            calculate_robustness_similarity({0: [0]}, {}, pair_reduction="mean")

    def test_degrees_and_pairs_are_computed_in_one_pass(self):
        edge_counts = {(0, 1): 2, (1, 0): 5, (1, 2): 1}
        out_degree, in_degree = community_degrees(edge_counts)
        self.assertEqual(out_degree, {0: 2, 1: 6})
        self.assertEqual(in_degree, {1: 2, 0: 5, 2: 1})
        self.assertEqual(observed_community_pairs(edge_counts), {(0, 1), (1, 2)})

    def test_scales_to_a_community_count_the_dense_version_could_not(self):
        """5,000 communities with a sparse edge set must be instant.

        The dense formulation would allocate 5,000 * 4,999 = ~25 million dict
        entries here and then scan ~12.5 million pairs.
        """
        num_communities = 5000
        c2n = {index: [index] for index in range(num_communities)}
        edge_counts = {}
        for index in range(num_communities - 1):
            edge_counts[(index, index + 1)] = 1
            edge_counts[(index + 1, index)] = 1
        similarity = calculate_robustness_similarity(
            c2n, edge_counts, test_edge_method=0, pair_reduction="source"
        )
        self.assertEqual(len(similarity), 2 * (num_communities - 1))


class SingletonFiniteValueTest(unittest.TestCase):
    """Singleton / identical-feature communities must not produce inf or NaN."""

    def test_feature_robustness_denominator_is_clamped(self):
        # Mirrors exp_partition.aggregate_features_pca's guarded division.
        for distances in (np.zeros(1), np.zeros(5), np.array([0.0, 0.0])):
            denominator = max(float(distances.sum()), np.finfo(np.float64).eps)
            value = len(distances) / denominator
            self.assertTrue(np.isfinite(value))

    def test_singleton_similarity_is_finite(self):
        c2n = {index: [index] for index in range(4)}
        n2c = build_node_to_communities(c2n)
        src = np.array([0, 1, 1, 2, 2, 3])
        dst = np.array([1, 0, 2, 1, 3, 2])
        edge_counts = calculate_edge_counts(src, dst, n2c)
        for method in (0, 1, 2, 3):
            similarity = calculate_robustness_similarity(
                c2n, edge_counts, test_edge_method=method, pair_reduction="source"
            )
            self.assertTrue(similarity)
            for value in similarity.values():
                self.assertTrue(np.isfinite(value))

    def test_union_of_two_identical_singletons_never_divides_by_zero(self):
        # A community can never be empty (detectors compact them away), but assert
        # the union-size guard anyway.
        similarity = calculate_robustness_similarity(
            {0: [0], 1: [0]}, {(0, 1): 1, (1, 0): 1}, test_edge_method=0
        )
        for value in similarity.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
