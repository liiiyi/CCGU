"""Tests for the mapping rules that Partition and Unlearn must agree on.

The unlearning path recomputes Equations (8), (9) and (11) for the communities a
deletion touched, and leaves every other mapped node alone.  If the two stages
implement a rule differently, a deletion silently rewrites parts of the mapped
graph it never touched -- so the agreement itself is worth a test.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.unlearning_core import (  # noqa: E402
    build_node_to_communities,
    calculate_edge_counts,
    calculate_robustness_similarity,
    remap_rows,
    remove_nodes_and_reindex,
)
from lib_utils.stats import majority_label  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def elbow_threshold(distances):
    """Equation (8), written out independently of either stage's class."""
    if len(distances) <= 1:
        return np.inf
    ordered = np.sort(distances)
    return ordered[int(np.argmax(np.diff(ordered)))]


class MajorityLabelTest(unittest.TestCase):
    def test_picks_the_most_frequent_value(self):
        self.assertEqual(majority_label([3, 1, 3, 2, 3]), 3)

    def test_breaks_ties_towards_the_smallest_value(self):
        self.assertEqual(majority_label([2, 2, 5, 5]), 2)

    def test_accepts_a_single_value(self):
        self.assertEqual(majority_label([7]), 7)

    def test_accepts_nested_shapes(self):
        self.assertEqual(majority_label(np.array([[1, 1], [2, 3]])), 1)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            majority_label([])


class ElbowThresholdTest(unittest.TestCase):
    """Equation (8) must be the consecutive difference, not a centred gradient."""

    def test_cuts_at_the_largest_gap(self):
        distances = np.array([0.1, 0.15, 0.2, 5.0, 5.1])
        threshold = elbow_threshold(distances)
        self.assertAlmostEqual(threshold, 0.2)
        self.assertEqual(int((distances <= threshold).sum()), 3)

    def test_np_gradient_would_cut_somewhere_else(self):
        # Documents why the centred difference is wrong here: it picks a different
        # index, so Partition and Unlearn disagreed about which nodes vote.
        distances = np.array([0.1, 0.15, 0.2, 5.0, 5.1])
        ordered = np.sort(distances)
        centred = ordered[int(np.argmax(np.gradient(ordered)))]
        self.assertNotAlmostEqual(centred, elbow_threshold(distances))

    def test_single_distance_keeps_everything(self):
        self.assertTrue(np.isinf(elbow_threshold(np.array([0.5]))))

    @unittest.skipUnless(HAS_SKLEARN and HAS_TORCH,
                         "needs the full environment to import both stages")
    def test_partition_and_unlearn_agree(self):
        from exp.exp_partition import GraphCommunityPartition
        from exp.exp_unlearn import Unlearn

        rng = np.random.RandomState(0)
        for _ in range(50):
            distances = rng.rand(rng.randint(2, 30))
            reference = elbow_threshold(distances)
            # Neither method touches `self`, so call them unbound.
            self.assertAlmostEqual(
                GraphCommunityPartition.calculate_threshold(None, distances), reference
            )
            self.assertAlmostEqual(
                Unlearn.calculate_threshold(None, distances), reference
            )


class Equation11Test(unittest.TestCase):
    """``test_edge_method=0`` must be Equation (11) exactly."""

    def test_matches_the_closed_form_on_a_two_community_graph(self):
        c2n = {0: [0, 1], 1: [2, 3]}
        # Two directed edges each way between the communities.
        src = np.array([0, 2, 1, 3])
        dst = np.array([2, 0, 3, 1])
        n2c = build_node_to_communities(c2n)
        counts = calculate_edge_counts(src, dst, n2c)
        similarity = calculate_robustness_similarity(
            c2n, counts, test_edge_method=0, include_jaccard=True
        )

        s_ij = 2.0            # edges from community 0 into community 1
        out_degree = 2.0      # community 0's total out-edges to other communities
        in_degree = 2.0       # community 1's total in-edges from other communities
        union = 4.0           # |C_0 union C_1|
        expected = (s_ij / np.sqrt(out_degree)) * (s_ij / np.sqrt(in_degree)) + s_ij / union
        self.assertAlmostEqual(similarity[(0, 1)], expected)
        self.assertAlmostEqual(similarity[(1, 0)], expected)

    def test_method_3_drops_the_robustness_term(self):
        c2n = {0: [0, 1], 1: [2, 3]}
        n2c = build_node_to_communities(c2n)
        counts = calculate_edge_counts(np.array([0, 2]), np.array([2, 0]), n2c)
        similarity = calculate_robustness_similarity(
            c2n, counts, test_edge_method=3, include_jaccard=True
        )
        self.assertAlmostEqual(similarity[(0, 1)], 1.0 / 4.0)

    def test_unconnected_communities_get_no_score(self):
        c2n = {0: [0], 1: [1], 2: [2]}
        n2c = build_node_to_communities(c2n)
        counts = calculate_edge_counts(np.array([0]), np.array([1]), n2c)
        similarity = calculate_robustness_similarity(c2n, counts, test_edge_method=0)
        self.assertIn((0, 1), similarity)
        self.assertNotIn((0, 2), similarity)
        self.assertNotIn((1, 2), similarity)

    def test_rejects_an_unknown_method(self):
        with self.assertRaises(ValueError):
            calculate_robustness_similarity({0: [0]}, {}, test_edge_method=7)


class DeletionBookkeepingTest(unittest.TestCase):
    def test_deleting_everything_leaves_no_community(self):
        c2n = {0: [0, 1], 1: [1]}
        updated, old_to_new, affected_old, affected_new = remove_nodes_and_reindex(
            c2n, [0, 1]
        )
        self.assertEqual(updated, {})
        self.assertEqual(old_to_new, {})
        self.assertEqual(affected_old, {0, 1})
        self.assertEqual(affected_new, set())
        self.assertEqual(remap_rows(["a", "b"], old_to_new), [])

    def test_deleting_an_absent_node_changes_nothing(self):
        c2n = {0: [0, 1], 1: [2]}
        updated, old_to_new, affected_old, _ = remove_nodes_and_reindex(c2n, [99])
        self.assertEqual(updated, c2n)
        self.assertEqual(old_to_new, {0: 0, 1: 1})
        self.assertEqual(affected_old, set())

    def test_reindexing_keeps_rows_aligned_with_communities(self):
        c2n = {0: [0], 1: [1], 2: [2], 3: [3]}
        updated, old_to_new, _, _ = remove_nodes_and_reindex(c2n, [1, 2])
        self.assertEqual(updated, {0: [0], 1: [3]})
        rows = np.array([[10.0], [11.0], [12.0], [13.0]])
        np.testing.assert_array_equal(
            np.asarray(remap_rows(rows, old_to_new)), np.array([[10.0], [13.0]])
        )

    def test_edge_counts_ignore_deleted_endpoints(self):
        c2n = {0: [0, 1], 1: [2, 3]}
        n2c = build_node_to_communities(c2n)
        before = calculate_edge_counts(np.array([1]), np.array([2]), n2c)
        self.assertEqual(before, {(0, 1): 1})

        updated, _, _, _ = remove_nodes_and_reindex(c2n, [1])
        after = calculate_edge_counts(
            np.array([1]), np.array([2]), build_node_to_communities(updated)
        )
        self.assertEqual(after, {})


if __name__ == "__main__":
    unittest.main()
