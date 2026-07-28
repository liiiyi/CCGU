"""Tests for the partition quality diagnostics and the fail-fast quality gate."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.methods.partition_diagnostics import (  # noqa: E402
    check_partition_health,
    disjoint_modularity,
    format_summary,
    summarise_partition,
)


class SummariseTest(unittest.TestCase):
    def setUp(self):
        # Node 2 belongs to two communities, node 5 to none.
        self.c2n = {0: [0, 1, 2], 1: [2, 3], 2: [4]}
        self.num_nodes = 6
        self.labels = np.array([0, 0, 1, 1, 2, 2])
        self.edges = (
            np.array([0, 1, 2, 3, 2, 4]),
            np.array([1, 0, 3, 2, 4, 2]),
        )

    def test_counts_communities_coverage_and_overlap(self):
        summary = summarise_partition(
            self.c2n, self.num_nodes, labels=self.labels, edges=self.edges,
            elapsed_seconds=1.25,
        )
        self.assertEqual(summary["num_communities"], 3)
        self.assertEqual(summary["num_nodes"], 6)
        self.assertAlmostEqual(summary["compression_ratio"], 0.5)
        self.assertAlmostEqual(summary["coverage"], 5 / 6.0)
        self.assertEqual(summary["uncovered_nodes"], 1)
        self.assertEqual(summary["overlap_histogram"]["1"], 4)
        self.assertEqual(summary["overlap_histogram"]["2"], 1)
        self.assertEqual(summary["overlap_histogram"]["0"], 1)
        self.assertAlmostEqual(summary["overlapping_node_fraction"], 1 / 6.0)
        self.assertAlmostEqual(summary["memberships_per_node_mean"], 6 / 5.0)
        self.assertEqual(summary["memberships_per_node_max"], 2)
        self.assertEqual(summary["community_size_min"], 1)
        self.assertEqual(summary["community_size_max"], 3)
        self.assertEqual(summary["singleton_communities"], 1)
        self.assertEqual(summary["tiny_communities_le2"], 2)
        self.assertAlmostEqual(summary["elapsed_seconds"], 1.25)

    def test_label_purity(self):
        summary = summarise_partition(self.c2n, self.num_nodes, labels=self.labels)
        # community 0 = labels (0,0,1) -> 2/3; community 1 = (1,1) -> 1; {4} -> 1.
        self.assertAlmostEqual(summary["label_purity_mean"], (2 / 3.0 + 1 + 1) / 3.0)

    def test_optional_inputs_are_optional(self):
        summary = summarise_partition(self.c2n, self.num_nodes)
        self.assertNotIn("label_purity_mean", summary)
        self.assertNotIn("modularity_primary", summary)
        self.assertNotIn("elapsed_seconds", summary)

    def test_format_summary_mentions_the_headline_numbers(self):
        summary = summarise_partition(
            self.c2n, self.num_nodes, labels=self.labels, edges=self.edges
        )
        text = format_summary(summary)
        self.assertIn("communities", text)
        self.assertIn("coverage", text)
        self.assertIn("overlap histogram", text)
        self.assertIn("label purity", text)


class ModularityTest(unittest.TestCase):
    def test_two_disconnected_triangles_reach_the_analytic_optimum(self):
        # Six undirected edges; each community holds 3 with volume 6 of 12.
        # Q = 2 * (3/6 - (6/12)^2) = 2 * 0.25 = 0.5.
        src = np.array([0, 1, 2, 3, 4, 5])
        dst = np.array([1, 2, 0, 4, 5, 3])
        primary = np.array([0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(
            disjoint_modularity(primary, src, dst, 6), 0.5, places=6
        )

    def test_all_nodes_in_one_community_gives_zero(self):
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        primary = np.zeros(3, dtype=np.int64)
        self.assertAlmostEqual(
            disjoint_modularity(primary, src, dst, 3), 0.0, places=6
        )

    def test_reversed_and_duplicate_edges_do_not_change_the_result(self):
        primary = np.array([0, 0, 0, 1, 1, 1])
        plain = disjoint_modularity(
            primary, np.array([0, 1, 2, 3, 4, 5]), np.array([1, 2, 0, 4, 5, 3]), 6
        )
        noisy = disjoint_modularity(
            primary,
            np.array([0, 1, 1, 2, 3, 4, 5, 0, 2]),
            np.array([1, 2, 0, 0, 4, 5, 3, 1, 2]),
            6,
        )
        self.assertAlmostEqual(plain, noisy, places=6)

    def test_edgeless_graph_is_not_a_number(self):
        primary = np.array([0, 1])
        self.assertTrue(
            np.isnan(disjoint_modularity(primary, np.array([]), np.array([]), 2))
        )


class HealthGateTest(unittest.TestCase):
    def test_healthy_partition_reports_no_problem(self):
        summary = summarise_partition(
            {index: [index] for index in range(20)}, num_nodes=20
        )
        self.assertEqual(check_partition_health(summary, min_communities=8), [])

    def test_too_few_communities_is_reported(self):
        summary = summarise_partition({0: [0, 1], 1: [2, 3]}, num_nodes=4)
        problems = check_partition_health(summary, min_communities=8)
        self.assertEqual(len(problems), 1)
        self.assertIn("only 2 communities", problems[0])

    def test_incomplete_coverage_is_reported(self):
        summary = summarise_partition(
            {index: [index] for index in range(10)}, num_nodes=12
        )
        problems = check_partition_health(summary, min_communities=8)
        self.assertEqual(len(problems), 1)
        self.assertIn("coverage", problems[0])
        self.assertIn("2 original nodes", problems[0])

    def test_both_problems_are_reported_together(self):
        summary = summarise_partition({0: [0, 1]}, num_nodes=5)
        self.assertEqual(len(check_partition_health(summary, min_communities=8)), 2)


if __name__ == "__main__":
    unittest.main()
