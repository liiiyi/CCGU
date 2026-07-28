"""Tests for the community-scale controls and the tail-community repair.

Both are post-paper engineering extensions.  Every knob defaults to "off", so the
first test asserts that the defaults change nothing, and the rest pin down the
behaviour when they are switched on.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fake_graph import (  # noqa: E402
    FakeDGLGraph,
    cliques_with_tail,
    ring_of_cliques,
)

try:
    from exp.methods.CCP import CommunityCentricPartition
    CCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    CCP_AVAILABLE = False


def sizes_of(c2n):
    return sorted(len(nodes) for nodes in c2n.values())


def coverage_of(c2n, num_nodes):
    covered = set()
    for nodes in c2n.values():
        covered.update(nodes)
    return covered == set(range(num_nodes))


def membership_counts(c2n):
    counts = {}
    for nodes in c2n.values():
        for node in nodes:
            counts[node] = counts.get(node, 0) + 1
    return counts


@unittest.skipUnless(CCP_AVAILABLE, "CCP needs networkx and scipy")
class DefaultsUnchangedTest(unittest.TestCase):
    def setUp(self):
        self.graph = ring_of_cliques(num_cliques=8, clique_size=10)

    def run_ccp(self, **overrides):
        args = {"random_seed": 4}
        args.update(overrides)
        detector = CommunityCentricPartition(self.graph, args=args)
        return detector, detector.partition()

    def test_scale_controls_default_to_disabled(self):
        detector, (_, c2n, _, _) = self.run_ccp()
        self.assertEqual(detector.max_community_size, 0)
        self.assertEqual(detector.tail_min_size, 0)
        self.assertFalse(detector.scale_report["size_cap"]["enabled"])
        self.assertFalse(detector.scale_report["tail_repair"]["enabled"])
        # Explicit zeros must give the identical partition to omitting them.
        _, (_, explicit, _, _) = self.run_ccp(
            ccp_max_community_size=0, ccp_tail_min_size=0
        )
        self.assertEqual(c2n, explicit)

    def test_effective_config_is_recorded(self):
        detector, _ = self.run_ccp(ccp_theta=12, ccp_max_community_size=30,
                                   ccp_tail_min_size=4)
        config = detector.effective_config
        self.assertEqual(config["ccp_theta"], 12)
        self.assertEqual(config["ccp_max_community_size"], 30)
        self.assertEqual(config["ccp_tail_min_size"], 4)
        self.assertEqual(config["random_seed"], 4)
        self.assertEqual(detector.scale_report["effective_config"], config)

    def test_scale_report_records_every_stage(self):
        detector, _ = self.run_ccp(ccp_max_community_size=25, ccp_tail_min_size=3)
        for stage in ("after_overlap", "after_size_cap", "after_tail_repair", "final"):
            self.assertIn(stage, detector.scale_report)
            self.assertIn("size_max", detector.scale_report[stage])
            self.assertIn("num_communities", detector.scale_report[stage])


@unittest.skipUnless(CCP_AVAILABLE, "CCP needs networkx and scipy")
class ScaleMonotonicityTest(unittest.TestCase):
    """Sanity/monotonic checks on the interpretable knobs."""

    def setUp(self):
        self.graph = ring_of_cliques(num_cliques=10, clique_size=12)
        self.num_nodes = self.graph.num_nodes()

    def counts_for(self, **overrides):
        args = {"random_seed": 4, "ccp_max_communities_per_node": 1}
        args.update(overrides)
        _, c2n, num_communities, _ = CommunityCentricPartition(
            self.graph, args=args
        ).partition()
        return num_communities, c2n

    def test_smaller_theta_never_yields_fewer_communities(self):
        previous = None
        for theta in (60, 30, 15, 8, 4):
            count, c2n = self.counts_for(ccp_theta=theta)
            self.assertLessEqual(max(sizes_of(c2n)), theta)
            if previous is not None:
                self.assertGreaterEqual(count, previous)
            previous = count

    def test_higher_resolution_does_not_yield_fewer_communities(self):
        # theta is set high so the split threshold is not what drives the count.
        # A ring of perfect cliques responds to resolution as a step rather than a
        # ramp -- the structure is unambiguous until the resolution term dominates --
        # so the sweep spans two orders of magnitude and only monotonicity plus a
        # strict increase between the extremes is asserted.
        counts, medians = [], []
        for resolution in (0.5, 1.0, 4.0, 16.0, 64.0):
            count, c2n = self.counts_for(
                ccp_resolution=resolution, ccp_theta=10 ** 6
            )
            counts.append(count)
            sizes = sizes_of(c2n)
            medians.append(sizes[len(sizes) // 2])
        self.assertEqual(counts, sorted(counts),
                         "community count should be non-decreasing in resolution")
        self.assertGreater(counts[-1], counts[0])
        self.assertEqual(medians, sorted(medians, reverse=True),
                         "median community size should be non-increasing in resolution")

    def test_max_community_size_caps_the_largest_community(self):
        # Overlap on: without a cap the overlap stage grows blocks past theta.
        _, uncapped, _, _ = CommunityCentricPartition(
            self.graph,
            args={"random_seed": 4, "ccp_theta": 12,
                  "ccp_max_communities_per_node": 3,
                  "ccp_overlap_threshold": 0.05},
        ).partition()
        self.assertGreater(max(sizes_of(uncapped)), 12)

        for cap in (40, 25, 14):
            _, capped, _, _ = CommunityCentricPartition(
                self.graph,
                args={"random_seed": 4, "ccp_theta": 12,
                      "ccp_max_communities_per_node": 3,
                      "ccp_overlap_threshold": 0.05,
                      "ccp_max_community_size": cap},
            ).partition()
            self.assertLessEqual(max(sizes_of(capped)), cap)
            self.assertTrue(coverage_of(capped, self.num_nodes))

    def test_cap_below_theta_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            CommunityCentricPartition(
                self.graph, args={"ccp_theta": 20, "ccp_max_community_size": 5}
            )
        self.assertIn("ccp_max_community_size", str(caught.exception))

    def test_tail_min_above_cap_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            CommunityCentricPartition(
                self.graph,
                args={"ccp_theta": 10, "ccp_max_community_size": 20,
                      "ccp_tail_min_size": 30},
            )
        self.assertIn("ccp_tail_min_size", str(caught.exception))


@unittest.skipUnless(CCP_AVAILABLE, "CCP needs networkx and scipy")
class TailRepairTest(unittest.TestCase):
    def setUp(self):
        # Four stable 12-cliques plus five weakly attached triangles: a genuine
        # tail sitting next to stable communities, which is the situation the
        # repair exists for.
        self.graph = cliques_with_tail(
            num_cliques=4, clique_size=12, num_tail_triangles=5
        )
        self.num_nodes = self.graph.num_nodes()

    def run_ccp(self, **overrides):
        args = {"random_seed": 4, "ccp_theta": 12, "ccp_max_communities_per_node": 1}
        args.update(overrides)
        detector = CommunityCentricPartition(self.graph, args=args)
        return detector, detector.partition()

    def test_fixture_really_has_a_tail(self):
        _, (_, before, _, _) = self.run_ccp()
        sizes = sizes_of(before)
        self.assertLess(min(sizes), 5, "fixture must produce small communities")
        self.assertGreaterEqual(max(sizes), 5, "and stable ones to merge into")

    def test_repair_removes_small_communities_and_keeps_coverage(self):
        _, (_, before, _, _) = self.run_ccp()
        detector, (_, after, _, _) = self.run_ccp(ccp_tail_min_size=5)

        self.assertTrue(coverage_of(after, self.num_nodes))
        self.assertLess(len(after), len(before), "merging must reduce the count")
        self.assertGreaterEqual(min(sizes_of(after)), 5,
                                "no community below the threshold may survive when "
                                "every tail block had a stable neighbour")
        stats = detector.scale_report["tail_repair"]
        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["tail_before"], 0)
        self.assertGreater(stats["stable_targets"], 0)
        self.assertGreater(stats["merged"], 0)
        # Tail blocks are never targets, so each one is either merged away or kept
        # for lack of a stable neighbour.  There is no third bucket.
        self.assertEqual(
            stats["tail_before"], stats["merged"] + stats["kept_no_target"]
        )
        self.assertEqual(len(before) - len(after), stats["merged"])

    def test_no_stable_target_is_an_honest_no_op(self):
        """With min_size above every block, nothing may be rearranged."""
        _, (_, before, _, _) = self.run_ccp()
        detector, (_, after, _, _) = self.run_ccp(ccp_tail_min_size=1000)
        self.assertEqual(before, after)
        stats = detector.scale_report["tail_repair"]
        self.assertEqual(stats["stable_targets"], 0)
        self.assertEqual(stats["merged"], 0)
        self.assertEqual(stats["kept_no_target"], stats["tail_before"])

    def test_repair_is_deterministic(self):
        _, (_, first, _, _) = self.run_ccp(ccp_tail_min_size=5)
        _, (_, second, _, _) = self.run_ccp(ccp_tail_min_size=5)
        self.assertEqual(first, second)

    def test_decisions_do_not_depend_on_block_order(self):
        """Frozen scoring means relabelling the blocks cannot change the outcome."""
        detector = CommunityCentricPartition(
            self.graph,
            args={"random_seed": 4, "ccp_theta": 12,
                  "ccp_max_communities_per_node": 1, "ccp_tail_min_size": 5},
        )
        coarse = detector._coarse_partition()
        blocks = detector._fine_partition(coarse)
        representations = detector._representations()
        forward, stats_forward = detector._repair_tail(
            [list(block) for block in blocks], representations
        )
        reverse, stats_reverse = detector._repair_tail(
            [list(block) for block in reversed(blocks)], representations
        )
        self.assertEqual(sorted(forward), sorted(reverse))
        self.assertEqual(stats_forward["merged"], stats_reverse["merged"])
        self.assertEqual(stats_forward["kept_no_target"],
                         stats_reverse["kept_no_target"])

    def test_repair_preserves_overlap(self):
        _, (_, before, _, _) = self.run_ccp(
            ccp_max_communities_per_node=3, ccp_overlap_threshold=0.05
        )
        _, (_, after, _, _) = self.run_ccp(
            ccp_max_communities_per_node=3, ccp_overlap_threshold=0.05,
            ccp_tail_min_size=5,
        )
        before_overlap = sum(1 for count in membership_counts(before).values() if count > 1)
        after_overlap = sum(1 for count in membership_counts(after).values() if count > 1)
        self.assertGreater(before_overlap, 0)
        self.assertGreater(after_overlap, 0, "repair must not flatten the overlap away")
        self.assertTrue(coverage_of(after, self.num_nodes))

    def test_isolated_nodes_are_kept_when_no_target_exists(self):
        # Two connected nodes plus two isolated ones: the isolated singletons have
        # no neighbouring community to merge into and must survive untouched.
        # A 6-clique (a stable target) plus two fully isolated nodes.
        src, dst = [], []
        for u in range(6):
            for v in range(u + 1, 6):
                src += [u, v]
                dst += [v, u]
        graph = FakeDGLGraph(src, dst, num_nodes=8,
                             feat=np.eye(8, dtype=np.float32))
        detector = CommunityCentricPartition(
            graph, args={"random_seed": 1, "ccp_theta": 10,
                         "ccp_max_communities_per_node": 1,
                         "ccp_tail_min_size": 4}
        )
        _, c2n, _, _ = detector.partition()
        self.assertTrue(coverage_of(c2n, 8))
        self.assertIn([6], list(c2n.values()), "isolated node 6 must survive")
        self.assertIn([7], list(c2n.values()), "isolated node 7 must survive")
        stats = detector.scale_report["tail_repair"]
        self.assertGreater(stats["stable_targets"], 0)
        self.assertEqual(stats["kept_no_target"], 2)
        self.assertEqual(stats["merged"], 0)

    def test_protected_singletons_are_never_merged_or_grown(self):
        held_out = np.zeros(self.num_nodes, dtype=bool)
        held_out[[0, 13, 27, 40]] = True
        graph = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(),
            label=self.graph.ndata["label"].numpy(),
            masks={"test_mask": held_out},
        )
        detector = CommunityCentricPartition(
            graph,
            args={"random_seed": 4, "ccp_theta": 12,
                  "ccp_max_communities_per_node": 1,
                  "ccp_protect_eval_nodes": "test",
                  "ccp_tail_min_size": 5},
        )
        _, c2n, _, _ = detector.partition()

        for node in np.nonzero(held_out)[0]:
            memberships = [c for c, nodes in c2n.items() if int(node) in nodes]
            self.assertEqual(len(memberships), 1,
                             "held-out node {} gained a membership".format(node))
            self.assertEqual(c2n[memberships[0]], [int(node)],
                             "held-out node {} was merged away".format(node))
        self.assertGreater(detector.scale_report["tail_repair"]["kept_protected"], 0)
        self.assertTrue(coverage_of(c2n, self.num_nodes))

    def test_repair_never_reads_labels(self):
        """The decision must be identical with the labels removed or shuffled."""
        with_labels = self.run_ccp(ccp_tail_min_size=5)[1][1]

        shuffled = self.graph.ndata["label"].numpy()[::-1].copy()
        relabelled = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(), label=shuffled,
        )
        _, other, _, _ = CommunityCentricPartition(
            relabelled,
            args={"random_seed": 4, "ccp_theta": 12,
                  "ccp_max_communities_per_node": 1, "ccp_tail_min_size": 5},
        ).partition()
        self.assertEqual(with_labels, other)

        unlabelled = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(),
        )
        _, without, _, _ = CommunityCentricPartition(
            unlabelled,
            args={"random_seed": 4, "ccp_theta": 12,
                  "ccp_max_communities_per_node": 1, "ccp_tail_min_size": 5},
        ).partition()
        self.assertEqual(with_labels, without)

    def test_higher_tail_min_size_never_increases_the_community_count(self):
        counts = []
        for min_size in (0, 3, 5, 12):
            _, (_, c2n, count, _) = self.run_ccp(ccp_tail_min_size=min_size)
            counts.append(count)
            self.assertTrue(coverage_of(c2n, self.num_nodes))
        self.assertEqual(counts, sorted(counts, reverse=True))


if __name__ == "__main__":
    unittest.main()
