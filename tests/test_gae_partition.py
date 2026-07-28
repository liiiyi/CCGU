"""Tests for the post-paper graph-auto-encoder community detector.

These need torch + DGL, so they skip cleanly in the lightweight CI job that
installs only numpy/scipy/scikit-learn/networkx.  Everything that can be checked
without a trained encoder -- argument validation, negative sampling -- is split
out so it exercises as little of the stack as possible.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fake_graph import FakeDGLGraph, ring_of_cliques  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DGL = importlib.util.find_spec("dgl") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_NETWORKX = importlib.util.find_spec("networkx") is not None
FULL_STACK = HAS_TORCH and HAS_DGL and HAS_SKLEARN and HAS_NETWORKX

if HAS_NETWORKX:
    from exp.methods.GAEPartition import GAEPartition  # noqa: E402


def blocks_of(c2n):
    return sorted(sorted(nodes) for nodes in c2n.values())


@unittest.skipUnless(HAS_NETWORKX, "needs networkx for the CCP base class")
class ValidationTest(unittest.TestCase):
    """Argument validation happens in __init__, before anything is trained."""

    def setUp(self):
        self.graph = ring_of_cliques(num_cliques=3, clique_size=5)

    def build(self, **overrides):
        args = {"random_seed": 1}
        args.update(overrides)
        return GAEPartition(self.graph, args=args)

    def test_rejects_non_positive_sizes(self):
        for key in ("gae_hidden_dim", "gae_latent_dim", "gae_epochs"):
            with self.assertRaises(ValueError, msg=key) as caught:
                self.build(**{key: 0})
            self.assertIn(key, str(caught.exception),
                          "the error must name the offending flag")

    def test_rejects_non_positive_learning_rate(self):
        with self.assertRaises(ValueError):
            self.build(gae_learning_rate=0.0)

    def test_rejects_unknown_device(self):
        with self.assertRaises(ValueError) as caught:
            self.build(gae_device="tpu")
        self.assertIn("gae_device", str(caught.exception))

    def test_rejects_negative_rounding(self):
        with self.assertRaises(ValueError):
            self.build(gae_round_decimals=-1)

    def test_accepts_and_records_valid_values(self):
        detector = self.build(
            gae_hidden_dim=8, gae_latent_dim=4, gae_epochs=3,
            gae_learning_rate=0.05, gae_device="cpu", gae_round_decimals=4,
        )
        config = detector.effective_config
        self.assertEqual(config["gae_hidden_dim"], 8)
        self.assertEqual(config["gae_latent_dim"], 4)
        self.assertEqual(config["gae_epochs"], 3)
        self.assertEqual(config["gae_device"], "cpu")
        self.assertEqual(config["gae_round_decimals"], 4)
        # The scale knobs are inherited unchanged.
        self.assertEqual(config["ccp_theta"], 20)


@unittest.skipUnless(HAS_TORCH and HAS_NETWORKX, "negative sampling needs torch")
class NegativeSamplingTest(unittest.TestCase):
    """Sampled negatives must be true non-edges, and reproducible."""

    def setUp(self):
        self.graph = ring_of_cliques(num_cliques=4, clique_size=6)
        self.detector = GAEPartition(self.graph, args={"random_seed": 7})
        self.edge_codes = self.detector._edge_codes()
        coo = self.detector.adjacency.tocoo()
        self.edges = set(zip(coo.row.tolist(), coo.col.tolist()))

    def generator(self, seed=7):
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def test_never_samples_a_self_pair_or_an_existing_edge(self):
        source, target, rounds = self.detector._sample_true_negatives(
            500, self.generator(), self.edge_codes
        )
        self.assertEqual(source.size, 500)
        self.assertEqual(target.size, 500)
        self.assertGreaterEqual(rounds, 1)
        self.assertFalse(np.any(source == target), "self-pairs must be rejected")
        for u, v in zip(source.tolist(), target.tolist()):
            self.assertNotIn((u, v), self.edges,
                             "sampled ({}, {}) is an existing edge".format(u, v))

    def test_is_deterministic_for_a_given_seed(self):
        first = self.detector._sample_true_negatives(
            200, self.generator(), self.edge_codes
        )
        second = self.detector._sample_true_negatives(
            200, self.generator(), self.edge_codes
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_different_seeds_give_different_negatives(self):
        first = self.detector._sample_true_negatives(
            200, self.generator(7), self.edge_codes
        )
        other = self.detector._sample_true_negatives(
            200, self.generator(8), self.edge_codes
        )
        self.assertFalse(np.array_equal(first[0], other[0]))

    def test_zero_request_returns_empty(self):
        source, target, rounds = self.detector._sample_true_negatives(
            0, self.generator(), self.edge_codes
        )
        self.assertEqual(source.size, 0)
        self.assertEqual(target.size, 0)
        self.assertEqual(rounds, 0)

    def test_edge_codes_cover_every_directed_edge(self):
        coo = self.detector.adjacency.tocoo()
        expected = np.sort(
            coo.row.astype(np.int64) * self.detector.num_nodes
            + coo.col.astype(np.int64)
        )
        np.testing.assert_array_equal(self.edge_codes, expected)


@unittest.skipUnless(FULL_STACK, "the trained encoder needs torch, dgl and sklearn")
class TrainedDetectorTest(unittest.TestCase):
    def setUp(self):
        self.graph = ring_of_cliques(num_cliques=6, clique_size=10)
        self.num_nodes = self.graph.num_nodes()

    def run_gae(self, **overrides):
        # A small encoder keeps the CPU test fast; the reproduction sweep uses the
        # defaults.
        args = {
            "random_seed": 4, "gae_device": "cpu", "gae_epochs": 20,
            "gae_hidden_dim": 16, "gae_latent_dim": 8, "ccp_theta": 10,
        }
        args.update(overrides)
        detector = GAEPartition(self.graph, args=args)
        return detector, detector.partition()

    def test_cpu_path_is_bitwise_reproducible(self):
        _, (_, first, _, _) = self.run_gae()
        _, (_, second, _, _) = self.run_gae()
        self.assertEqual(blocks_of(first), blocks_of(second))

    def test_covers_every_node_with_contiguous_ids(self):
        _, (n2c, c2n, num_communities, elapsed) = self.run_gae()
        self.assertEqual(num_communities, len(c2n))
        self.assertEqual(sorted(c2n), list(range(num_communities)))
        covered = set()
        for nodes in c2n.values():
            self.assertTrue(nodes)
            covered.update(nodes)
        self.assertEqual(covered, set(range(self.num_nodes)))
        rebuilt = {}
        for community, nodes in c2n.items():
            for node in nodes:
                rebuilt.setdefault(node, []).append(community)
        self.assertEqual({k: sorted(v) for k, v in n2c.items()},
                         {k: sorted(v) for k, v in rebuilt.items()})
        self.assertGreaterEqual(elapsed, 0.0)

    def test_reports_the_training_run(self):
        detector, _ = self.run_gae()
        report = detector.gae_report
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["epochs"], 20)
        self.assertEqual(report["negative_shortfall"], 0)
        self.assertGreater(report["num_positive_edges"], 0)
        self.assertLess(report["final_loss"], report["first_loss"],
                        "link reconstruction should improve")
        self.assertIn("gae", detector.scale_report)

    def test_representations_are_the_learned_embedding(self):
        """Overlap and tail repair must score on the learned space, not raw features."""
        from exp.methods.CCP import CommunityCentricPartition

        detector, _ = self.run_gae()
        learned = detector._representations()
        self.assertEqual(learned.shape, (self.num_nodes, 8))

        baseline = CommunityCentricPartition(
            self.graph, args={"random_seed": 4, "ccp_theta": 10}
        )._representations()
        self.assertNotEqual(learned.shape, baseline.shape,
                            "the learned latent width must differ from the raw features")

    def test_target_size_controls_the_cluster_count(self):
        """k = ceil(n / ccp_theta), so a smaller target must give more clusters."""
        detector_coarse, (_, _, coarse_count, _) = self.run_gae(ccp_theta=20)
        detector_fine, (_, _, fine_count, _) = self.run_gae(ccp_theta=5)
        self.assertGreater(fine_count, coarse_count)
        self.assertEqual(
            detector_coarse.gae_report["num_clusters_requested"],
            int(np.ceil(self.num_nodes / 20.0)),
        )
        self.assertEqual(
            detector_fine.gae_report["num_clusters_requested"],
            int(np.ceil(self.num_nodes / 5.0)),
        )

    def test_labels_never_influence_the_partition(self):
        _, (_, with_labels, _, _) = self.run_gae()

        shuffled = self.graph.ndata["label"].numpy()[::-1].copy()
        relabelled = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(), label=shuffled,
        )
        detector = GAEPartition(
            relabelled,
            args={"random_seed": 4, "gae_device": "cpu", "gae_epochs": 20,
                  "gae_hidden_dim": 16, "gae_latent_dim": 8, "ccp_theta": 10},
        )
        self.assertEqual(blocks_of(with_labels), blocks_of(detector.partition()[1]))

        unlabelled = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(),
        )
        detector = GAEPartition(
            unlabelled,
            args={"random_seed": 4, "gae_device": "cpu", "gae_epochs": 20,
                  "gae_hidden_dim": 16, "gae_latent_dim": 8, "ccp_theta": 10},
        )
        self.assertEqual(blocks_of(with_labels), blocks_of(detector.partition()[1]))

    def test_protected_singletons_survive(self):
        held_out = np.zeros(self.num_nodes, dtype=bool)
        held_out[[0, 11, 25, 41]] = True
        graph = FakeDGLGraph(
            self.graph._src, self.graph._dst, self.num_nodes,
            feat=self.graph.ndata["feat"].numpy(),
            label=self.graph.ndata["label"].numpy(),
            masks={"test_mask": held_out},
        )
        detector = GAEPartition(
            graph,
            args={"random_seed": 4, "gae_device": "cpu", "gae_epochs": 20,
                  "gae_hidden_dim": 16, "gae_latent_dim": 8, "ccp_theta": 10,
                  "ccp_protect_eval_nodes": "test", "ccp_tail_min_size": 4},
        )
        _, c2n, _, _ = detector.partition()
        for node in np.nonzero(held_out)[0]:
            memberships = [c for c, nodes in c2n.items() if int(node) in nodes]
            self.assertEqual(len(memberships), 1)
            self.assertEqual(c2n[memberships[0]], [int(node)])

    def test_works_without_node_features(self):
        graph = FakeDGLGraph(self.graph._src, self.graph._dst, self.num_nodes)
        detector = GAEPartition(
            graph,
            args={"random_seed": 4, "gae_device": "cpu", "gae_epochs": 10,
                  "gae_hidden_dim": 8, "gae_latent_dim": 4, "ccp_theta": 10},
        )
        _, c2n, _, _ = detector.partition()
        covered = set(node for nodes in c2n.values() for node in nodes)
        self.assertEqual(covered, set(range(self.num_nodes)))
        self.assertIn("log-degree", detector.gae_report["feature_source"])

    def test_auto_device_falls_back_to_cpu(self):
        import torch

        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            detector = GAEPartition(
                self.graph,
                args={"random_seed": 4, "gae_device": "auto", "gae_epochs": 5,
                      "gae_hidden_dim": 8, "gae_latent_dim": 4, "ccp_theta": 10},
            )
            detector.partition()
            self.assertEqual(detector.gae_report["device"], "cpu")
        finally:
            torch.cuda.is_available = original

    def test_explicit_cuda_without_cuda_raises(self):
        import torch

        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            detector = GAEPartition(
                self.graph,
                args={"random_seed": 4, "gae_device": "cuda", "gae_epochs": 5,
                      "gae_hidden_dim": 8, "gae_latent_dim": 4},
            )
            with self.assertRaises(RuntimeError) as caught:
                detector.partition()
            self.assertIn("cuda", str(caught.exception))
        finally:
            torch.cuda.is_available = original

    def test_thread_count_is_restored_after_training(self):
        import torch

        torch.set_num_threads(4)
        self.run_gae()
        self.assertEqual(torch.get_num_threads(), 4)


if __name__ == "__main__":
    unittest.main()
