import unittest

import numpy as np

from exp.unlearning_core import (
    build_node_to_communities,
    calculate_edge_counts,
    calculate_robustness_similarity,
    remap_rows,
    remove_nodes_and_reindex,
)


class FakeDGLGraph:
    """Tiny DGL-shaped graph used only for CPU smoke testing."""

    def __init__(self, src, dst):
        self._src = np.asarray(src)
        self._dst = np.asarray(dst)

    def edges(self):
        return self._src, self._dst


class CommunityUnlearningSmokeTest(unittest.TestCase):
    def setUp(self):
        self.c2n = {
            0: [0, 1, 2],
            1: [2, 3],
            2: [4],
        }
        self.graph = FakeDGLGraph(
            src=[0, 1, 2, 2, 3, 3, 4, 4],
            dst=[1, 0, 3, 4, 2, 4, 2, 3],
        )

    def test_overlapping_node_is_removed_from_every_community(self):
        updated, old_to_new, affected_old, affected_new = (
            remove_nodes_and_reindex(self.c2n, [2])
        )

        self.assertEqual(updated, {0: [0, 1], 1: [3], 2: [4]})
        self.assertEqual(old_to_new, {0: 0, 1: 1, 2: 2})
        self.assertEqual(affected_old, {0, 1})
        self.assertEqual(affected_new, {0, 1})
        self.assertNotIn(2, build_node_to_communities(updated))

    def test_empty_community_is_removed_and_rows_are_reindexed(self):
        updated, old_to_new, _, _ = remove_nodes_and_reindex(
            self.c2n,
            [4],
        )

        self.assertEqual(updated, {0: [0, 1, 2], 1: [2, 3]})
        self.assertEqual(old_to_new, {0: 0, 1: 1})
        self.assertEqual(remap_rows(["a", "b", "c"], old_to_new), ["a", "b"])

    def test_deleted_node_contributes_no_mapped_edge(self):
        updated, _, _, _ = remove_nodes_and_reindex(self.c2n, [2])
        n2c = build_node_to_communities(updated)
        src, dst = self.graph.edges()
        counts = calculate_edge_counts(src, dst, n2c)
        similarity = calculate_robustness_similarity(
            updated,
            counts,
            test_edge_method=2,
        )

        self.assertNotIn(2, n2c)
        self.assertTrue(all(np.isfinite(score) for score in similarity.values()))
        self.assertEqual(similarity[(1, 2)], similarity[(2, 1)])


if __name__ == "__main__":
    unittest.main()
