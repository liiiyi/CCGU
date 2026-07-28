"""Guard the preserved GNN backbones against being silently swapped or normalised.

The repository deliberately keeps two backbone families:

*   ``lib_gnn_model/<arch>/*_batch.py`` -- the GIF-style PyG nets driven by
    ``lib_gnn_model/node_classifier.py`` through a ``NeighborSampler``.  These set
    ``add_self_loops=True``, use GAT dropout 0.6 and 8x8 hidden heads.
*   ``lib_gnn_model/{GCN,GAT,GraphSAGE}.py`` -- the plain DGL nets that
    ``NodeClassifierDGL`` (and therefore the whole documented CGE pipeline) uses.

The author's earlier experiments found the GraphEraser-style backbone performed
much worse under nominally identical hyper-parameters, for reasons that were never
pinned down, so these implementations must stay as they are.  This module asserts
the wiring and the architectural constants so that any future edit that
"normalises" them fails loudly instead of quietly changing every reported number.
See exp/methods/README.md.
"""

import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_DGL = importlib.util.find_spec("dgl") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PYG = importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(HAS_TORCH and HAS_DGL, "needs torch and dgl")
class DGLBackboneWiringTest(unittest.TestCase):
    """``NodeClassifierDGL`` must resolve to the preserved DGL nets, unchanged."""

    def _build(self, name):
        from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL

        args = {"target_model": name}
        return NodeClassifierDGL(1433, 7, args).model

    def test_targets_resolve_to_the_preserved_classes(self):
        from lib_gnn_model.GAT import GATNet
        from lib_gnn_model.GCN import GCNNet
        from lib_gnn_model.GraphSAGE import GraphSAGENet

        self.assertIsInstance(self._build("GCN"), GCNNet)
        self.assertIsInstance(self._build("GAT"), GATNet)
        self.assertIsInstance(self._build("SAGE"), GraphSAGENet)

    def test_unsupported_target_model_raises_a_clear_error(self):
        from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL

        # MLP/GIN/SGC remain valid --target_model choices but are not wired to a
        # DGL net; the failure must name the model rather than crash obscurely.
        for name in ("MLP", "GIN", "SGC"):
            with self.assertRaises(ValueError) as caught:
                NodeClassifierDGL(8, 2, {"target_model": name})
            self.assertIn(name, str(caught.exception))

    def test_architecture_constants_are_unchanged(self):
        from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv

        gcn = self._build("GCN")
        self.assertIsInstance(gcn.conv1, GraphConv)
        self.assertIsInstance(gcn.conv2, GraphConv)
        self.assertEqual(tuple(gcn.conv1.weight.shape), (1433, 16))
        self.assertEqual(tuple(gcn.conv2.weight.shape), (16, 7))
        self.assertEqual(gcn.dropout_rate, 0.5)
        self.assertEqual(gcn.conv1._norm, "both")

        gat = self._build("GAT")
        self.assertIsInstance(gat.conv1, GATConv)
        self.assertEqual(gat.num_heads, 8)
        self.assertEqual(tuple(gat.conv1.fc.weight.shape), (8 * 16, 1433))
        self.assertEqual(tuple(gat.conv2.fc.weight.shape), (7, 8 * 16))
        # DGL's GATConv defaults: no feature or attention dropout.  The canonical
        # GAT and the GIF-style net in lib_gnn_model/gat/gat_net_batch.py use 0.6.
        self.assertEqual(gat.conv1.feat_drop.p, 0.0)
        self.assertEqual(gat.conv1.attn_drop.p, 0.0)
        self.assertEqual(gat.dropout_rate, 0.5)

        sage = self._build("SAGE")
        self.assertIsInstance(sage.conv1, SAGEConv)
        self.assertEqual(sage._aggre_type if hasattr(sage, "_aggre_type")
                         else sage.conv1._aggre_type, "mean")
        self.assertTrue(hasattr(sage.conv1, "fc_self"),
                        "SAGEConv's explicit root term is load bearing here")

    def test_gcn_and_gat_ignore_the_root_node_without_self_loops(self):
        """Documents a measured property of the mapped graph, not a defect.

        The mapped graph is built only from ``sim[(i, j)]`` with ``i != j``, so it
        has no self-loops, and DGL's GraphConv/GATConv add none.  A mapped node's
        own features therefore never reach its own representation under GCN/GAT,
        while SAGEConv's ``fc_self`` keeps them.  This is one plausible
        implementation-level reason the three backbones separate the way they do;
        it is not a proven cause.
        """
        import dgl
        import torch

        graph = dgl.graph(([0], [1]), num_nodes=2)
        features = torch.eye(2)

        from dgl.nn.pytorch import GraphConv, SAGEConv

        conv = GraphConv(2, 2, bias=False, allow_zero_in_degree=True)
        with torch.no_grad():
            conv.weight.copy_(torch.eye(2))
        self.assertTrue(torch.allclose(conv(graph, features)[0], torch.zeros(2)),
                        "GraphConv must not be adding self-loops")

        sage = SAGEConv(2, 2, "mean", bias=False)
        with torch.no_grad():
            sage.fc_self.weight.copy_(torch.eye(2))
            sage.fc_neigh.weight.zero_()
        self.assertTrue(torch.allclose(sage(graph, features), features),
                        "SAGEConv must keep the root node's own features")

    def test_forward_returns_raw_logits(self):
        """The training loops use F.cross_entropy, so no log_softmax may sneak in."""
        import dgl
        import torch

        graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 0]), num_nodes=2))
        features = torch.rand(2, 8)
        for name in ("GCN", "GAT", "SAGE"):
            args = {"target_model": name}
            from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL

            model = NodeClassifierDGL(8, 3, args)
            device = next(model.parameters()).device
            logits = model(features.to(device), graph.to(device), None)
            self.assertEqual(tuple(logits.shape), (2, 3))
            row_sums = logits.exp().sum(dim=1)
            self.assertFalse(
                bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)),
                "{} appears to return log-probabilities, not logits".format(name),
            )


@unittest.skipUnless(HAS_TORCH and HAS_PYG, "needs torch and torch_geometric")
class GIFBackboneFamilyTest(unittest.TestCase):
    """The GIF-style PyG family must also stay as it is, for provenance."""

    def test_gif_style_nets_enable_self_loops(self):
        from lib_gnn_model.gcn.gcn_net_batch import GCNNet as BatchGCNNet

        net = BatchGCNNet(16, 3)
        for conv in net.convs:
            self.assertTrue(conv.add_self_loops)
            self.assertFalse(conv.cached)

    def test_gif_style_gat_keeps_dropout_0_6(self):
        from lib_gnn_model.gat.gat_net_batch import GATNet as BatchGATNet

        net = BatchGATNet(16, 3)
        self.assertAlmostEqual(net.dropout, 0.6)
        self.assertEqual(net.convs[0].heads, 8)
        self.assertTrue(net.convs[0].add_self_loops)


if __name__ == "__main__":
    unittest.main()
