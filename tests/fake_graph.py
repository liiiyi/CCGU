"""A DGL-shaped graph stub, so the community-detection tests need no torch/DGL.

``exp/methods/CCP.py`` and ``exp/methods/partition_diagnostics.py`` only ever ask a
graph for ``num_nodes()``, ``edges()`` and ``ndata['feat'] / ndata['label']``, and
they call ``.numpy()`` on whatever those return.  Reproducing that surface keeps
the CPU test suite installable from numpy/scipy/networkx alone.
"""

import numpy as np


class FakeTensor:
    """Minimal stand-in for a torch tensor: ``.numpy()``, ``.cpu()``, ``.shape``."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array

    def cpu(self):
        return self

    def detach(self):
        return self

    @property
    def shape(self):
        return self._array.shape

    def __len__(self):
        return len(self._array)


class FakeDGLGraph:
    def __init__(self, src, dst, num_nodes, feat=None, label=None, masks=None):
        self._src = np.asarray(src, dtype=np.int64)
        self._dst = np.asarray(dst, dtype=np.int64)
        self._num_nodes = int(num_nodes)
        self.ndata = {}
        if feat is not None:
            self.ndata["feat"] = FakeTensor(np.asarray(feat, dtype=np.float32))
        if label is not None:
            self.ndata["label"] = FakeTensor(np.asarray(label))
        for name, mask in (masks or {}).items():
            self.ndata[name] = FakeTensor(np.asarray(mask, dtype=bool))

    def num_nodes(self):
        return self._num_nodes

    def number_of_nodes(self):
        return self._num_nodes

    def edges(self):
        return FakeTensor(self._src), FakeTensor(self._dst)


def ring_of_cliques(num_cliques=6, clique_size=8, seed=0):
    """Cliques joined in a ring: an unambiguous community structure to test on."""
    rng = np.random.RandomState(seed)
    src, dst = [], []
    num_nodes = num_cliques * clique_size
    for clique in range(num_cliques):
        members = list(range(clique * clique_size, (clique + 1) * clique_size))
        for index, u in enumerate(members):
            for v in members[index + 1:]:
                src += [u, v]
                dst += [v, u]
        bridge_from = members[-1]
        bridge_to = ((clique + 1) % num_cliques) * clique_size
        src += [bridge_from, bridge_to]
        dst += [bridge_to, bridge_from]

    # One informative feature block per clique, so the attribute term agrees with
    # the structure rather than fighting it.
    features = np.zeros((num_nodes, num_cliques), dtype=np.float32)
    for node in range(num_nodes):
        features[node, node // clique_size] = 1.0
    features += 0.01 * rng.rand(num_nodes, num_cliques).astype(np.float32)
    labels = np.arange(num_nodes) // clique_size
    return FakeDGLGraph(src, dst, num_nodes, feat=features, label=labels)


def cliques_with_tail(num_cliques=4, clique_size=12, num_tail_triangles=5, seed=0):
    """Large cliques in a ring plus weakly attached triangles.

    Modularity keeps the triangles as their own 3-node communities (one bridging
    edge cannot pay for the degree-product penalty of absorbing them), so the
    partition has a genuine tail of small communities sitting next to stable large
    ones -- which is the situation ``--ccp_tail_min_size`` exists for.
    """
    rng = np.random.RandomState(seed)
    src, dst = [], []

    def connect(u, v):
        src.extend([u, v])
        dst.extend([v, u])

    core = num_cliques * clique_size
    for clique in range(num_cliques):
        members = list(range(clique * clique_size, (clique + 1) * clique_size))
        for index, u in enumerate(members):
            for v in members[index + 1:]:
                connect(u, v)
        connect(members[-1], ((clique + 1) % num_cliques) * clique_size)

    num_nodes = core + 3 * num_tail_triangles
    for triangle in range(num_tail_triangles):
        base = core + 3 * triangle
        connect(base, base + 1)
        connect(base + 1, base + 2)
        connect(base, base + 2)
        # one bridge into a core clique
        connect(base, (triangle % num_cliques) * clique_size)

    groups = num_cliques + num_tail_triangles
    features = np.zeros((num_nodes, groups), dtype=np.float32)
    for node in range(core):
        features[node, node // clique_size] = 1.0
    for triangle in range(num_tail_triangles):
        for offset in range(3):
            features[core + 3 * triangle + offset, num_cliques + triangle] = 1.0
    features += 0.01 * rng.rand(num_nodes, groups).astype(np.float32)

    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[:core] = np.arange(core) // clique_size
    labels[core:] = num_cliques + np.arange(3 * num_tail_triangles) // 3
    return FakeDGLGraph(src, dst, num_nodes, feat=features, label=labels)
