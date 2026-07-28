"""CCP -- Community-Centric Partition.

Deterministic, overlapping, structure *and* attribute aware community detection
for the CGE mapping stage.

Why this module exists
----------------------
The paper describes the partition stage as a two-level pipeline:

    "CGE introduces the Louvain (Blondel et al. 2008) to initialize communities
     and uses the OSLOM method (Lancichinetti et al. 2011) to optimize community
     structure."                                  -- Community-Centric Mapping

    "For small datasets, Louvain's dataset delineation is used as the initialized
     community and OSLOM is used to optimize the community structure.  For large
     datasets, we evaluate the conductivity of the initialized communities
     divided by Louvain, select the communities with poor conductance for
     fine-grained division [...]"                  -- Appendix D (arXiv)

and it argues that *overlapping* communities are what makes the mapped graph
work ("overlapping community detection better suits CGE").

The only partition method actually reachable from the CLI before this change was
``--partition oslom``, which contains no Louvain stage and is not OSLOM: it is a
label-propagation variant, byte-for-byte identical to ``exp/methods/SLPA.py``
apart from dropping SLPA's cap on communities per node.  See
exp/methods/README.md.

CCP reconstructs the pipeline the paper describes, with a fine-grained stage that
is deterministic instead of stochastic label propagation:

    1. Coarse level.  Louvain modularity maximisation (Eq. 2) on the simple
       undirected projection of the graph, seeded.
    2. Fine level.  Any community larger than ``theta`` is re-partitioned with
       Louvain on its induced subgraph, recursively, until every community is at
       most ``theta`` nodes or refuses to split; unsplittable blocks are chopped
       deterministically in BFS order.  ``theta`` is the target community size.
       Optionally (``conductance_threshold``) only communities whose conductance
       (Eq. 25) exceeds a threshold are refined, which is the paper's
       large-dataset variant.
    3. Overlap level.  A node is additionally attached to up to
       ``max_communities_per_node - 1`` neighbouring communities whose
       *belonging coefficient* clears ``overlap_threshold``.  Belonging is the
       geometric mean of a structural and an attribute term::

           b_s(v, C) = |N(v) inter C| / |N(v)|
           b_a(v, C) = (1 + cos(z_v, mu_C)) / 2
           b(v, C)   = sqrt(b_s(v, C) * b_a(v, C))

       where ``z`` are graph-smoothed node representations -- the parameter-free
       linear encoder of a graph auto-encoder, ``Z = A_hat^L X`` with
       ``A_hat = D~^-1/2 (A + I) D~^-1/2`` -- and ``mu_C`` is the L2-normalised
       centroid of community ``C`` in that space.  Using smoothed rather than raw
       features is what makes the attribute term agree with the community
       structure instead of fighting it, and it is the same signal the mapped
       features (Eq. 5) and mapped labels (Eq. 9) are later built from.
       Contracting communities into super nodes joined by a similarity function
       is the construction of CC-GNN (Li et al., ICDM 2022), which the paper
       cites; CCP supplies the overlapping variant of it that CGE needs.

Community-scale control (post-paper engineering extension, off by default)
-------------------------------------------------------------------------
Nothing below is in the paper.  Every knob defaults to a value that reproduces the
behaviour without it, and none of them may be tuned against test metrics -- see
exp/methods/README.md for starting points and warnings.

    ``resolution`` / ``fine_resolution``  higher normally yields more, smaller
                                          communities (Louvain modularity)
    ``theta``                             target size: blocks above it are split
    ``max_community_size``                hard cap enforced after the overlap
                                          stage, bounding the large tail
    ``tail_min_size``                     opt-in repair of the small tail: blocks
                                          below it merge into the *stable*
                                          neighbour (a block already at or above
                                          ``tail_min_size``) with the highest
                                          label-free attachment score.  Tail
                                          blocks are never targets, so merges
                                          never chain; a tail block with no stable
                                          neighbour is kept unchanged

The two tails are controlled independently and the pipeline order is fixed:
coarse Louvain -> fine refinement (``theta``) -> overlap -> singleton backfill ->
size cap (``max_community_size``) -> tail repair (``tail_min_size``).  The backfill
runs *before* the two scale stages so that protected evaluation singletons and
isolated nodes are real blocks there and can be exempted explicitly rather than by
accident.  Tail repair runs after the cap, so a merge can lift a block back above
the cap; that is logged as a warning and visible in ``scale_report``.

Determinism
-----------
Every step is a deterministic function of ``(graph, features, seed)``:
``networkx.louvain_communities`` is called with an explicit seed, the recursive
refinement iterates communities in sorted id order, tail-repair decisions are
computed against a frozen pre-repair state in ``(size, lowest member)`` order, and
ties in both the overlap ranking and the tail-repair ranking break on
``(-score, community id)``.  Two runs with the same seed produce byte-identical
``c2n``; different seeds do not, and that variation is real -- see
exp/methods/README.md.

Complexity
----------
``O(E log N)`` for Louvain, ``O(L * nnz(X))`` for the propagation and
``O(sum_v deg(v) * k)`` for the overlap stage.  Contrast the legacy
label-propagation implementation, whose per-node label memory grows by one entry
per iteration and therefore costs ``O(E * T^2)`` label copies for ``T``
iterations.
"""

import logging
import time
from collections import Counter, defaultdict, deque

import numpy as np
import scipy.sparse as sp

try:  # pragma: no cover - exercised implicitly by the CLI
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:  # pragma: no cover
    nx = None
    louvain_communities = None


def _edges_from_graph(graph):
    """Return ``(src, dst)`` int64 arrays for a DGL graph or a networkx graph."""
    if hasattr(graph, "edges") and hasattr(graph, "num_nodes"):
        src, dst = graph.edges()
        if hasattr(src, "cpu"):
            src, dst = src.cpu(), dst.cpu()
        return (
            np.asarray(src.numpy(), dtype=np.int64),
            np.asarray(dst.numpy(), dtype=np.int64),
        )
    if nx is not None and isinstance(graph, nx.Graph):
        if graph.number_of_edges() == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        edge_array = np.asarray(list(graph.edges()), dtype=np.int64)
        return edge_array[:, 0], edge_array[:, 1]
    raise TypeError("CCP needs a DGL graph or a networkx graph, got %r" % type(graph))


def _num_nodes_of(graph):
    if hasattr(graph, "num_nodes"):
        return int(graph.num_nodes())
    return int(graph.number_of_nodes())


def _features_of(graph):
    """Node features as a float32 numpy array, or ``None`` when unavailable."""
    data = getattr(graph, "ndata", None)
    if not data or "feat" not in data:
        return None
    features = data["feat"]
    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "cpu"):
        features = features.cpu()
    return np.asarray(features.numpy(), dtype=np.float32)


def build_symmetric_adjacency(src, dst, num_nodes):
    """Simple undirected adjacency: symmetrised, de-duplicated, no self-loops."""
    keep = src != dst
    src, dst = src[keep], dst[keep]
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    adjacency = sp.coo_matrix(
        (np.ones(both_src.size, dtype=np.float32), (both_src, both_dst)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency.data[:] = 1.0  # collapse multi-edges
    adjacency.eliminate_zeros()
    return adjacency


def propagate_features(adjacency, features, steps):
    """``A_hat^steps X`` with ``A_hat = D~^-1/2 (A + I) D~^-1/2`` (GAE/SGC encoder)."""
    num_nodes = adjacency.shape[0]
    augmented = adjacency + sp.eye(num_nodes, dtype=np.float32, format="csr")
    degrees = np.asarray(augmented.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float32)
    positive = degrees > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    scaling = sp.diags(inverse_sqrt)
    normalised = (scaling @ augmented @ scaling).tocsr()

    smoothed = features
    for _ in range(max(0, int(steps))):
        smoothed = normalised @ smoothed
    return np.asarray(smoothed, dtype=np.float32)


def _l2_normalise(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _bfs_chunks(adjacency, nodes, chunk_size):
    """Split ``nodes`` into BFS-contiguous chunks of at most ``chunk_size``."""
    node_set = set(int(node) for node in nodes)
    order = []
    remaining = sorted(node_set)
    visited = set()
    for start in remaining:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            order.append(current)
            neighbours = adjacency.indices[
                adjacency.indptr[current] : adjacency.indptr[current + 1]
            ]
            for neighbour in sorted(int(n) for n in neighbours):
                if neighbour in node_set and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
    return [order[i : i + chunk_size] for i in range(0, len(order), chunk_size)]


def community_conductance(adjacency, nodes):
    """Conductance of a node set (Eq. 25); 0.0 for an empty-volume set."""
    nodes = np.asarray(sorted(int(node) for node in nodes), dtype=np.int64)
    if nodes.size == 0:
        return 0.0
    membership = np.zeros(adjacency.shape[0], dtype=bool)
    membership[nodes] = True
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    volume = float(degrees[nodes].sum())
    total_volume = float(degrees.sum())
    if volume <= 0.0 or total_volume - volume <= 0.0:
        return 0.0
    submatrix = adjacency[nodes][:, nodes]
    internal = float(submatrix.sum())
    cut = volume - internal
    return cut / min(volume, total_volume - volume)


class CommunityCentricPartition:
    """Louvain-initialised, deterministically refined, overlapping partition."""

    def __init__(
        self,
        graph,
        args=None,
        seed=0,
        resolution=1.0,
        fine_resolution=1.0,
        theta=20,
        max_communities_per_node=3,
        overlap_threshold=0.35,
        propagation_steps=2,
        conductance_threshold=None,
        max_refine_depth=8,
        protect_eval_nodes="none",
        test_ratio=0.2,
        max_community_size=0,
        tail_min_size=0,
    ):
        self.logger = logging.getLogger("CCP")
        if louvain_communities is None:  # pragma: no cover
            raise ImportError("CCP requires networkx>=2.8 for louvain_communities")

        args = args or {}
        self.graph = graph
        self.seed = int(args.get("random_seed", seed))
        self.resolution = float(args.get("ccp_resolution", resolution))
        self.fine_resolution = float(args.get("ccp_fine_resolution", fine_resolution))
        self.theta = int(args.get("ccp_theta", theta))
        self.max_communities_per_node = int(
            args.get("ccp_max_communities_per_node", max_communities_per_node)
        )
        self.overlap_threshold = float(
            args.get("ccp_overlap_threshold", overlap_threshold)
        )
        self.propagation_steps = int(
            args.get("ccp_propagation_steps", propagation_steps)
        )
        raw_conductance = args.get("ccp_conductance_threshold", conductance_threshold)
        # A negative CLI value means "refine every oversized community".
        if raw_conductance is not None and float(raw_conductance) < 0:
            raw_conductance = None
        self.conductance_threshold = (
            None if raw_conductance is None else float(raw_conductance)
        )
        self.max_refine_depth = int(max_refine_depth)
        self.protect_eval_nodes = str(
            args.get("ccp_protect_eval_nodes", protect_eval_nodes)
        )
        self.test_ratio = float(args.get("test_ratio", test_ratio))
        # Community-scale controls (post-paper engineering extension; 0 = disabled,
        # which is the default and reproduces the behaviour without them).
        self.max_community_size = int(
            args.get("ccp_max_community_size", max_community_size) or 0
        )
        self.tail_min_size = int(args.get("ccp_tail_min_size", tail_min_size) or 0)
        if self.max_community_size and self.max_community_size < self.theta:
            raise ValueError(
                "--ccp_max_community_size ({}) must be >= --ccp_theta ({}); the cap "
                "is applied after overlap expansion, so a value below the split "
                "threshold would immediately re-split every block".format(
                    self.max_community_size, self.theta
                )
            )
        if self.tail_min_size and self.max_community_size and (
            self.tail_min_size > self.max_community_size
        ):
            raise ValueError(
                "--ccp_tail_min_size ({}) must be <= --ccp_max_community_size ({})"
                .format(self.tail_min_size, self.max_community_size)
            )
        self.effective_config = {
            "ccp_resolution": self.resolution,
            "ccp_fine_resolution": self.fine_resolution,
            "ccp_theta": self.theta,
            "ccp_max_community_size": self.max_community_size,
            "ccp_tail_min_size": self.tail_min_size,
            "ccp_max_communities_per_node": self.max_communities_per_node,
            "ccp_overlap_threshold": self.overlap_threshold,
            "ccp_propagation_steps": self.propagation_steps,
            "ccp_conductance_threshold": self.conductance_threshold,
            "ccp_protect_eval_nodes": self.protect_eval_nodes,
            "random_seed": self.seed,
        }

        self._cuda_index = int(args.get("cuda", 0) or 0)
        self.num_nodes = _num_nodes_of(graph)
        src, dst = _edges_from_graph(graph)
        self.edges = (src, dst)
        self.adjacency = build_symmetric_adjacency(src, dst, self.num_nodes)
        self.protected = self._protected_mask()

    def _protected_mask(self):
        """Original nodes that must stay in singleton communities, or ``None``.

        ``exp/exp_train.py::calculate_tvt_split_nucleus`` promotes every singleton
        community whose one original node is held out into the mapped test set.
        That branch is the codebase's own mechanism for evaluating the mapped model
        on real held-out original nodes, and it only fires if the partition leaves
        those nodes unaggregated.  Opt in with ``--ccp_protect_eval_nodes``; the
        default 'none' keeps the historical behaviour of partitioning every node.
        """
        mode = self.protect_eval_nodes
        if mode in ("none", "", "False", "false"):
            return None
        if mode not in ("test", "test_val"):
            raise ValueError(
                "--ccp_protect_eval_nodes must be one of none, test, test_val "
                "(got {!r})".format(mode)
            )
        from lib_utils.splits import original_holdout_mask

        mask = original_holdout_mask(
            self.graph,
            self.num_nodes,
            self.test_ratio,
            self.seed,
            include_val=(mode == "test_val"),
        )
        self.logger.info(
            "CCP will keep %d held-out original node(s) as singleton communities "
            "(--ccp_protect_eval_nodes %s)",
            int(mask.sum()), mode,
        )
        return mask

    # ------------------------------------------------------------------ stages
    #: Human-readable name of the base (pre-refinement) stage, for the run log.
    #: Subclasses that replace the base stage override it -- see GAEPartition.
    base_stage_label = "coarse Louvain"

    def _ensure_nx_graph(self):
        """Build (once) the simple undirected networkx view the later stages use."""
        existing = getattr(self, "_nx_graph", None)
        if existing is not None:
            return existing
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(self.num_nodes))
        coo = sp.triu(self.adjacency, k=1).tocoo()
        nx_graph.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))
        self._nx_graph = nx_graph
        return nx_graph

    def _coarse_partition(self):
        nx_graph = self._ensure_nx_graph()

        if self.protected is None:
            target = nx_graph
        else:
            # Held-out nodes are excluded from modularity optimisation entirely, so
            # they cannot end up inside an aggregated community.
            target = nx_graph.subgraph(
                [node for node in range(self.num_nodes) if not self.protected[node]]
            )
        communities = louvain_communities(
            target,
            resolution=self.resolution,
            seed=self.seed,
        )
        return [sorted(int(node) for node in community) for community in communities]

    def _refine(self, community, depth):
        """Recursively split an oversized community; returns a list of blocks."""
        if len(community) <= self.theta or depth >= self.max_refine_depth:
            return [community]
        if self.conductance_threshold is not None:
            conductance = community_conductance(self.adjacency, community)
            if conductance <= self.conductance_threshold:
                return [community]

        subgraph = self._nx_graph.subgraph(community)
        if subgraph.number_of_edges() == 0:
            # Modularity is undefined on an edgeless subgraph (networkx divides by
            # the squared degree sum).  A Louvain community always has edges, but a
            # block produced by an embedding-space clustering -- see
            # exp/methods/GAEPartition.py -- need not be connected at all.
            return _bfs_chunks(self.adjacency, community, self.theta)
        parts = louvain_communities(
            subgraph,
            resolution=self.fine_resolution,
            seed=self.seed + depth,
        )
        parts = [sorted(int(node) for node in part) for part in parts if part]
        if len(parts) <= 1 or max(len(part) for part in parts) == len(community):
            # Louvain refuses to split this block (e.g. a clique): chop it
            # deterministically so the mapped graph does not keep a giant node.
            return _bfs_chunks(self.adjacency, community, self.theta)

        refined = []
        for part in sorted(parts, key=lambda block: (block[0], len(block))):
            refined.extend(self._refine(part, depth + 1))
        return refined

    def _fine_partition(self, coarse):
        fine = []
        for community in sorted(coarse, key=lambda block: (block[0], len(block))):
            fine.extend(self._refine(community, 0))
        return [block for block in fine if block]

    def _representations(self):
        features = _features_of(self.graph)
        if features is None or features.size == 0:
            return None
        smoothed = propagate_features(
            self.adjacency, features, self.propagation_steps
        )
        return _l2_normalise(smoothed)

    def _expand_overlap(self, blocks, representations):
        """Attach nodes to neighbouring communities with high belonging."""
        extra_slots = self.max_communities_per_node - 1
        memberships = [list(block) for block in blocks]
        if extra_slots <= 0 or not blocks:
            return memberships

        primary = np.full(self.num_nodes, -1, dtype=np.int64)
        for index, block in enumerate(blocks):
            primary[np.asarray(block, dtype=np.int64)] = index

        centroids = None
        if representations is not None:
            centroids = np.zeros(
                (len(blocks), representations.shape[1]), dtype=np.float32
            )
            for index, block in enumerate(blocks):
                centroids[index] = representations[
                    np.asarray(block, dtype=np.int64)
                ].mean(axis=0)
            centroids = _l2_normalise(centroids)

        indptr, indices = self.adjacency.indptr, self.adjacency.indices
        attachments = defaultdict(list)
        for node in range(self.num_nodes):
            if self.protected is not None and self.protected[node]:
                continue
            neighbours = indices[indptr[node] : indptr[node + 1]]
            degree = neighbours.size
            if degree == 0:
                continue
            own = primary[node]
            neighbour_communities = primary[neighbours]
            candidates, counts = np.unique(
                neighbour_communities[neighbour_communities >= 0],
                return_counts=True,
            )
            keep = candidates != own
            candidates, counts = candidates[keep], counts[keep]
            if candidates.size == 0:
                continue

            structural = counts / float(degree)
            if centroids is None:
                attribute = np.ones_like(structural, dtype=np.float32)
            else:
                # One gemv per node rather than one dot per candidate.
                cosine = centroids[candidates] @ representations[node]
                attribute = 0.5 * (1.0 + cosine)
            belonging = np.sqrt(
                np.clip(structural, 0.0, None) * np.clip(attribute, 0.0, None)
            )
            above = np.nonzero(belonging >= self.overlap_threshold)[0]
            if above.size == 0:
                continue
            # Ties break on the lower community id, so the result is seed-stable.
            order = above[
                np.lexsort((candidates[above], -belonging[above]))
            ]
            for index in order[:extra_slots]:
                attachments[int(candidates[index])].append(node)

        for candidate, nodes in attachments.items():
            memberships[candidate].extend(nodes)
        return [sorted(set(block)) for block in memberships]

    def _enforce_max_size(self, blocks):
        """Split any block above ``max_community_size`` into BFS-contiguous chunks.

        The overlap stage can grow a block well past ``theta`` (on Cora it reaches
        ~120 nodes from a ``theta`` of 20), and a very large mapped node averages
        away the feature and label signal the mapping is supposed to preserve.
        Disabled by default.
        """
        cap = self.max_community_size
        stats = {"enabled": bool(cap), "cap": cap, "blocks_split": 0, "blocks_added": 0}
        if not cap:
            return blocks, stats

        capped = []
        for block in blocks:
            if len(block) <= cap:
                capped.append(block)
                continue
            # Protected singletons can never exceed the cap, so nothing here can
            # split an intentional evaluation community.
            chunks = _bfs_chunks(self.adjacency, block, cap)
            stats["blocks_split"] += 1
            stats["blocks_added"] += len(chunks) - 1
            capped.extend(chunks)
        self.logger.info(
            "CCP size cap %d: split %d block(s) into %d extra block(s)",
            cap, stats["blocks_split"], stats["blocks_added"],
        )
        return capped, stats

    def _repair_tail(self, blocks, representations):
        """Reassign detector-generated tail communities into a stable neighbour.

        Policy, stated exactly:

        * A **tail** block is a non-protected block with fewer than
          ``tail_min_size`` nodes.
        * A **stable target** is a non-protected block with at least
          ``tail_min_size`` nodes.  Tail blocks are never targets, so a merge can
          never chain and the outcome does not depend on the order in which the
          merges are applied.
        * Every score is computed against the **frozen pre-repair state**: the
          structural term counts edges from the tail block's nodes into the target
          block's nodes, and the attribute term compares their pre-repair
          centroids.  Nothing is recomputed as merges accumulate.
        * A tail block with no stable-target neighbour is **kept unchanged** --
          including a fully isolated one, which by construction has no neighbour
          at all.  If the partition contains no stable target whatsoever the
          repair is a logged no-op rather than an arbitrary rearrangement.
        * Protected evaluation singletons are exempt as a source (never merged
          away, or the hold-out protocol loses its evaluation nodes) and as a
          target (never grown, or they stop being singletons).

        So every tail block ends up in exactly one of two buckets, ``merged`` or
        ``kept_no_target``, and they sum to ``tail_before``.

        Label free by construction: the decision reads only the observed adjacency
        and the propagated-feature centroids, never node labels, so it cannot leak
        supervision into the mapping.
        """
        min_size = self.tail_min_size
        stats = {
            "enabled": bool(min_size and min_size > 1),
            "min_size": min_size,
            "tail_before": 0,
            "stable_targets": 0,
            "merged": 0,
            "kept_no_target": 0,
            "kept_protected": 0,
            "nodes_reassigned": 0,
            "memberships_deduplicated": 0,
        }
        if not stats["enabled"] or not blocks:
            return blocks, stats

        sizes = [len(block) for block in blocks]
        protected_blocks = set()
        if self.protected is not None:
            for index, block in enumerate(blocks):
                if len(block) == 1 and self.protected[block[0]]:
                    protected_blocks.add(index)
        stats["kept_protected"] = len(protected_blocks)

        tail = [
            index
            for index, size in enumerate(sizes)
            if size < min_size and index not in protected_blocks
        ]
        stable_targets = {
            index
            for index, size in enumerate(sizes)
            if size >= min_size and index not in protected_blocks
        }
        stats["tail_before"] = len(tail)
        stats["stable_targets"] = len(stable_targets)
        if not tail:
            return blocks, stats
        if not stable_targets:
            # Honest no-op: there is no stable community to reassign into, so the
            # partition is left exactly as the detector produced it.
            stats["kept_no_target"] = len(tail)
            self.logger.info(
                "CCP tail repair (min size %d): %d tail block(s) but no stable "
                "target of that size exists, so nothing was merged.  Lower "
                "--ccp_tail_min_size or raise --ccp_theta to get a coarser base.",
                min_size, len(tail),
            )
            return blocks, stats

        membership = defaultdict(list)
        for index, block in enumerate(blocks):
            for node in block:
                membership[node].append(index)

        centroids = None
        if representations is not None:
            centroids = np.zeros(
                (len(blocks), representations.shape[1]), dtype=np.float32
            )
            for index, block in enumerate(blocks):
                centroids[index] = representations[
                    np.asarray(block, dtype=np.int64)
                ].mean(axis=0)
            centroids = _l2_normalise(centroids)

        indptr, indices = self.adjacency.indptr, self.adjacency.indices
        # Decisions are read off the frozen state; the order below only fixes the
        # tie-break, it cannot change any score.
        decisions = {}
        for index in sorted(tail, key=lambda i: (sizes[i], blocks[i][0])):
            counts = Counter()
            for node in blocks[index]:
                for neighbour in indices[indptr[node] : indptr[node + 1]]:
                    for candidate in membership[int(neighbour)]:
                        if candidate in stable_targets:
                            counts[candidate] += 1
            if not counts:
                stats["kept_no_target"] += 1
                continue
            total = float(sum(counts.values()))
            scored = []
            for candidate, count in counts.items():
                structural = count / total
                if centroids is None:
                    attribute = 1.0
                else:
                    attribute = 0.5 * (
                        1.0 + float(np.dot(centroids[index], centroids[candidate]))
                    )
                score = float(np.sqrt(max(structural, 0.0) * max(attribute, 0.0)))
                scored.append((-score, candidate))
            scored.sort()
            decisions[index] = scored[0][1]

        absorbed = defaultdict(list)
        for index, target in sorted(decisions.items()):
            absorbed[target].append(index)
        stats["merged"] = len(decisions)

        merged_away = set(decisions)
        repaired = []
        for index, block in enumerate(blocks):
            if index in merged_away:
                continue
            if index not in absorbed:
                repaired.append(block)
                continue
            union = set(block)
            raw_total = len(block)
            for source in absorbed[index]:
                union.update(blocks[source])
                raw_total += len(blocks[source])
                stats["nodes_reassigned"] += len(blocks[source])
            stats["memberships_deduplicated"] += raw_total - len(union)
            repaired.append(sorted(union))
        if self.max_community_size:
            oversized = sum(
                1 for block in repaired if len(block) > self.max_community_size
            )
            if oversized:
                self.logger.warning(
                    "CCP tail repair pushed %d block(s) above "
                    "--ccp_max_community_size %d; the cap is applied before repair",
                    oversized, self.max_community_size,
                )
        self.logger.info(
            "CCP tail repair (min size %d): %d tail block(s) against %d stable "
            "target(s); %d merged, %d kept for lack of a stable neighbour, "
            "%d protected singleton(s) left alone",
            min_size, stats["tail_before"], stats["stable_targets"],
            stats["merged"], stats["kept_no_target"], stats["kept_protected"],
        )
        return repaired, stats

    @staticmethod
    def _size_diagnostics(blocks):
        sizes = np.asarray([len(block) for block in blocks] or [0], dtype=np.int64)
        return {
            "num_communities": len(blocks),
            "size_min": int(sizes.min()),
            "size_median": float(np.median(sizes)),
            "size_mean": float(sizes.mean()),
            "size_max": int(sizes.max()),
            "singletons": int(np.count_nonzero(sizes == 1)),
        }

    # -------------------------------------------------------------------- API
    def partition(self):
        """Return ``(n2c, c2n, num_communities, elapsed_seconds)``."""
        start_time = time.time()

        coarse = self._coarse_partition()
        self.logger.info(
            "%s base stage (seed %d): %d communities",
            self.base_stage_label,
            self.seed,
            len(coarse),
        )

        blocks = self._fine_partition(coarse)
        self.logger.info(
            "CCP fine refinement (theta %d, resolution %.3f): %d communities",
            self.theta,
            self.fine_resolution,
            len(blocks),
        )

        representations = self._representations()
        if representations is None:
            self.logger.info(
                "CCP: graph has no node features, using structural belonging only"
            )
        blocks = self._expand_overlap(blocks, representations)

        # Held-out and isolated nodes belong to no aggregated community but must
        # still be represented, otherwise the mapping silently drops them.  This
        # runs before the scale stages so that protected evaluation singletons are
        # real blocks there and can be exempted explicitly rather than by accident.
        covered = set()
        for block in blocks:
            covered.update(block)
        singletons = 0
        for node in range(self.num_nodes):
            if node not in covered:
                blocks.append([node])
                singletons += 1
        if singletons:
            self.logger.info("CCP added %d singleton community(ies)", singletons)

        self.scale_report = {
            "after_overlap": self._size_diagnostics(blocks),
            "effective_config": dict(self.effective_config),
        }

        blocks, cap_stats = self._enforce_max_size(blocks)
        self.scale_report["size_cap"] = cap_stats
        self.scale_report["after_size_cap"] = self._size_diagnostics(blocks)

        blocks, tail_stats = self._repair_tail(blocks, representations)
        self.scale_report["tail_repair"] = tail_stats
        self.scale_report["after_tail_repair"] = self._size_diagnostics(blocks)

        c2n = {}
        n2c = defaultdict(list)
        for community, block in enumerate(blocks):
            c2n[community] = list(block)
            for node in block:
                n2c[int(node)].append(community)

        elapsed = time.time() - start_time
        self.scale_report["final"] = self._size_diagnostics(list(c2n.values()))
        self.logger.info(
            "CCP produced %d communities over %d nodes in %.2f s (config %s)",
            len(c2n),
            self.num_nodes,
            elapsed,
            self.effective_config,
        )
        return dict(n2c), c2n, len(c2n), elapsed
