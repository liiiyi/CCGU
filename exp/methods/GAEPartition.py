"""GAE -- a label-free graph-auto-encoder community detector.

**Post-paper extension.**  The paper's partition stage is Louvain + OSLOM and it
uses a graph auto-encoder only to compute the Information Retention *metric*
(Appendix C).  Its conclusion nonetheless names this direction explicitly --
"Future work will leverage deep learning-based methods to enhance data insights"
-- and its ablation invites substitution: "its great potential supports the
integration of any more effective community detection method or graph mapping
algorithm."  This module is that substitution, offered as an additional
``--partition gae`` choice.  No legacy method and no default changes.

Pipeline
--------
1. **Encode** (structure + features).  A two-layer DGL ``GraphConv`` encoder maps
   ``X`` to a latent ``Z`` on the self-looped simple undirected graph.
2. **Train** a link-reconstruction objective: inner-product decoder,
   ``BCEWithLogits`` over the observed edges plus an equal number of *true*
   non-edges.  Negatives are drawn by deterministic vectorised rejection sampling
   against the adjacency, so a sampled pair is never a self-pair and never an
   existing edge.  No labels are read anywhere and no validation or test metric is
   consulted; the only objective is reconstructing the graph the detector was
   handed.
3. **Cluster** ``Z`` with k-means, ``k = ceil(n_clusterable / ccp_theta)`` -- the
   same target-size knob the rest of the pipeline uses.
4. **Reuse** CCP's overlap assignment, size cap, tail repair, protected-singleton
   handling, diagnostics and id compaction by subclassing
   ``CommunityCentricPartition`` and overriding only the base-block stage and the
   representation.  ``_representations`` returns the *learned* embedding, so the
   deep representation drives the overlap and tail-repair scores too, not just the
   initial clustering.  ``n2c`` / ``c2n`` therefore keep exactly the same shape and
   guarantees as CCP.

Device, and what "GPU" does and does not buy
--------------------------------------------
The encoder's forward and backward run on CUDA when available (``--gae_device
auto``, override with ``cuda`` / ``cpu``).  k-means, the overlap stage and the
tail repair stay on the CPU, so the end-to-end partition speedup is partial -- see
exp/methods/README.md.  No claim is made that the legacy
Louvain/Infomap/OSLOM detectors are GPU accelerated: at audit time no compatible
RAPIDS/cuGraph backend was installable in this pinned Python 3.8 environment, so
they run on the CPU.

Determinism, measured
---------------------
DGL's sparse message passing sums contributions in a non-associative order that
depends on the thread schedule, which was measured on this machine as:

    CPU, 1 thread   bitwise identical across runs
    CPU, 8 threads  embeddings differ by up to 2.7e-07
    CUDA            embeddings differ by up to 4.8e-07

A 1e-07 wobble is irrelevant to the embedding but *not* to k-means, which can flip
a boundary node and cascade into a visibly different partition.  Two measures
address it: the encoder is trained with ``torch.set_num_threads(1)`` (restored
afterwards), and the embedding is L2-normalised and rounded to
``--gae_round_decimals`` places before clustering.  What that buys, measured on
Cora with ``--ccp_protect_eval_nodes test``:

    same device, repeated runs   bitwise identical -- CPU and CUDA, seeds 4 and 5
    CPU vs CUDA, same seed       agrees on seed 4; **disagrees on seed 5**
                                 (1,512 vs 1,535 communities)

So the partition is reproducible if you keep the device fixed, and the device is
part of the configuration -- it is recorded in ``gae_report['device']`` and in the
run id.  Rounding narrows the gap but cannot close it: use ``--gae_device cpu``
when exact cross-machine reproducibility matters more than the encoder speedup.
No stronger claim is made.
"""

import logging
import math
import time

import numpy as np

from exp.methods.CCP import CommunityCentricPartition, _features_of, _l2_normalise


class GAEPartition(CommunityCentricPartition):
    """Louvain replaced by a trained graph auto-encoder; everything else shared."""

    base_stage_label = "GAE encoder + k-means"

    def __init__(
        self,
        graph,
        args=None,
        hidden_dim=64,
        latent_dim=16,
        epochs=100,
        learning_rate=0.01,
        device="auto",
        round_decimals=6,
        **kwargs
    ):
        super(GAEPartition, self).__init__(graph, args=args, **kwargs)
        args = args or {}
        self.logger = logging.getLogger("GAE")
        self.hidden_dim = int(args.get("gae_hidden_dim", hidden_dim))
        self.latent_dim = int(args.get("gae_latent_dim", latent_dim))
        self.epochs = int(args.get("gae_epochs", epochs))
        self.learning_rate = float(args.get("gae_learning_rate", learning_rate))
        self.device_choice = str(args.get("gae_device", device))
        self.round_decimals = int(args.get("gae_round_decimals", round_decimals))
        self._validate()

        self.effective_config.update(
            {
                "gae_hidden_dim": self.hidden_dim,
                "gae_latent_dim": self.latent_dim,
                "gae_epochs": self.epochs,
                "gae_learning_rate": self.learning_rate,
                "gae_device": self.device_choice,
                "gae_round_decimals": self.round_decimals,
            }
        )
        self.gae_report = {}
        self.embedding = None

    def _validate(self):
        if self.hidden_dim < 1:
            raise ValueError("--gae_hidden_dim must be >= 1")
        if self.latent_dim < 1:
            raise ValueError("--gae_latent_dim must be >= 1")
        if self.epochs < 1:
            raise ValueError("--gae_epochs must be >= 1")
        if not self.learning_rate > 0:
            raise ValueError("--gae_learning_rate must be > 0")
        if self.device_choice not in ("auto", "cuda", "cpu"):
            raise ValueError(
                "--gae_device must be one of auto, cuda, cpu (got {!r})".format(
                    self.device_choice
                )
            )
        if self.round_decimals < 0:
            raise ValueError("--gae_round_decimals must be >= 0")

    # ------------------------------------------------------------------ device
    def _resolve_device(self):
        import torch

        if self.device_choice == "cpu":
            return torch.device("cpu")
        if self.device_choice == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "--gae_device cuda was requested but torch.cuda.is_available() "
                    "is False; use --gae_device auto for a CPU fallback"
                )
            return torch.device("cuda:%d" % self._cuda_index)
        if torch.cuda.is_available():
            return torch.device("cuda:%d" % self._cuda_index)
        self.logger.info(
            "CUDA is unavailable, falling back to the CPU for the GAE encoder"
        )
        return torch.device("cpu")

    # --------------------------------------------------------- negative sampling
    def _edge_codes(self):
        """Sorted ``u * N + v`` codes of every directed edge, for membership tests."""
        coo = self.adjacency.tocoo()
        codes = coo.row.astype(np.int64) * self.num_nodes + coo.col.astype(np.int64)
        return np.sort(codes)

    def _sample_true_negatives(self, count, generator, edge_codes):
        """``count`` node pairs that are neither self-pairs nor existing edges.

        Deterministic given ``generator``: candidates are drawn in a fixed order and
        rejected by a vectorised membership test, so the result depends only on the
        seed.  Oversamples each round to keep the number of rounds small; the round
        cap is a safety net for pathologically dense graphs, and a shortfall is
        reported rather than silently padded with invalid pairs.
        """
        import torch

        if count <= 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, 0

        collected_u, collected_v = [], []
        remaining = count
        rounds = 0
        max_rounds = 32
        while remaining > 0 and rounds < max_rounds:
            rounds += 1
            draw = int(min(max(remaining * 2, 1024), 4_000_000))
            source = torch.randint(
                0, self.num_nodes, (draw,), generator=generator
            ).numpy().astype(np.int64)
            target = torch.randint(
                0, self.num_nodes, (draw,), generator=generator
            ).numpy().astype(np.int64)

            keep = source != target
            source, target = source[keep], target[keep]
            codes = source * self.num_nodes + target
            position = np.searchsorted(edge_codes, codes)
            position = np.minimum(position, edge_codes.size - 1)
            is_edge = edge_codes[position] == codes
            source, target = source[~is_edge], target[~is_edge]

            if source.size > remaining:
                source, target = source[:remaining], target[:remaining]
            collected_u.append(source)
            collected_v.append(target)
            remaining -= source.size

        negative_src = np.concatenate(collected_u) if collected_u else np.zeros(0, np.int64)
        negative_dst = np.concatenate(collected_v) if collected_v else np.zeros(0, np.int64)
        return negative_src, negative_dst, rounds

    # ------------------------------------------------------------------ encoder
    def _train_encoder(self):
        """Train the encoder and return the latent embedding as a numpy array."""
        import dgl
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from dgl.nn.pytorch import GraphConv

        features = _features_of(self.graph)
        feature_source = "node features"
        if features is None or features.size == 0:
            # Structure-only fallback keeps the interface working on featureless
            # graphs (e.g. the RDF datasets).
            degrees = np.asarray(self.adjacency.sum(axis=1)).reshape(-1, 1)
            features = np.log1p(degrees).astype(np.float32)
            feature_source = "log-degree (graph has no node features)"
            self.logger.info("graph has no node features, encoding log-degree only")

        device = self._resolve_device()

        # Single-threaded training makes the CPU path bitwise reproducible; see the
        # module docstring for the measured numbers.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(self.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)

            coo = self.adjacency.tocoo()
            graph = dgl.graph(
                (coo.row.astype(np.int64), coo.col.astype(np.int64)),
                num_nodes=self.num_nodes,
            )
            graph = dgl.add_self_loop(graph).to(device)
            node_features = torch.tensor(features, dtype=torch.float32, device=device)

            class Encoder(nn.Module):
                def __init__(self, in_dim, hidden, latent):
                    super(Encoder, self).__init__()
                    self.conv1 = GraphConv(in_dim, hidden)
                    self.conv2 = GraphConv(hidden, latent)

                def forward(self, block, inputs):
                    hidden = functional.relu(self.conv1(block, inputs))
                    return self.conv2(block, hidden)

            encoder = Encoder(
                node_features.shape[1], self.hidden_dim, self.latent_dim
            ).to(device)
            optimizer = torch.optim.Adam(
                encoder.parameters(), lr=self.learning_rate
            )

            # Positive edges: the upper triangle of the simple undirected graph.
            upper = self.adjacency.tocoo()
            keep = upper.row < upper.col
            positive_src = torch.tensor(
                upper.row[keep].astype(np.int64), device=device
            )
            positive_dst = torch.tensor(
                upper.col[keep].astype(np.int64), device=device
            )
            num_positive = int(positive_src.numel())
            if num_positive == 0:
                self.logger.info("graph has no edges, skipping GAE training")
                with torch.no_grad():
                    latent = encoder(graph, node_features)
                self.gae_report = {
                    "device": str(device),
                    "feature_source": feature_source,
                    "epochs": 0,
                    "num_positive_edges": 0,
                }
                return latent.cpu().numpy()

            edge_codes = self._edge_codes()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)

            started = time.time()
            losses = []
            sampling_rounds = 0
            negative_shortfall = 0
            for epoch in range(self.epochs):
                encoder.train()
                latent = encoder(graph, node_features)
                negative_u, negative_v, rounds = self._sample_true_negatives(
                    num_positive, generator, edge_codes
                )
                sampling_rounds += rounds
                negative_shortfall += num_positive - negative_u.size
                negative_src = torch.tensor(negative_u, device=device)
                negative_dst = torch.tensor(negative_v, device=device)

                positive_score = (
                    latent[positive_src] * latent[positive_dst]
                ).sum(dim=1)
                negative_score = (
                    latent[negative_src] * latent[negative_dst]
                ).sum(dim=1)
                scores = torch.cat([positive_score, negative_score])
                targets = torch.cat(
                    [
                        torch.ones_like(positive_score),
                        torch.zeros_like(negative_score),
                    ]
                )
                loss = functional.binary_cross_entropy_with_logits(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                if (epoch + 1) % 25 == 0:
                    self.logger.info(
                        "GAE epoch %d/%d, reconstruction loss %.6f",
                        epoch + 1, self.epochs, losses[-1],
                    )

            encoder.eval()
            with torch.no_grad():
                latent = encoder(graph, node_features)
        finally:
            torch.set_num_threads(previous_threads)

        self.gae_report = {
            "device": str(device),
            "feature_source": feature_source,
            "epochs": self.epochs,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_positive_edges": num_positive,
            "negative_sampling_rounds": sampling_rounds,
            "negative_shortfall": int(negative_shortfall),
            "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "encoder_seconds": time.time() - started,
        }
        if negative_shortfall:
            self.logger.warning(
                "GAE negative sampling fell short by %d pair(s) in total; the graph "
                "may be too dense for rejection sampling",
                negative_shortfall,
            )
        self.logger.info(
            "GAE encoder trained on %s in %.2f s (loss %.6f -> %.6f)",
            device, self.gae_report["encoder_seconds"],
            self.gae_report["first_loss"], self.gae_report["final_loss"],
        )
        return latent.cpu().numpy()

    # ---------------------------------------------------------- representations
    def _representations(self):
        """The learned embedding, L2-normalised -- used by overlap and tail repair.

        Overriding this is what makes the deep representation drive *every*
        downstream assignment.  Without it the parent's propagated raw features
        would score the overlap and the tail repair, and the encoder would only
        have influenced the initial clustering.
        """
        if self.embedding is None:
            self.embedding = self._quantised_embedding(self._train_encoder())
        return self.embedding

    def _quantised_embedding(self, embedding):
        """L2-normalise, then round, so 1e-07 reduction wobble cannot move k-means."""
        normalised = _l2_normalise(np.asarray(embedding, dtype=np.float32))
        if self.round_decimals > 0:
            normalised = np.round(normalised, self.round_decimals)
        return np.ascontiguousarray(normalised, dtype=np.float32)

    # ----------------------------------------------------------------- clusters
    def _coarse_partition(self):
        """k-means over the learned embedding, with ``k`` set by the target size."""
        from sklearn.cluster import KMeans

        # The refinement stage, the BFS chunker and the conductance gate all work on
        # the networkx view, so make sure it exists even though Louvain is not used.
        self._ensure_nx_graph()

        if self.protected is None:
            clusterable = np.arange(self.num_nodes)
        else:
            clusterable = np.nonzero(~self.protected)[0]
        if clusterable.size == 0:
            return []

        embedding = self._representations()

        target = max(1, self.theta)
        num_clusters = max(1, int(math.ceil(clusterable.size / float(target))))
        num_clusters = min(num_clusters, clusterable.size)
        started = time.time()
        assignment = KMeans(
            n_clusters=num_clusters,
            random_state=self.seed,
            n_init=10,
        ).fit_predict(embedding[clusterable])
        cluster_seconds = time.time() - started

        blocks = {}
        for node, cluster in zip(clusterable.tolist(), assignment.tolist()):
            blocks.setdefault(int(cluster), []).append(int(node))
        self.gae_report.update(
            {
                "num_clusters_requested": num_clusters,
                "num_clusters_returned": len(blocks),
                "kmeans_seconds": cluster_seconds,
                "clusterable_nodes": int(clusterable.size),
            }
        )
        self.logger.info(
            "GAE k-means: %d cluster(s) over %d node(s) in %.2f s "
            "(k = ceil(n / ccp_theta), ccp_theta = %d)",
            len(blocks), clusterable.size, cluster_seconds, self.theta,
        )
        return [sorted(nodes) for _, nodes in sorted(blocks.items())]

    def partition(self):
        result = super(GAEPartition, self).partition()
        self.scale_report["gae"] = dict(self.gae_report)
        return result
