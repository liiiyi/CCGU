import copy
import logging
import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score

import config
from exp.exp import Exp
from exp.unlearning_core import (
    build_node_to_communities,
    calculate_edge_counts,
    calculate_robustness_similarity,
    remap_rows,
    remove_nodes_and_reindex,
)
from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL
from exp.methods.WeightOptimizer import EdgeWeightOptimizer


class Unlearn(Exp):
    """Community update followed by deterministic mapped-graph retraining."""

    def __init__(self, args):
        super(Unlearn, self).__init__(args)
        self.nucleus_graph = None
        self.logger = logging.getLogger("ExpUnlearn")
        self.load_data()

        set_file = (
            config.COM_PATH
            + self.args["dataset_name"]
            + "/"
            + self.args["partition"]
            + "_select_n"
        )
        selected_nodes = (
            self.data_store.load_param(address=set_file)
            if os.path.exists(set_file)
            else set()
        )

        num_communities_before = self.num_communities
        start_time = time.time()
        influence = self.gen_unlearning_influence(selected_nodes)
        update_start = time.time()
        self.unlearning(influence)
        community_update_seconds = time.time() - update_start
        retrain_start = time.time()
        self.retrain_model()
        retrain_seconds = time.time() - retrain_start
        self.unlearn_time = time.time() - start_time

        self.logger.info(
            "Unlearning completed: %d original nodes, %d affected communities, %.4f seconds",
            len(influence["unlearning_nodes"]),
            len(influence["unlearning_communities"]),
            self.unlearn_time,
        )

        metrics = {
            "stage": "unlearn",
            "num_nodes_original": int(self.num_nodes),
            "num_unlearned_nodes": len(influence["unlearning_nodes"]),
            "unlearn_fraction_of_original_nodes": (
                len(influence["unlearning_nodes"]) / float(self.num_nodes)
                if self.num_nodes else None
            ),
            "num_affected_communities": len(influence["unlearning_communities"]),
            "num_eligible_training_nodes": int(self.num_eligible_training_nodes),
            "num_communities_before": int(num_communities_before),
            "num_communities_after": int(self.num_communities),
            "community_update_seconds": community_update_seconds,
            "retrain_seconds": retrain_seconds,
            "unlearn_seconds": self.unlearn_time,
        }
        metrics.update(getattr(self, "retrain_metrics", {}))
        self.write_metrics(metrics)

    def load_data(self):
        """Load the original DGL graph and the deployed mapped-graph artifacts."""
        if not self.args["dgl_data"]:
            raise ValueError("CGE unlearning requires the existing DGL data interface.")

        self.graph = self.data_store.load_graph_from_dgl()
        self.num_nodes = self.graph.number_of_nodes()

        community_data = self.data_store.load_communities_info()
        self.n2c = community_data["n2c"]
        self.c2n = community_data["c2n"]
        self.num_communities = community_data["num_c"]

        nucleus_data = self.data_store.load_nucleus_data(if_sim=0)
        self.new_feats = np.asarray(nucleus_data["new_feats"]).copy()
        self.new_labels = np.asarray(nucleus_data["new_labels"]).copy()

        # Prefer the split the deployed model was actually trained on, so that the
        # updated model is evaluated on the same held-out mapped nodes.  Fall back
        # to the mapping stage's masks when Train has not been run yet.
        masks = None
        if self.data_store.has_nucleus_masks():
            candidate = self.data_store.load_nucleus_masks()
            if int(candidate.get("num_communities", -1)) == int(self.num_communities):
                masks = candidate
                self.logger.info(
                    "using the train/val/test split recorded by the Train stage "
                    "(seed %s, test_ratio %s)",
                    candidate.get("random_seed"),
                    candidate.get("test_ratio"),
                )
            else:
                self.logger.warning(
                    "stored split covers %s communities but the community file has "
                    "%d; ignoring it and falling back to the mapping-stage masks",
                    candidate.get("num_communities"),
                    self.num_communities,
                )
        if masks is None:
            self.logger.warning(
                "no deployment split found; using the mapping-stage masks.  Run "
                "--exp Train first if you want before/after numbers on one split."
            )
            masks = nucleus_data
        self.train_mask = np.asarray(masks["train_mask"], dtype=bool).copy()
        self.val_mask = np.asarray(masks["val_mask"], dtype=bool).copy()
        self.test_mask = np.asarray(masks["test_mask"], dtype=bool).copy()
        self.sim = dict(self.data_store.load_nucleus_data(if_sim=1))
        self.num_classes = self._original_num_classes()

    def _original_num_classes(self):
        """Output width taken from the original label space, matching Train."""
        node_data = getattr(self.graph, "ndata", {})
        if "label" in node_data:
            return int(node_data["label"].max().item()) + 1
        return int(np.asarray(self.new_labels).max()) + 1

    def gen_unlearning_influence(self, selected_nodes=None):
        """Sample a training-node request and locate every overlapping community."""
        del selected_nodes  # Kept for compatibility with the historical method call.
        self.n2c = build_node_to_communities(self.c2n)

        training_communities = {
            community
            for community, is_training in enumerate(self.train_mask)
            if is_training and community in self.c2n
        }
        training_nodes = sorted(
            {
                int(node)
                for community in training_communities
                for node in self.c2n[community]
            }
        )
        if not training_nodes:
            raise ValueError("No training nodes are available for unlearning.")

        unlearn_ratio = self.args["unlearn_ratio"]
        if unlearn_ratio > 1:
            num_unlearning_nodes = int(unlearn_ratio)
        elif 0 < unlearn_ratio <= 1:
            num_unlearning_nodes = max(1, int(unlearn_ratio * len(training_nodes)))
        else:
            raise ValueError("unlearn_ratio must be positive.")
        if num_unlearning_nodes > len(training_nodes):
            raise ValueError(
                "unlearn_ratio {} requests {} nodes but only {} eligible training "
                "nodes exist".format(
                    unlearn_ratio, num_unlearning_nodes, len(training_nodes)
                )
            )
        self.num_eligible_training_nodes = len(training_nodes)

        # A dedicated RNG, so the sampled request depends only on --random_seed and
        # not on how much of the global stream the earlier stages happened to draw.
        sampler = random.Random(self.args["random_seed"])
        unlearning_nodes = sampler.sample(training_nodes, num_unlearning_nodes)
        self.logger.info(
            "unlearning request: %d nodes (%.4f%% of the %d original nodes, %.4f%% "
            "of the %d eligible training nodes), seed %d",
            num_unlearning_nodes,
            100.0 * num_unlearning_nodes / max(1, self.num_nodes),
            self.num_nodes,
            100.0 * num_unlearning_nodes / max(1, len(training_nodes)),
            len(training_nodes),
            self.args["random_seed"],
        )
        unlearning_communities = {
            community
            for node in unlearning_nodes
            for community in self.n2c.get(node, ())
        }
        sim_pairs = {
            pair
            for pair in self.sim
            if pair[0] in unlearning_communities
            or pair[1] in unlearning_communities
        }

        influence_dict = {
            "features_communities": set(unlearning_communities),
            "labels_communities": set(unlearning_communities),
            "sim_communities_pairs": sim_pairs,
            "unlearning_nodes": unlearning_nodes,
            "unlearning_communities": unlearning_communities,
        }
        self.unlearning_set = unlearning_nodes
        self.influence_dict = influence_dict
        return influence_dict

    def calculate_threshold(self, distances):
        if len(distances) <= 1:
            return np.inf
        sorted_distances = np.sort(distances)
        gradients = np.diff(sorted_distances)
        return sorted_distances[int(np.argmax(gradients))]

    def unlearning(self, influence_dict):
        """Update communities and rebuild all data derived from deleted nodes."""
        (
            updated_c2n,
            old_to_new,
            _affected_old,
            affected_new,
        ) = remove_nodes_and_reindex(
            self.c2n,
            influence_dict["unlearning_nodes"],
        )
        if not updated_c2n:
            raise ValueError("The unlearning request removed every community.")

        self.c2n = updated_c2n
        self.n2c = build_node_to_communities(self.c2n)
        self.num_communities = len(self.c2n)

        self.new_feats = np.asarray(remap_rows(self.new_feats, old_to_new))
        self.new_labels = np.asarray(remap_rows(self.new_labels, old_to_new))
        self.train_mask = np.asarray(remap_rows(self.train_mask, old_to_new), dtype=bool)
        self.val_mask = np.asarray(remap_rows(self.val_mask, old_to_new), dtype=bool)
        self.test_mask = np.asarray(remap_rows(self.test_mask, old_to_new), dtype=bool)

        if self.args["agg_feat"] == "pca":
            self.new_feats = self.recalculate_features_pca(affected_new)
        elif self.args["agg_feat"] == "mean":
            self.new_feats = self.recalculate_features_mean(affected_new)
        else:
            raise ValueError("Unlearning supports the existing 'pca' and 'mean' feature mappings.")

        if self.args["agg_label"] == "th":
            self.new_labels = self.recalculate_labels_th1(
                affected_new,
                self.new_feats,
            )
        elif self.args["agg_label"] == "all":
            self.new_labels = self.recalculate_labels_all(affected_new)
        elif self.args["agg_label"] == "km":
            self.new_labels = self.recalculate_labels_kmeans(affected_new)
        else:
            raise ValueError("Unknown label mapping method.")

        self.sim = self.recalculate_sim_rubost(None)

        updated_nucleus_data = {
            "new_feats": self.new_feats,
            "new_labels": self.new_labels,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }
        self.data_store.save_communities_info(
            {
                "n2c": self.n2c,
                "c2n": self.c2n,
                "num_c": self.num_communities,
            }
        )
        self.data_store.save_nucleus_data(updated_nucleus_data, if_sim=0)
        self.data_store.save_nucleus_data(self.sim, if_sim=1)
        self.data_store.save_unlearned_data(
            {
                "unlearning_nodes": list(influence_dict["unlearning_nodes"]),
                "unlearning_communities": sorted(
                    influence_dict["unlearning_communities"]
                ),
            },
            suffix="request",
        )

    def _features_numpy(self):
        features = self.graph.ndata["feat"]
        if hasattr(features, "detach"):
            features = features.detach()
        if hasattr(features, "cpu"):
            features = features.cpu()
        return np.asarray(features.numpy())

    def _labels_numpy(self):
        labels = self.graph.ndata["label"]
        if hasattr(labels, "detach"):
            labels = labels.detach()
        if hasattr(labels, "cpu"):
            labels = labels.cpu()
        return np.asarray(labels.numpy())

    def recalculate_features_mean(self, features_communities):
        features = self._features_numpy()
        updated = self.new_feats.copy()
        for community in features_communities:
            updated[community] = features[self.c2n[community]].mean(axis=0)
        return updated.astype("float32", copy=False)

    def recalculate_features_pca(self, features_communities):
        """Recompute Equation (5) for affected, non-empty communities."""
        features = self._features_numpy()
        updated = self.new_feats.copy()
        n_components_ratio = 0.05

        for community in features_communities:
            nodes = self.c2n[community]
            if len(nodes) <= 1 / n_components_ratio:
                community_feature = features[nodes].mean(axis=0)
                distances = np.linalg.norm(
                    features[nodes] - community_feature,
                    axis=1,
                )
            else:
                num_components = max(
                    1,
                    int(np.ceil(len(nodes) * n_components_ratio)),
                )
                num_components = min(num_components, features.shape[1])
                transformed = PCA(n_components=num_components).fit_transform(
                    features[nodes].T
                )
                community_feature = transformed.mean(axis=1)
                distances = np.linalg.norm(
                    transformed - transformed.mean(axis=0),
                    axis=1,
                )

            if self.args["use_feat_rb"]:
                denominator = max(float(distances.sum()), np.finfo(float).eps)
                robustness = len(nodes) / denominator
                community_feature = community_feature * (
                    0.1 * np.exp(-robustness) + 1.0
                )
            updated[community] = community_feature

        return updated.astype("float32", copy=False)

    def recalculate_labels_th1(self, labels_communities, np_new_feats):
        labels = self._labels_numpy()
        features = self._features_numpy()
        updated = self.new_labels.copy()

        for community in labels_communities:
            nodes = np.asarray(self.c2n[community], dtype=int)
            distances = np.asarray(
                [
                    euclidean(np_new_feats[community], node_feature)
                    for node_feature in features[nodes]
                ]
            )
            threshold = self.calculate_threshold(distances)
            selected_nodes = nodes[distances <= threshold]
            voting_nodes = selected_nodes if len(selected_nodes) else nodes
            updated[community] = mode(
                labels[voting_nodes],
                keepdims=False,
            ).mode
        return updated.astype("int64", copy=False)

    def recalculate_labels_all(self, labels_communities):
        labels = self._labels_numpy()
        updated = self.new_labels.copy()
        for community in labels_communities:
            nodes = np.asarray(self.c2n[community], dtype=int)
            updated[community] = mode(labels[nodes], keepdims=False).mode
        return updated.astype("int64", copy=False)

    def recalculate_labels_kmeans(self, labels_communities):
        labels = self._labels_numpy()
        features = self._features_numpy()
        updated = self.new_labels.copy()
        for community in labels_communities:
            nodes = np.asarray(self.c2n[community], dtype=int)
            if len(nodes) < 2:
                voting_nodes = nodes
            else:
                cluster_labels = KMeans(
                    n_clusters=2,
                    random_state=0,
                    n_init=10,
                ).fit_predict(features[nodes])
                largest_cluster = mode(
                    cluster_labels,
                    keepdims=False,
                ).mode
                voting_nodes = nodes[cluster_labels == largest_cluster]
            updated[community] = mode(
                labels[voting_nodes],
                keepdims=False,
            ).mode
        return updated.astype("int64", copy=False)

    def calculate_edge_counts(self, n2c, c2n):
        del c2n  # The mapping is already represented by n2c.
        source, target = self.graph.edges()
        if hasattr(source, "cpu"):
            source = source.cpu()
            target = target.cpu()
        return calculate_edge_counts(source.numpy(), target.numpy(), n2c)

    def recalculate_sim_rubost(self, sim_communities_pairs):
        """Recompute mapped-edge scores after deletion.

        The argument is retained for compatibility.  All scores are rebuilt
        because a deleted boundary edge changes the degree normalizers in
        Equation (11), including some neighbor-to-neighbor scores.
        """
        del sim_communities_pairs
        edge_counts = self.calculate_edge_counts(self.n2c, self.c2n)
        if self.args["agg_edge"] == "jaccard":
            test_edge_method = 3
            include_jaccard = True
        elif self.args["agg_edge"] == "rubost":
            test_edge_method = self.args["test_edge_method"]
            include_jaccard = True
        elif self.args["agg_edge"] == "opt":
            optimizer_model = EdgeWeightOptimizer(
                num_communities=self.num_communities,
                initial_edge_counts=edge_counts,
                new_feats=self.new_feats,
                args=self.args,
            )
            optimizer = torch.optim.Adam(
                optimizer_model.parameters(),
                lr=0.01,
            )
            for _ in range(1000):
                optimizer.zero_grad()
                loss = optimizer_model()
                loss.backward()
                optimizer.step()
            return optimizer_model.calculate_sim()
        else:
            raise ValueError("Unknown edge mapping method.")
        # pair_reduction='source' is what exp_partition uses, so the deployed and
        # the post-deletion mapped graphs come out of the same rule.  On the
        # bidirectional DGL benchmark graphs 'source' and 'max' coincide, but on a
        # genuinely directed graph they do not, and disagreeing here would mean an
        # unlearning request silently rebuilt edges the deployment never had.
        return calculate_robustness_similarity(
            self.c2n,
            edge_counts,
            test_edge_method=test_edge_method,
            include_jaccard=include_jaccard,
            pair_reduction="source",
        )

    def calculate_sim2edge_threshold(self):
        if self.args["th_sim2edge"] < 0 or not self.sim:
            return 0.0
        values = np.sort(np.fromiter(self.sim.values(), dtype=float))
        if len(values) == 1:
            return float(values[0])
        gradients = np.diff(values)
        control = self.args["th_sim2edge"]
        if control > 0:
            count = max(1, int(len(values) * control))
            start = max(0, len(gradients) - count)
            index = start + int(np.argmax(gradients[start:]))
        else:
            index = int(np.argmax(gradients))
        return float(values[index])

    def gen_nucleus_graph(self):
        """Build the adjusted mapped graph using the existing DGL interface."""
        threshold = self.calculate_sim2edge_threshold()
        edges = [
            edge
            for edge, weight in self.sim.items()
            if weight >= threshold
        ]
        source = [edge[0] for edge in edges]
        target = [edge[1] for edge in edges]
        nucleus_graph = dgl.graph(
            (source, target),
            num_nodes=self.num_communities,
        )
        if self.args["use_edge_weight"]:
            nucleus_graph.edata["weight"] = torch.tensor(
                [self.sim[edge] for edge in edges],
                dtype=torch.float32,
            )

        nucleus_graph.ndata["feat"] = torch.tensor(
            self.new_feats,
            dtype=torch.float32,
        )
        nucleus_graph.ndata["label"] = torch.tensor(
            self.new_labels,
            dtype=torch.long,
        )
        nucleus_graph.ndata["train_mask"] = torch.tensor(
            self.train_mask,
            dtype=torch.bool,
        )
        nucleus_graph.ndata["val_mask"] = torch.tensor(
            self.val_mask,
            dtype=torch.bool,
        )
        nucleus_graph.ndata["test_mask"] = torch.tensor(
            self.test_mask,
            dtype=torch.bool,
        )

        zero_degree_nodes = (
            nucleus_graph.in_degrees() == 0
        ).nonzero(as_tuple=False).flatten()
        if len(zero_degree_nodes):
            nucleus_graph = dgl.remove_nodes(
                nucleus_graph,
                zero_degree_nodes,
            )
        self.nucleus_graph = nucleus_graph
        self.data_store.save_nucleus_graph(nucleus_graph)
        return nucleus_graph

    def retrain_model(self):
        """Retrain a fresh backbone on the adjusted mapped graph."""
        graph = self.gen_nucleus_graph()
        device = torch.device(
            f'cuda:{self.args["cuda"]}'
            if torch.cuda.is_available()
            else "cpu"
        )
        graph = graph.to(device)
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
        if not bool(train_mask.any()):
            raise ValueError("No mapped training nodes remain after unlearning.")

        num_feats = features.shape[1]
        # Must match the head that Train built, otherwise before/after Macro F1
        # are computed over different label spaces.
        num_classes = max(self.num_classes, int(labels.max().item()) + 1)
        model = NodeClassifierDGL(num_feats, num_classes, self.args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args["train_lr"],
            weight_decay=self.args["train_weight_decay"],
        )

        best_score = float("-inf")
        best_model = None
        for epoch in range(self.args["num_epochs"]):
            model.train()
            logits = model(
                features,
                graph,
                graph.edata.get("weight", None),
            )
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score = self._mask_f1(model, graph, features, labels, val_mask)
            selection_score = score if not np.isnan(score) else -float(loss.item())
            if selection_score > best_score:
                best_score = selection_score
                best_model = copy.deepcopy(model.state_dict())
            self.logger.info(
                "Unlearning retrain epoch %d, loss %.6f, val macro-F1 %.6f",
                epoch,
                loss.item(),
                score,
            )

        if best_model is not None:
            model.load_state_dict(best_model)
        self.model = model
        self.nucleus_graph = graph

        test_f1 = self._mask_f1(model, graph, features, labels, test_mask)
        test_acc = self._mask_accuracy(
            model,
            graph,
            features,
            labels,
            test_mask,
        )
        val_f1 = self._mask_f1(model, graph, features, labels, val_mask)
        val_acc = self._mask_accuracy(model, graph, features, labels, val_mask)
        self.logger.info(
            "Updated mapped graph test macro-F1 %.6f, accuracy %.6f",
            test_f1,
            test_acc,
        )
        self.retrain_metrics = {
            "mapped_graph_nodes": int(graph.number_of_nodes()),
            "mapped_graph_edges": int(graph.number_of_edges()),
            "train_nodes": int(train_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "num_classes": int(num_classes),
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
        }

        model_path = self.data_store.target_model_file + "_unlearned.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_feats": num_feats,
                "num_classes": num_classes,
                "target_model": self.args["target_model"],
                "unlearning_nodes": list(self.unlearning_set),
            },
            model_path,
        )
        self.logger.info("Updated model saved to %s", model_path)
        self.retrain_metrics["model_path"] = model_path
        return model

    @staticmethod
    def _mask_f1(model, graph, features, labels, mask):
        if not bool(mask.any()):
            return float("nan")
        model.eval()
        with torch.no_grad():
            logits = model(features, graph, graph.edata.get("weight", None))
            prediction = logits[mask].argmax(dim=1)
        return f1_score(
            labels[mask].cpu(),
            prediction.cpu(),
            average="macro",
        )

    @staticmethod
    def _mask_accuracy(model, graph, features, labels, mask):
        if not bool(mask.any()):
            return float("nan")
        model.eval()
        with torch.no_grad():
            logits = model(features, graph, graph.edata.get("weight", None))
            prediction = logits[mask].argmax(dim=1)
        return accuracy_score(labels[mask].cpu(), prediction.cpu())
