import json
import logging
import math
import os.path

import matplotlib.pyplot as plt
import networkx as nx

from exp.exp import Exp
from exp.methods.SLPA import SLPA
from exp.methods.OSLOM import OSLOM
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import log
import config
import time
import torch
from sklearn.cluster import KMeans
from exp.methods.WeightOptimizer import EdgeWeightOptimizer
from exp.methods.CCP import CommunityCentricPartition
from exp.unlearning_core import (
    calculate_edge_counts as core_calculate_edge_counts,
    calculate_robustness_similarity,
)
from exp.methods.partition_diagnostics import (
    check_partition_health,
    format_summary,
    summarise_partition,
)
from lib_utils.stats import majority_label
import torch.optim as optim

# The optional community-detection backends (hnswlib for CCD/NIKM, infomap for
# Infomap, cdlib for the legacy Louvain wrapper) are imported inside
# gen_community().  Importing them at module scope used to make `--exp Partition`
# impossible for *every* partition method whenever one optional wheel was absent.

class GraphCommunityPartition(Exp):
    def __init__(self, args):
        super(GraphCommunityPartition, self).__init__(args)
        self.slpa_time = None
        self.logger = logging.getLogger('ExpPartition')
        self.args = args

        self.aggregate_feat_info = self.args['agg_feat']
        self.aggregate_label_info = self.args['agg_label']
        self.aggregate_e_info = self.args['agg_edge']

        self.logger = logging.getLogger('ExpGraphCommunityPartition')
        self.load_data()

        calculate_time = self.aggregate()

        config_str = (
            f"\n"
            f"Dataset Name: {args['dataset_name']}\n"
            f"Aggregation Feature Method: {args['agg_feat']}\n"
            f"Aggregation Label Method: {args['agg_label']}\n"
            f"Aggregation Edge Method: {args['agg_edge']}\n"
            f"Partition Method: {args['partition']}\n"
            f"Threshold Similarity to Edge: {args['th_sim2edge']}\n"
            f"Test Edge Method: {args['test_edge_method']}\n"
            f"Use Edge Weight: {args['use_edge_weight']}"
        )
        self.logger.info(config_str)

        self.logger.info(f'Time Consumption of Partition: {self.time_partition}')
        self.logger.info(
            f'Time Consumption of Calculation (mapping only, detection excluded): '
            f'{calculate_time}')

        metrics = {
            'stage': 'partition',
            'partition_seconds': self.time_partition,
            'mapping_seconds': calculate_time,
            'partition_diagnostics': getattr(self, 'partition_summary', None),
            'partition_attempts': getattr(self, 'partition_attempts', None),
            'num_mapped_edges_scored': getattr(self, 'num_sim_pairs', None),
            'partition_scale_report': getattr(self, 'partition_scale_report', None),
            'detector_backend': getattr(self, 'detector_backend', None),
        }
        metrics.update(getattr(self, 'stage_times', {}))
        self.write_metrics(metrics)

    def load_data(self):
        if self.args['dgl_data']:
            self.graph = self.data_store.load_graph_from_dgl()
            self.num_nodes = self.graph.number_of_nodes()
            np.set_printoptions(threshold=np.inf)
        else:
            self.data = self.data_store.load_raw_data()
            self.gen_train_graph()

    def gen_train_graph(self):
        edges = [(self.data.edge_index[0][i], self.data.edge_index[1][i]) for i in range(self.data.edge_index[0].shape[0])]
        nodes = set(edge[0] for edge in edges) | set(edge[1] for edge in edges)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    #: Partition methods whose implementation is a label-propagation variant.  The
    #: one historically named 'oslom' contains no Louvain stage and is not OSLOM;
    #: see exp/methods/README.md.
    LEGACY_LABEL_PROPAGATION = ('oslom', 'slpa')

    def _run_partition(self, seed):
        """Dispatch to the requested community-detection backend.

        Returns ``(n2c, c2n, num_communities, elapsed_seconds)``.  Optional
        third-party backends are imported here so that a missing wheel only
        affects the method that needs it.
        """
        method = self.args['partition']
        args = dict(self.args)
        args['random_seed'] = seed

        if method == 'ccp':
            detector = CommunityCentricPartition(self.graph, args=args)
            result = detector.partition()
            self.partition_scale_report = getattr(detector, 'scale_report', None)
            self.detector_backend = 'networkx-louvain (CPU)'
            return result

        if method == 'gae':
            from exp.methods.GAEPartition import GAEPartition

            detector = GAEPartition(self.graph, args=args)
            result = detector.partition()
            self.partition_scale_report = getattr(detector, 'scale_report', None)
            self.detector_backend = 'dgl-gae ({})'.format(
                (getattr(detector, 'gae_report', {}) or {}).get('device', 'unknown')
            )
            return result

        if method == 'slpa':
            slpa = SLPA(self.graph, max_iterations=50, threshold=0.2, max_communities_per_node=5)
            return slpa.SLPA_partition()

        if method == 'oslom':
            oslom = OSLOM(self.graph, args=args)
            return oslom.OSLOM_partition()

        if method == 'lpa':
            from exp.methods.LPA import LPA

            return LPA(self.graph, args=args).lpa_partition()

        if method == 'louvain':
            from exp.methods.Louvain import Louvain

            return Louvain(self.graph, args=args).louvain_partition()

        if method == 'infomap':
            try:
                from exp.methods.Infomap import Infomap
            except ImportError as error:  # pragma: no cover - optional backend
                raise ImportError(
                    "--partition infomap needs the 'infomap' package "
                    "(pip install infomap==2.7.1)"
                ) from error
            return Infomap(self.graph, args=args, seed=seed).partition()

        if method == 'nikm':
            try:
                from exp.methods.GetEmbed import GetEmbed
                from exp.methods.NIKM import NIKM
            except ImportError as error:  # pragma: no cover - optional backend
                raise ImportError(
                    "--partition nikm needs its embedding backend; "
                    "see reproduction/requirements-lock.txt"
                ) from error
            embeddings = GetEmbed(self.graph, args).generate_embeddings()
            return NIKM(self.graph, embeddings=embeddings, args=args).run()

        if method == 'test':
            try:
                from exp.methods.CCD import CentroidCommunityDetection as CCD
            except ImportError as error:  # pragma: no cover - optional backend
                raise ImportError(
                    "--partition test needs hnswlib (pip install hnswlib==0.8.0)"
                ) from error
            n2c, c2n, elapsed = CCD(self.graph, args=args).run()
            return n2c, c2n, len(c2n), elapsed

        raise ValueError(
            "unknown --partition {!r}; choose one of "
            "ccp, gae, slpa, oslom, nikm, lpa, louvain, test, infomap".format(method)
        )

    def _diagnose(self, c2n, elapsed_seconds):
        """Summarise partition quality and log it."""
        labels = None
        edges = None
        if hasattr(self.graph, 'ndata') and 'label' in getattr(self.graph, 'ndata', {}):
            labels = self.graph.ndata['label'].cpu().numpy()
        if hasattr(self.graph, 'edges') and hasattr(self.graph, 'num_nodes'):
            src, dst = self.graph.edges()
            edges = (src.cpu().numpy(), dst.cpu().numpy())
        summary = summarise_partition(
            c2n,
            num_nodes=self.num_nodes,
            labels=labels,
            edges=edges,
            elapsed_seconds=elapsed_seconds if isinstance(elapsed_seconds, float) else None,
        )
        summary['dataset'] = self.args['dataset_name']
        summary['method'] = self.args['partition']
        summary['seed'] = self.args['random_seed']
        self.logger.info('\n%s', format_summary(summary))
        return summary

    def gen_community(self):
        if os.path.exists(self.data_store.community_file):
            # The cache key does not include --random_seed, so reusing it means the
            # seed on this command line had no influence on the partition.  Say so
            # loudly: a silently reused partition is the easiest way to mistake one
            # seed's community structure for another's.
            self.logger.warning(
                "reusing the existing community file %s; --random_seed %s did NOT "
                "affect this partition.  Delete the file (or point CCGU_DATA_ROOT at "
                "a fresh directory) to re-detect communities.",
                self.data_store.community_file, self.args['random_seed'],
            )
            com_data = self.data_store.load_communities_info()
            n2c = com_data['n2c']
            c2n = com_data['c2n']
            num_communities = com_data['num_c']

            self.time_partition = 'Partition File is already exist.'
            self.partition_summary = self._diagnose(c2n, None)
        else:
            if self.args['partition'] in self.LEGACY_LABEL_PROPAGATION:
                self.logger.warning(
                    "--partition %s selects the legacy label-propagation implementation. "
                    "It is kept verbatim for comparability, but it is not OSLOM, it has no "
                    "Louvain initialisation stage, and it produces an almost disjoint "
                    "partition.  Use --partition ccp for the paper-described coarse-to-fine "
                    "process; see exp/methods/README.md.",
                    self.args['partition'],
                )

            attempts = []
            base_seed = self.args['random_seed']
            max_attempts = 1 + max(0, self.args['partition_max_retries'])
            result = None
            for attempt in range(max_attempts):
                seed = base_seed + attempt
                self.logger.info(
                    'partition attempt %d/%d with seed %d (method %s)',
                    attempt + 1, max_attempts, seed, self.args['partition'],
                )
                n2c, c2n, num_communities, self.time_partition = self._run_partition(seed)
                summary = self._diagnose(c2n, self.time_partition)
                summary['attempt'] = attempt + 1
                summary['attempt_seed'] = seed
                problems = check_partition_health(
                    summary, min_communities=self.args['partition_min_communities']
                )
                summary['quality_gate_problems'] = problems
                attempts.append(summary)
                if not problems:
                    result = (n2c, c2n, num_communities, summary)
                    break
                self.logger.warning(
                    'partition attempt %d failed the quality gate: %s',
                    attempt + 1, '; '.join(problems),
                )

            # Every attempt is recorded, so a retry can never quietly hide a bad
            # draw, and the gate never looks at downstream model quality.
            self.partition_attempts = attempts
            if result is None:
                raise RuntimeError(
                    "every partition attempt failed the quality gate ({} attempt(s)); "
                    "last reasons: {}".format(
                        len(attempts), '; '.join(attempts[-1]['quality_gate_problems'])
                    )
                )
            n2c, c2n, num_communities, self.partition_summary = result

            com_data = {
                'n2c': n2c,
                'c2n': c2n,
                'num_c': num_communities
            }
            self.data_store.save_communities_info(params=com_data)
            self._save_partition_report(attempts)

        self._check_communities(c2n)

        return n2c, c2n, num_communities

    def _save_partition_report(self, attempts):
        report_path = self.data_store.community_file + '_diagnostics.json'
        payload = {
            'dataset': self.args['dataset_name'],
            'method': self.args['partition'],
            'random_seed': self.args['random_seed'],
            'partition_max_retries': self.args['partition_max_retries'],
            'partition_min_communities': self.args['partition_min_communities'],
            'attempts': attempts,
            'accepted_attempt': attempts[-1]['attempt'] if attempts else None,
        }
        if self.args['partition'] in ('ccp', 'gae'):
            payload['ccp_config'] = {
                key: self.args[key]
                for key in sorted(self.args)
                if key.startswith('ccp_') or key.startswith('gae_')
            }
            payload['scale_report'] = getattr(self, 'partition_scale_report', None)
            payload['detector_backend'] = getattr(self, 'detector_backend', None)
        with open(report_path, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        self.logger.info('partition diagnostics written to %s', report_path)

    def _check_communities(self, c2n):
        # Initialize a dictionary to count the occurrences of each node in communities
        node_counts = {}

        # Count the number of times each node appears in the communities
        for community, nodes in c2n.items():
            for node in nodes:
                if node not in node_counts:
                    node_counts[node] = 0
                node_counts[node] += 1

        # Initialize a dictionary to count the number of nodes that overlap a specific number of times
        overlap_counts = {i: 0 for i in range(1, 6)}

        # Count the number of nodes that overlap 1, 2, 3, 4, or 5 times
        for node, count in node_counts.items():
            if count in overlap_counts:
                overlap_counts[count] += 1
            else:
                overlap_counts[5] += 1  # Count nodes that overlap more than 5 times as 5

        self.logger.info(f"Number of communities: {len(c2n)}")
        # Log the results
        for i in range(1, 6):
            self.logger.info(f"Number of nodes overlapping {i} time(s): {overlap_counts[i]}")

    def aggregate(self):
        """Build the mapped features, labels, masks and edges.

        Returns the mapping wall clock **excluding** community detection, which is
        reported separately as ``self.time_partition``.  Timing them together would
        make a detector comparison meaningless, because Equation (11) over every
        observed community pair can cost more than the detector itself on a graph
        with many communities.
        """
        n2c, c2n, num_communities = self.gen_community()

        # Mapping clock starts here, after detection.
        start_time = time.time()
        self.stage_times = {}
        stage_start = time.time()
        if self.aggregate_feat_info == 'pca':
            new_feats = self.aggregate_features_pca(c2n)
        elif self.aggregate_feat_info == 'mean':
            new_feats = self.aggregate_features_mean(c2n)
        else:
            self.logger.info("Error param")
            return 1
        self.stage_times['feature_mapping_seconds'] = time.time() - stage_start

        stage_start = time.time()
        if self.aggregate_label_info == 'th':
            new_labels, selected_nodes = self.aggregate_labels_th1(c2n, np_new_feats=new_feats)
            set_file = config.COM_PATH + self.args['dataset_name'] + "/" + self.args['partition'] + '_select_n'
            self.data_store.save_param(data=selected_nodes, address=set_file)
        elif self.aggregate_label_info == 'all':
            new_labels = self.aggregate_labels_all(c2n)
        elif self.aggregate_label_info == 'km':
            new_labels, time_agg_label = self.aggregate_labels_kmeans(c2n, np_new_feats=new_feats)
        else:
            self.logger.info("Error param")
            return 1
        self.stage_times['label_mapping_seconds'] = time.time() - stage_start

        stage_start = time.time()
        train_mask, val_mask, test_mask = self.generate_masks(num=num_communities, test_size=self.args['test_ratio'])

        # save
        nucleus_data = {
            'new_feats': new_feats,
            'new_labels': new_labels,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

        self.data_store.save_nucleus_data(params=nucleus_data, if_sim=0)
        self.stage_times['mask_and_save_seconds'] = time.time() - stage_start

        stage_start = time.time()
        if self.aggregate_e_info == 'jaccard':
            sim = self.calculate_sim(n2c, c2n)
        elif self.aggregate_e_info == 'rubost':
            sim = self.calculate_nucleus_sim(n2c, c2n, n_communities=num_communities)
        elif self.aggregate_e_info == 'opt':
            sim = self.calculate_optimizer_sim(n2c, c2n, new_feats)
        else:
            self.logger.info("Error param")
            return 1
        self.stage_times['edge_mapping_seconds'] = time.time() - stage_start
        self.logger.info(f'{len(sim)} pairs are found.')
        self.num_sim_pairs = len(sim)

        # sim = self.instrument.smooth_edge_weights(sim, threshold_weight=0.5)

        # save
        self.data_store.save_nucleus_data(params=sim, if_sim=1)

        return time.time() - start_time

    def generate_masks(self, num, test_size=0.2):
        """
        Param
        num: number of communities
        """
        seed = self.args['random_seed']
        X = np.array(range(num))
        train, other, _, _ = train_test_split(X, X, test_size=test_size, random_state=seed)
        val, test, _, _ = train_test_split(X[other], X[other], test_size=0.5, random_state=seed)
        train_mask = np.array([False] * num)
        val_mask = np.array([False] * num)
        test_mask = np.array([False] * num)
        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True
        return train_mask, val_mask, test_mask

    def aggregate_features_mean(self, c2n):
        in_feats = self.graph.ndata['feat'].shape[1]
        X = self.graph.ndata['feat'].numpy()
        np_new_feats = np.zeros((len(c2n), in_feats), dtype='float32')

        for community, nodes in c2n.items():
            np_new_feats[community] = np.mean(X[nodes], axis=0)

        self.logger.info('New shapes for feat by mean.')
        return np_new_feats

    def aggregate_features_pca(self, c2n, n_components_ratio=0.05):
        """
        :param c2n: dict, C2N
        :param n_components_ratio: float, PCA
        :return: FEAT & LABEL
        """
        in_feats = self.graph.ndata['feat'].shape[1]
        X = self.graph.ndata['feat'].numpy()
        np_new_feats = np.zeros((len(c2n), in_feats), dtype='float32')
        feature_robustness = {}

        for community, nodes in c2n.items():
            if len(nodes) <= 1 / n_components_ratio:

                community_feature = X[nodes].mean(axis=0)
                np_new_feats[community] = community_feature
                distances = np.linalg.norm(X[nodes] - community_feature, axis=1)
            else:
                num_components = max(1, int(np.ceil(len(nodes) * n_components_ratio)))
                if num_components >= in_feats:
                    num_components = in_feats
                pca = PCA(n_components=num_components)
                transformed_feats = pca.fit_transform(X[nodes].T)

                community_feature = transformed_feats.mean(axis=0)
                np_new_feats[community] = transformed_feats.mean(axis=1)

                distances = np.linalg.norm(transformed_feats - community_feature, axis=1)

            # A singleton community, or one whose members share identical features,
            # has distances summing to exactly 0.  Dividing by it yields `inf` plus a
            # numpy RuntimeWarning for every such community -- and the hold-out
            # protocol deliberately creates thousands of singletons.  Clamp the
            # denominator, matching Unlearn.recalculate_features_pca, so the stored
            # value is finite whether or not --use_feat_rb is set.
            denominator = max(float(distances.sum()), np.finfo(np.float64).eps)
            feature_robustness[community] = len(nodes) / denominator

        # instru = Instruments
        # smoothed_robustness = instru.smooth_edge_weights(feature_robustness, threshold_weight=0.5)

        if self.args['use_feat_rb']:
            for community, robustness in feature_robustness.items():
                robustness = 0.1 * np.exp(-robustness) + 1.0
                np_new_feats[community] *= robustness

        self.logger.info('New shapes for feat are generated by PCA.')

        return np_new_feats

    def calculate_threshold(self, distances):
        """Equation (8): tau = d_{argmax g} with g_i = d_{i+1} - d_i on sorted d.

        The consecutive difference is what Equation (8) defines, and it is what
        ``exp/exp_unlearn.py`` recomputes for the communities a deletion touches.
        Using ``np.gradient`` here instead (a centred difference, which also
        shifts the arg-max) made Partition and Unlearn disagree, so an unlearning
        request silently relabelled communities it had not modified.
        """
        if len(distances) <= 1:
            return np.inf
        sorted_distances = np.sort(distances)
        gradients = np.diff(sorted_distances)

        # Optional: self.plot_gradients(sorted_distances, gradients)

        max_gradient_idx = int(np.argmax(gradients))
        threshold = sorted_distances[max_gradient_idx]
        return threshold

    def plot_gradients(self, distances, gradients):
        plt.figure()
        plt.plot(distances, label='Distances')
        plt.plot(gradients, label='Gradients')
        plt.xlabel('Sorted Nodes')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Distance and Gradient Plot')
        plt.show()

    def aggregate_labels_th1(self, c2n, np_new_feats):
        """Equations (7)-(9): distance to the mapped feature, elbow cut, majority vote.

        This method was the body of a triple-quoted block, so ``--agg_label th``
        -- the default, and the setting that implements Equation (9) -- raised
        ``AttributeError: 'GraphCommunityPartition' object has no attribute
        'aggregate_labels_th1'`` before any mapping happened.  It is restored
        verbatim except that the SciPy mode call goes through
        ``lib_utils.stats.majority_label`` so it behaves the same on every
        supported SciPy, and it mirrors ``Unlearn.recalculate_labels_th1``.

        Returns the mapped labels and the set of original nodes that actually
        voted; ``exp/exp_unlearn.py`` reads that set back from disk.
        """
        Y = self.graph.ndata['label'].cpu().numpy()
        X = self.graph.ndata['feat'].cpu().numpy()
        np_new_labels = np.zeros(len(c2n), dtype='int64')

        # Set to store all selected nodes across iterations
        selected_nodes_set = set()

        for community, nodes in tqdm(c2n.items(), desc="Aggregating Labels by th"):
            nodes = np.asarray(nodes, dtype=int)
            community_features = X[nodes]
            distances = np.array([euclidean(np_new_feats[community], node_feat) for node_feat in community_features])

            # Calculate the threshold automatically (Equation 8)
            threshold = self.calculate_threshold(distances)

            # Select nodes with distance less than the threshold
            selected_nodes = nodes[distances <= threshold]

            # Add selected nodes to the set
            selected_nodes_set.update(int(node) for node in selected_nodes)

            if len(selected_nodes) > 0:
                np_new_labels[community] = majority_label(Y[selected_nodes])
            else:
                # If no nodes are selected, use the original majority vote for robustness
                np_new_labels[community] = majority_label(Y[nodes])

        self.logger.info('New shapes for label are generated by major.')

        # Return both the new labels and the set of all selected nodes
        return np_new_labels, selected_nodes_set

    def aggregate_labels_kmeans(self, c2n, np_new_feats):
        """
        :param c2n: dict, C2N
        :param np_new_feats: np.array, MEAN_FEAT
        :return: COMM_LABEL
        """
        Y = self.graph.ndata['label'].numpy()
        X = self.graph.ndata['feat'].numpy()
        np_new_labels = np.zeros(len(c2n), dtype='int64')

        start_time = time.time()

        for community, nodes in tqdm(c2n.items(), desc="Aggregating labels with K-means"):
            if np.any(np.array(nodes) >= len(X)):
                self.logger.error(f"Community {community} has nodes out of bounds: {nodes}")
                continue

            community_features = X[nodes]
            kmeans = KMeans(n_clusters=2, random_state=0).fit(community_features)
            labels = kmeans.labels_
            largest_cluster_label = majority_label(labels)
            largest_cluster_indices = [node for node, label in zip(nodes, labels) if label == largest_cluster_label]

            if len(largest_cluster_indices) > 0:
                np_new_labels[community] = majority_label(Y[largest_cluster_indices])
            else:
                np_new_labels[community] = majority_label(Y[nodes])

        end_time = time.time()
        time_agg_label = end_time - start_time
        self.logger.info(f'New shapes for label by K-means. Aggregation time: {time_agg_label:.2f} seconds')
        return np_new_labels, time_agg_label

    def aggregate_labels_th(self, c2n, np_new_feats):
        """
        :param c2n: dict, C2N
        :param np_new_feats: ndarray, FEAT
        :return: COMM_LABEL
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Y = self.graph.ndata['label'].to(device)
        X = self.graph.ndata['feat'].to(device)
        np_new_labels = torch.zeros(len(c2n), dtype=torch.int64, device=device)

        np_new_feats = torch.tensor(np_new_feats, device=device)

        start_time = time.time()

        for community, nodes in tqdm(c2n.items(), desc="Aggregating Labels"):
            nodes = torch.tensor(nodes, device=device)
            community_features = X[nodes]

            distances = torch.tensor(
                [euclidean(np_new_feats[community].cpu().numpy(), node_feat.cpu().numpy()) for node_feat in
                 community_features], device=device)

            threshold = self.calculate_threshold(distances.cpu().numpy())

            selected_nodes = nodes[distances <= threshold]

            if len(selected_nodes) > 0:
                np_new_labels[community] = majority_label(Y[selected_nodes].cpu().numpy())
            else:
                np_new_labels[community] = majority_label(Y[nodes].cpu().numpy())

        end_time = time.time()
        time_agg_label = end_time - start_time

        self.logger.info('New shapes for label by major.')

        return np_new_labels.cpu().numpy(), time_agg_label



    def aggregate_labels_all(self, c2n):
        """
        :param c2n: dict, C2N
        :return: COMM_LABEL
        """
        Y = self.graph.ndata['label'].numpy()
        np_new_labels = np.zeros(len(c2n), dtype='int64')

        for community, nodes in c2n.items():
            np_new_labels[community] = majority_label(Y[nodes])

        self.logger.info('New shapes for label by major.')
        return np_new_labels

    def aggregate_edges_jaccard_op(self, n2c, c2n):
        sim = {}
        union_size = {}
        community_set = {}

        src, dst = self.graph.edges()
        src, dst = src.numpy(), dst.numpy()
        for idx in tqdm(range(len(src))):
            edge = (src[idx], dst[idx])
            if not(edge[0] in n2c and edge[1] in n2c):
                continue
            src_communities = n2c[edge[0]]
            dst_communities = n2c[edge[1]]
            for a in src_communities:
                for b in dst_communities:
                    k = (a, b)
                    if k[0] != k[1]:
                        if k in sim:
                            sim[k] += 1
                        else:
                            sim[k] = 1

        # calculate sim
        for k in tqdm(sim.keys()):
            union_size_key = k if k[0] < k[1] else (k[1], k[0])  # small number first
            if union_size_key in union_size:
                denominator = union_size[union_size_key]
            else:
                if union_size_key[0] not in community_set:
                    community_set[union_size_key[0]] = set(c2n[union_size_key[0]])
                if union_size_key[1] not in community_set:
                    community_set[union_size_key[1]] = set(c2n[union_size_key[1]])
                denominator = len(community_set[union_size_key[0]].union(community_set[union_size_key[1]]))
                # denominator = len(set(community2node[union_size_key[0]]).union(set(community2node[union_size_key[1]])))
                union_size[union_size_key] = denominator
            sim[k] /= denominator
        return sim

    def calculate_edge_counts(self, n2c, c2n):
        """Directed original-graph edge counts between communities, sparsely.

        Delegates to ``exp.unlearning_core.calculate_edge_counts`` so that the
        deployed mapped graph and the post-deletion mapped graph are built by the
        same code.  Only pairs that carry an edge are stored: pre-allocating every
        ordered pair is quadratic in the community count, which on Coauthor-CS
        with held-out evaluation nodes (~5,600 communities) is ~31 million
        dictionary entries before a single edge is read.  The returned counts are
        identical -- the previous version only differed by also storing zeros, and
        every reader accesses it through ``.get(key, 0)``.
        """
        del c2n  # the mapping is fully described by n2c
        src, dst = self.graph.edges()
        if hasattr(src, 'cpu'):
            src, dst = src.cpu(), dst.cpu()
        return core_calculate_edge_counts(src.numpy(), dst.numpy(), n2c)

    def calculate_sim(self, n2c, c2n):
        sim = self.aggregate_edges_jaccard_op(n2c, c2n)
        self.logger.info(f'{len(sim)} similarity pairs are found.')

        # store

        sim_vals = list(sim.values())
        sim_vals.sort()

        # store

        try:
            fig = plt.Figure()
            plt.plot(sim_vals)
            fig.suptitle(f'Similarity scores:')
            plt.xlabel('Index')
            plt.ylabel('Similarity score')

            # fig.show()
            # fig.savefig()
        except:
            self.logger.info('Fail to save figure')

        return sim

    def calculate_nucleus_sim(self, n2c, c2n, n_communities):
        """
        Calculate similarity scores between communities using different edge robustness methods
        and Jaccard similarity.

        Parameters:
            n2c: dict, node to community mapping
            c2n: dict, community to nodes mapping
            n_communities: int, number of communities
            edge_counts: dict, edge counts between communities
            test_edge_method: int, method for edge robustness calculation (0, 1, 2)
        """
        self.logger.info('Using similarity measure: node-based, Jaccard with edge robustness')

        del n_communities  # the community ids are exactly the keys of c2n

        edge_counts = self.calculate_edge_counts(n2c, c2n)

        # community_set: dictionary
        # key: community
        # val: set of its member nodes
        community_set = {community: set(c2n[community]) for community in c2n}

        # Sparse Equation (11).  The previous formulation scanned all
        # n_communities^2 / 2 ordered pairs and recomputed both degree normalisers
        # with an inner loop over every community for each connected pair, i.e.
        # O(C^2 + P*C); this is O(P log P) after one pass over the edge counts.
        # `pair_reduction='source'` keeps the historical read-out, s_ij taken from
        # the (low, high) direction only.
        sim = calculate_robustness_similarity(
            c2n,
            edge_counts,
            test_edge_method=self.args['test_edge_method'],
            include_jaccard=True,
            pair_reduction='source',
        )

        #save data
        set_file = config.COM_PATH + self.args['dataset_name'] + "/" + self.args['partition'] + '_comset'
        self.data_store.save_param(data=community_set, address=set_file)

        return sim

    def calculate_optimizer_sim(self, n2c, c2n, new_feats):
        edge_counts = self.calculate_edge_counts(n2c, c2n)
        optimizer_model = EdgeWeightOptimizer(num_communities=len(c2n), initial_edge_counts=edge_counts, new_feats=new_feats, args=self.args)
        optimizer = optim.Adam(optimizer_model.parameters(), lr=0.01)

        num_epochs = 1000
        for epoch in tqdm(range(num_epochs), desc="Optimizer Sim"):
            optimizer.zero_grad()
            loss = optimizer_model()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        return optimizer_model.calculate_sim()






