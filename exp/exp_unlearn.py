import logging
import math
import os.path

import matplotlib.pyplot as plt
import networkx as nx

from exp.exp import Exp
from exp.methods.SLPA import SLPA
from exp.methods.OSLOM import OSLOM
from exp.methods.NIKM import NIKM
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import log
import config
import time
import torch
from sklearn.cluster import KMeans
from exp.methods.GetEmbed import GetEmbed as ge
from exp.methods.Louvain import Louvain
from exp.methods.WeightOptimizer import EdgeWeightOptimizer
import torch.optim as optim
from exp.methods.CCD import CentroidCommunityDetection as CCD
from exp.methods.Infomap import Infomap

class Unlearn(Exp):
    def __init__(self, args):
        super(Unlearn, self).__init__(args)
        self.nucleus_graph = None
        self.logger = logging.getLogger('ExpUnlearn')
        self.load_data()

        set_file = config.COM_PATH + self.args['dataset_name'] + "/" + self.args['partition'] + '_select_n'
        selected_nodes = self.data_store.save_param(address=set_file)

        stime = time.time()
        inf = self.gen_unlearning_influence(selected_nodes)
        self.unlearning(inf)
        unlearn_time = time.time() - stime

        config_str = (
            f"\n"
            f"Dataset Name: {args['dataset_name']}\n"
            f"Aggregation Feature Method: {args['agg_feat']}\n"
            f"Aggregation Label Method: {args['agg_label']}\n"
            f"Aggregation Edge Method: {args['agg_edge']}\n"
            f"Partition Method: {args['partition']}\n"
            f"Threshold Similarity to Edge: {args['th_sim2edge']}\n"
            f"Test Edge Method: {args['test_edge_method']}\n"
            f"Use Edge Weight: {args['use_edge_weight']}\n"
            f"GNN Model: {args['target_model']}"
        )
        self.logger.info(config_str)

        self.logger.info(f'Unlearn time: {unlearn_time} seconds')

    def load_data(self):
        if self.args['dgl_data']:
            self.graph = self.data_store.load_graph_from_dgl()
            self.num_nodes = self.graph.number_of_nodes()
            np.set_printoptions(threshold=np.inf)
        else:
            self.data = self.data_store.load_raw_data()
            self.gen_train_graph()
    
    def gen_unlearning_influence(self):
        # N2C
        self.n2c = {}  # node -> list of communities
        for community, nodes in self.c2n.items():
            for node in nodes:
                if node not in self.n2c:
                    self.n2c[node] = []
                self.n2c[node].append(community)

        self.logger.info(f"Constructed node-to-community mapping with {len(self.n2c)} nodes.")

        # GEN UNLEARNING REQUEST
        unlearn_ratio = self.args['unlearn_ratio']
        all_nodes = list(self.n2c.keys())

        num_all_nodes = len(all_nodes)
        if unlearn_ratio > 1:
            num_unlearning_nodes = int(unlearn_ratio)
        elif 0 < unlearn_ratio <= 1:
            num_unlearning_nodes = int(unlearn_ratio * num_all_nodes)
        else:
            raise ValueError("Invalid unlearn_ratio value")

        unlearning_set = random.sample(all_nodes, num_unlearning_nodes)
        self.unlearning_set = unlearning_set

        self.logger.info(f"[Unlearning] Selected {num_unlearning_nodes} nodes for unlearning.")

        # Confirm Affected Communities
        unlearning_communities = set()
        for node in unlearning_set:
            if node in self.n2c:
                unlearning_communities.update(self.n2c[node])

        self.logger.info(f"[Unlearning] {len(unlearning_communities)} communities are affected.")

        # RECULCULATE
        features_communities = set(unlearning_communities)
        labels_communities = set(unlearning_communities)

        sim_communities_pairs = set()
        for (comm1, comm2), _ in self.sim.items():
            if comm1 in unlearning_communities or comm2 in unlearning_communities:
                sim_communities_pairs.add((comm1, comm2))

        self.logger.info(f"[Recalc] {len(features_communities)} feature communities, "
                         f"{len(labels_communities)} label communities, "
                         f"{len(sim_communities_pairs)} sim pairs need recalculation.")

        # DICT
        influence_dict = {
            'features_communities': features_communities,
            'labels_communities': labels_communities,
            'sim_communities_pairs': sim_communities_pairs,
            'unlearning_nodes': unlearning_set,
            'unlearning_communities': unlearning_communities
        }

        self.influence_dict = influence_dict
        self.logger.info("[Unlearning] Influence dictionary successfully generated and stored.")

        return influence_dict
    
    def calculate_threshold(self, distances):
        if len(distances) <= 1:
            return np.inf
        sorted_distances = np.sort(distances)
        gradients = np.gradient(sorted_distances)
        max_gradient_idx = np.argmax(gradients)
        threshold = sorted_distances[max_gradient_idx]
        return threshold

    def unlearning(self, influence_dict):
        features_communities = influence_dict['features_communities']
        labels_communities = influence_dict['labels_communities']
        sim_communities_pairs = influence_dict['sim_communities_pairs']

        new_feats = self.recalculate_features_pca(features_communities)

        new_labels = self.recalculate_labels_th1(labels_communities, new_feats)

        new_sim = self.recalculate_sim_rubost(sim_communities_pairs)

        updated_nucleus_data = {
            'new_feats': new_feats,
            'new_labels': new_labels,
        }
        self.data_store.save_nucleus_data(params=updated_nucleus_data, if_sim=0)
        self.data_store.save_nucleus_data(params=new_sim, if_sim=1)

        self.new_feats = new_feats
        self.new_labels = new_labels
        self.sim = new_sim

    def recalculate_features_pca(self, features_communities):
        in_feats = self.graph.ndata['feat'].shape[1]
        X = self.graph.ndata['feat'].numpy()
        np_new_feats = np.zeros((len(self.c2n), in_feats), dtype='float32')

        for community in features_communities:
            nodes = self.c2n[community]
            if len(nodes) <= 1 / self.args['n_components_ratio']:
                np_new_feats[community] = X[nodes].mean(axis=0)
            else:
                pca = PCA(n_components=max(1, int(np.ceil(len(nodes) * self.args['n_components_ratio']))))
                transformed_feats = pca.fit_transform(X[nodes].T)
                np_new_feats[community] = transformed_feats.mean(axis=1)

        return np_new_feats

    def recalculate_labels_th1(self, labels_communities, np_new_feats):
        Y = self.graph.ndata['label'].numpy()
        np_new_labels = np.zeros(len(self.c2n), dtype='int64')

        for community in labels_communities:
            nodes = np.array(self.c2n[community])
            community_features = self.graph.ndata['feat'].numpy()[nodes]
            distances = np.array([euclidean(np_new_feats[community], node_feat) for node_feat in community_features])
            threshold = self.calculate_threshold(distances)
            selected_nodes = nodes[distances <= threshold]

            if len(selected_nodes) > 0:
                np_new_labels[community] = mode(Y[selected_nodes])[0][0]
            else:
                np_new_labels[community] = mode(Y[nodes])[0][0]

        return np_new_labels

    def calculate_edge_counts(self, n2c, c2n):
        edge_counts = {}
        for cu in c2n.keys():
            for cv in c2n.keys():
                if cu != cv:
                    edge_counts[(cu, cv)] = 0

        src, dst = self.graph.edges()
        for u, v in zip(src.numpy(), dst.numpy()):
            cu_list = n2c[u] if isinstance(n2c[u], list) else [n2c[u]]
            cv_list = n2c[v] if isinstance(n2c[v], list) else [n2c[v]]
            for cu in cu_list:
                for cv in cv_list:
                    if cu != cv:
                        edge_counts[(cu, cv)] += 1
        return edge_counts

    def recalculate_sim_rubost(self, sim_communities_pairs):
        test_edge_method = self.args['test_edge_method']
        np_new_sim = {}

        community_set = {c: set(self.c2n[c]) for c in self.c2n}
        edge_counts = self.calculate_edge_counts(self.n2c, self.c2n)

        for (i, j) in sim_communities_pairs:
            if i in community_set and j in community_set:
                edge_count = edge_counts.get((i, j), 0)
                if edge_count > 0:
                    out_degree_A = sum(edge_counts.get((i, k), 0) for k in self.c2n if k != i)
                    in_degree_B = sum(edge_counts.get((k, j), 0) for k in self.c2n if k != j)

                    if out_degree_A > 0 and in_degree_B > 0:
                        if test_edge_method == 0:
                            robustness_A2B = (edge_count / np.sqrt(out_degree_A)) * (edge_count / np.sqrt(in_degree_B))
                        elif test_edge_method == 1:
                            robustness_A2B = np.log1p(edge_count) / (np.log1p(out_degree_A) + np.log1p(in_degree_B))
                        elif test_edge_method == 2:
                            log_edge = np.log1p(edge_count)
                            log_out = np.log1p(out_degree_A)
                            log_in = np.log1p(in_degree_B)
                            robustness_A2B = (log_edge / np.sqrt(log_out)) * (log_edge / np.sqrt(log_in))
                        else:
                            robustness_A2B = 0
                        np_new_sim[(i, j)] = robustness_A2B
                        np_new_sim[(j, i)] = robustness_A2B
                    else:
                        np_new_sim[(i, j)] = np_new_sim[(j, i)] = 0

        return np_new_sim
