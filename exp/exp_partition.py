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
from instruments import Instruments
from exp.methods.CCD import CentroidCommunityDetection as CCD
from exp.methods.Infomap import Infomap

class GraphCommunityPartition(Exp):
    def __init__(self, args):
        super(GraphCommunityPartition, self).__init__(args)
        self.instrument = Instruments
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
        self.logger.info(f'Time Consumption of Calculation: {calculate_time}')

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

    def gen_community(self):
        #TODO: if exsit, load
        if os.path.exists(self.data_store.community_file):
            self.logger.info("Community file exists. Loading the file.")
            com_data = self.data_store.load_communities_info()
            n2c = com_data['n2c']
            c2n = com_data['c2n']
            num_communities = com_data['num_c']

            self.time_partition = 'Partition File is already exist.'
        else:
            if self.args['partition'] == 'slpa':
                slpa = SLPA(self.graph, max_iterations=50, threshold=0.2, max_communities_per_node=5)
                n2c, c2n, num_communities, self.time_partition = slpa.SLPA_partition()
            elif self.args['partition'] == 'oslom':
                oslom = OSLOM(self.graph, args=self.args)
                n2c, c2n, num_communities, self.time_partition = oslom.OSLOM_partition()
            elif self.args['partition'] == 'infomap':
                infomap = Infomap(self.graph, args=self.args)
                n2c, c2n, num_communities, self.time_partition = infomap.partition()
            elif self.args['partition'] == 'test':
                ccd = CCD(self.graph, args=self.args)
                n2c, c2n, self.time_partition = ccd.run()
                num_communities = len(c2n)
            else:
                self.logger.info("Error Method Name.")
                return 1

            com_data = {
                'n2c': n2c,
                'c2n': c2n,
                'num_c': num_communities
            }
            self.data_store.save_communities_info(params=com_data)

        self._check_communities(c2n)

        return n2c, c2n, num_communities

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

        start_time = time.time()

        n2c, c2n, num_communities = self.gen_community()

        if self.aggregate_feat_info == 'pca':
            new_feats = self.aggregate_features_pca(c2n)
        elif self.aggregate_feat_info == 'mean':
            new_feats = self.aggregate_features_mean(c2n)
        else:
            self.logger.info("Error param")
            return 1

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


        if self.aggregate_e_info == 'jaccard':
            sim = self.calculate_sim(n2c, c2n)
        elif self.aggregate_e_info == 'rubost':
            sim = self.calculate_nucleus_sim(n2c, c2n, n_communities=num_communities)
        elif self.aggregate_e_info == 'opt':
            sim = self.calculate_optimizer_sim(n2c, c2n, new_feats)
        else:
            self.logger.info("Error param")
            return 1
        self.logger.info(f'{len(sim)} pairs are found.')

        # sim = self.instrument.smooth_edge_weights(sim, threshold_weight=0.5)

        # save
        self.data_store.save_nucleus_data(params=sim, if_sim=1)

        return time.time() - start_time

    def generate_masks(self, num, test_size=0.2):
        """
        Param
        num: number of communities
        """
        X = np.array(range(num))
        train, other, _, _ = train_test_split(X, X, test_size=test_size)
        val, test, _, _ = train_test_split(X[other], X[other], test_size=0.5)
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

            feature_robustness[community] = len(nodes) / distances.sum()

        # instru = Instruments
        # smoothed_robustness = instru.smooth_edge_weights(feature_robustness, threshold_weight=0.5)

        if self.args['use_feat_rb']:
            for community, robustness in feature_robustness.items():
                robustness = 0.1 * np.exp(-robustness) + 1.0
                np_new_feats[community] *= robustness

        self.logger.info('New shapes for feat are generated by PCA.')

        return np_new_feats

    def calculate_threshold(self, distances):
        if len(distances) <= 1:
            return np.inf
        # Sort distances
        sorted_distances = np.sort(distances)
        # Compute gradient
        gradients = np.gradient(sorted_distances)
        # Optional: Plot gradients

        # self.plot_gradients(sorted_distances, gradients)

        # Find the index with the maximum gradient
        max_gradient_idx = np.argmax(gradients)
        # Use the distance at this index as the threshold
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
        Y = self.graph.ndata['label'].numpy()
        X = self.graph.ndata['feat'].numpy()
        np_new_labels = np.zeros(len(c2n), dtype='int64')

        # Set to store all selected nodes across iterations
        selected_nodes_set = set()

        for community, nodes in tqdm(c2n.items(), desc="Aggregating Labels by th"):
            nodes = np.array(nodes)
            community_features = X[nodes]
            distances = np.array([euclidean(np_new_feats[community], node_feat) for node_feat in community_features])

            # Calculate the threshold automatically
            threshold = self.calculate_threshold(distances)

            # Select nodes with distance less than the threshold
            selected_nodes = nodes[distances <= threshold]

            # Add selected nodes to the set
            selected_nodes_set.update(selected_nodes)

            if len(selected_nodes) > 0:
                np_new_labels[community] = mode(Y[selected_nodes])[0][0]
            else:
                # If no nodes are selected, use the original majority vote for robustness
                np_new_labels[community] = mode(Y[nodes])[0][0]

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
            largest_cluster_label = mode(labels)[0][0]
            largest_cluster_indices = [node for node, label in zip(nodes, labels) if label == largest_cluster_label]

            if len(largest_cluster_indices) > 0:
                np_new_labels[community] = mode(Y[largest_cluster_indices])[0][0]
            else:
                np_new_labels[community] = mode(Y[nodes])[0][0]

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
                np_new_labels[community] = mode(Y[selected_nodes].cpu().numpy())[0][0]
            else:
                np_new_labels[community] = mode(Y[nodes].cpu().numpy())[0][0]

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
            np_new_labels[community] = mode(Y[nodes])[0][0]

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
        edge_counts = {}

        # Initialize edge_counts with zeros
        for community_u in c2n.keys():
            for community_v in c2n.keys():
                if community_u != community_v:
                    edge_counts[(community_u, community_v)] = 0

        # Iterate through all edges in the graph
        src, dst = self.graph.edges()
        for u, v in zip(src.numpy(), dst.numpy()):
            communities_u = n2c[u] if isinstance(n2c[u], list) else [n2c[u]]
            communities_v = n2c[v] if isinstance(n2c[v], list) else [n2c[v]]

            for community_u in communities_u:
                for community_v in communities_v:
                    if community_u != community_v:
                        edge_counts[(community_u, community_v)] += 1

        return edge_counts

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

        test_edge_method = self.args['test_edge_method']

        # sim[(A,B)] := similarity between community A and B
        sim = {}
        # community_set: dictionary
        # key: community
        # val: set of its member nodes
        community_set = {}

        edge_counts = self.calculate_edge_counts(n2c, c2n)

        # Calculate community sets
        for community in c2n:
            community_set[community] = set(c2n[community])

        for i in range(n_communities):
            for j in range(i + 1, n_communities):
                if i in community_set and j in community_set:
                    edge_count_bet_ij = edge_counts.get((i, j), 0)
                    if edge_count_bet_ij > 0:
                        union_size_val = len(community_set[i].union(community_set[j]))
                        jaccard_sim = edge_count_bet_ij / union_size_val

                        edge_count = edge_counts.get((i, j), 0)
                        out_degree_A = sum(edge_counts.get((i, k), 0) for k in range(n_communities) if k != i)
                        in_degree_B = sum(edge_counts.get((k, j), 0) for k in range(n_communities) if k != j)


                        if out_degree_A > 0 and in_degree_B > 0:
                            if test_edge_method == 0:
                                robustness_A2B = (edge_count / math.sqrt(out_degree_A)) * (edge_count / math.sqrt(in_degree_B))
                            elif test_edge_method == 1:
                                robustness_A2B = log(1 + edge_count) / (log(1 + out_degree_A) + log(1 + in_degree_B))
                            elif test_edge_method == 2:
                                log_edge_count = math.log1p(edge_count)
                                log_out_degree_A = math.log1p(out_degree_A)
                                log_in_degree_B = math.log1p(in_degree_B)
                                robustness_A2B = (log_edge_count / math.sqrt(log_out_degree_A)) * (log_edge_count / math.sqrt(log_in_degree_B))
                            elif test_edge_method == 3:
                                robustness_A2B = 0
                            else:
                                raise ValueError("Invalid test_edge_method. Choose from 0, 1, 2.")

                            # print(f"({i}, {j}): {robustness_A2B}, {jaccard_sim}")

                            final_score = robustness_A2B + jaccard_sim

                            if final_score != 0:
                                sim[(i, j)] = final_score
                                sim[(j, i)] = final_score

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




