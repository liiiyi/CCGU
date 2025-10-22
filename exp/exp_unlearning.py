import random
import time

import torch

from exp.exp import Exp
import networkx as nx
import numpy as np
import logging
import dgl
import tensorflow as tf
from scipy.spatial.distance import euclidean
import math
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from lib_gnn_model.node_classifier import NodeClassifier
from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import os
import config


class TrainModel(Exp):
    def __init__(self, args):
        super(TrainModel, self).__init__(args)
        self.nucleus_graph = None
        self.logger = logging.getLogger('ExpTrainNucleusGraph')
        self.load_data()

        set_file = config.COM_PATH + self.args['dataset_name'] + "/" + self.args['partition'] + '_select_n'
        selected_nodes = self.data_store.save_param(address=set_file)
        inf = self.gen_unlearning_influence(selected_nodes)
        self.unlearning(inf)


        regen_graph = False
        if os.path.exists(self.data_store.nugraph_file) and not regen_graph:
            self.nucleus_graph = self.data_store.load_nucleus_graph()
            time_gen_graph = 'None'
        else:
            start_time = time.time()
            self.gen_nucleus_graph(self.c2n)
            time_gen_graph = time.time() - start_time
            self.data_store.save_nucleus_graph(self.nucleus_graph)



        self.deter_model()
        self.train_model()

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

        self.logger.info(f'Time Consumption of Generating nucleus graph: {time_gen_graph} seconds')
        self.logger.info(f'Training time: {self.train_time} seconds')


    def load_data(self):
        self.logger.info('loading dataset')
        if self.args['dgl_data']:
            self.graph = self.data_store.load_graph_from_dgl()
            self.num_nodes = self.graph.number_of_nodes()

        load_communities_data = self.data_store.load_communities_info()
        self.n2c = load_communities_data['n2c']
        self.c2n = load_communities_data['c2n']
        self.num_communities = load_communities_data['num_c']

        load_nucleus_data = self.data_store.load_nucleus_data(if_sim=0)
        self.new_feats = load_nucleus_data['new_feats']
        self.new_labels = load_nucleus_data['new_labels']
        # self.train_mask = load_nucleus_data['train_mask']
        self.val_mask = load_nucleus_data['val_mask']
        self.test_mask_original = load_nucleus_data['test_mask']
        self.sim = self.data_store.load_nucleus_data(if_sim=1)


    def gen_nucleus_graph(self, c2n):
        # calculate th automatically
        # --code--
        th_sim2edge = self.args['th_sim2edge']
        if th_sim2edge >= 0:
            th = self.calculate_sim2edge_threshold()
            info = 'automatically' if th_sim2edge == 0 else f'by th_sim2edge: {th_sim2edge}'
            self.logger.info(f'Threshold(sim2edge, calculate {info}): {th}')
        else:
            th = 0
        self.logger.info('generating nucleus graph')
        nucleus_graph = nx.DiGraph()
        nucleus_graph.add_nodes_from(range(self.num_communities))

        nucleus_graph.add_edges_from([edge for edge in self.sim if self.sim[edge] >= th])
        nucleus_graph = dgl.convert.from_networkx(nucleus_graph)

        if self.args['use_edge_weight']:
            weights = torch.tensor([self.sim[edge] for edge in self.sim if self.sim[edge] >= th], dtype=torch.float32)
            nucleus_graph.edata['weight'] = weights


        device = torch.device(f'cuda:{self.args["cuda"]}' if torch.cuda.is_available() else 'cpu')

        # 将图转移到设备上
        nucleus_graph = nucleus_graph.to(device)

        # 设置节点特征和标签
        nucleus_graph.ndata['feat'] = torch.tensor(self.new_feats, dtype=torch.float32).to(device)
        nucleus_graph.ndata['label'] = torch.tensor(self.new_labels, dtype=torch.float32).to(device)

        # 计算训练、验证和测试集的掩码
        train_mask, val_mask, test_mask = self.calculate_tvt_split_nucleus(c2n)
        nucleus_graph.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool).to(device)
        nucleus_graph.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool).to(device)
        nucleus_graph.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool).to(device)

        degrees = nucleus_graph.in_degrees().to(torch.device('cpu')).numpy()
        zero_degree_nodes = [node for node, degree in enumerate(degrees) if degree == 0]
        nucleus_graph = dgl.remove_nodes(nucleus_graph, zero_degree_nodes)

        self.nucleus_graph = nucleus_graph
        self.logger.info(nucleus_graph)


    def gen_masks(self, X, n_data, test_size):
        train, other, _, _ = train_test_split(X, X, test_size=test_size)
        val, test, _, _ = train_test_split(other, other, test_size=0.5)
        train_mask = np.array([False] * n_data)
        val_mask = np.array([False] * n_data)
        test_mask = np.array([False] * n_data)
        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True

        return (
            train_mask,
            val_mask,
            test_mask
        )

    def calculate_tvt_split_nucleus(self, c2n):
        test_size = self.args['test_ratio']
        # Applicable only to graph partitioning methods without isolation
        self.logger.info('calculating train/val/test mask')
        test_communities = [community for community in c2n if
                            len(c2n[community]) == 1 and tf.reduce_sum(
                                tf.gather(tf.cast(self.test_mask_original, dtype=tf.float32),
                                          c2n[community])).numpy().item() > 0.5]
        train_communities = [community for community in c2n if len(c2n[community]) > 1]
        available_communities = list(set(range(self.num_communities)).difference(set(test_communities)))
        self.logger.info(
            f'Num of Communities: {self.num_communities}, Test sets meeting specific criteria: {len(test_communities)}, Ava train Sets: {len(available_communities)}')
        self.logger.info(f'Test ratio: {test_size}')
        train_mask_, val_mask_, test_mask_ = self.gen_masks(available_communities, self.num_communities, test_size=test_size)

        if test_communities:
            for community in test_communities:
                test_mask_[community] = True

        self.logger.info(f"Training set size: {train_mask_.sum()}")
        self.logger.info(f"Validation set size: {val_mask_.sum()}")
        self.logger.info(f"Test set size: {test_mask_.sum()}")

        return (train_mask_, val_mask_, test_mask_)



    def calculate_sim2edge_threshold(self):
        """
        Calculate the similarity threshold based on gradient change and control parameter.
        :param sim: list of similarity values
        :param control: percentage value to control the range of selection for the threshold.
                        If control is 0, the threshold will be automatically calculated.
        :return: calculated threshold
        """
        sim = list(self.sim.values())
        control = self.args['th_sim2edge']
        # Sort the similarity values
        sim.sort()

        # Calculate all possible gradients
        gradients = np.diff(sim)

        if control > 0:
            # Get the number of elements to consider based on control percentage
            num_elements_to_consider = int(len(sim) * control)

            # Select the highest gradient change within the controlled range
            selected_gradient_idx = np.argmax(gradients[-num_elements_to_consider:])

            # Calculate the threshold using the selected gradient index
            threshold = sim[-num_elements_to_consider + selected_gradient_idx]
        else:
            # Automatically calculate the threshold without control
            selected_gradient_idx = np.argmax(gradients)
            threshold = sim[selected_gradient_idx]

        return threshold

    def deter_model(self):
        self.device = torch.device(f'cuda:{self.args["cuda"]}' if torch.cuda.is_available() else 'cpu')

        self.num_feats = self.nucleus_graph.ndata['feat'].shape[1]
        self.num_classes = len(torch.unique(self.nucleus_graph.ndata['label']))

        self.model = NodeClassifierDGL(self.num_feats, self.num_classes, self.args)
        self.graph = self.nucleus_graph.to(self.device)
        self.features = self.graph.ndata['feat'].to(self.device)
        self.labels = self.graph.ndata['label'].to(self.device)
        self.train_mask = self.graph.ndata['train_mask'].to(self.device)
        self.val_mask = self.graph.ndata['val_mask'].to(self.device)
        self.test_mask = self.graph.ndata['test_mask'].to(self.device)

    def train_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['train_lr'], weight_decay=self.args['train_weight_decay'])
        best_val_score = 0
        best_model = None
        start_time = time.time()

        for epoch in range(self.args['num_epochs']):
            self.model.train()
            logits = self.model(self.features, self.graph, self.graph.edata.get('weight', None))
            loss = F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_score, test_score = self._evaluate_model()
            self.logger.info(f'Epoch {epoch}, Loss: {loss.item()}, Val F1: {val_score}, Test F1: {test_score}')

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = self.model.state_dict()

        self.train_time = time.time() - start_time


        if best_model:
            self.model.load_state_dict(best_model)
            # self.data_store.save_model(self.model)
            self.logger.info('Best model saved.')

        final_val_f1, final_val_acc, final_test_f1, final_test_acc = self._evaluate_model(final=True)
        self.logger.info(f'Final Validation F1: {final_val_f1}, Final Validation Acc: {final_val_acc}')
        self.logger.info(f'Final Test F1: {final_test_f1}, Final Test Acc: {final_test_acc}')

        return final_val_f1, final_val_acc, final_test_f1, final_test_acc

    def _evaluate_model(self, final=False):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.features, self.graph, self.graph.edata.get('weight', None))
            val_logits = logits[self.val_mask]
            test_logits = logits[self.test_mask]

            val_pred = val_logits.max(1)[1]
            test_pred = test_logits.max(1)[1]

            val_f1 = f1_score(self.labels[self.val_mask].cpu(), val_pred.cpu(), average='macro')
            test_f1 = f1_score(self.labels[self.test_mask].cpu(), test_pred.cpu(), average='macro')

            val_acc = accuracy_score(self.labels[self.val_mask].cpu(), val_pred.cpu())
            test_acc = accuracy_score(self.labels[self.test_mask].cpu(), test_pred.cpu())

            if not final:
                self.logger.info(f'Validation F1: {val_f1}, Validation Acc: {val_acc}')
                self.logger.info(f'Test F1: {test_f1}, Test Acc: {test_acc}')

        return (val_f1, val_acc, test_f1, test_acc) if final else (val_f1, test_f1)

    def gen_unlearning_influence(self, selected_nodes):
        # 1. 根据dgl图中训练集的社区号确定所有处于训练集社区的节点号v_trainset
        v_trainset = set()
        for community, nodes in self.c2n.items():
            if community in self.nucleus_graph.nodes().data['train_mask']:
                v_trainset.update(nodes)

        v_trainset = list(v_trainset)

        # 2. 根据self.args['unlearn_ratio']生成遗忘请求
        unlearn_ratio = self.args['unlearn_ratio']
        num_trainset_nodes = len(v_trainset)

        if unlearn_ratio > 1:
            # 如果unlearn_ratio是一个大于1的整数
            num_unlearning_nodes = int(unlearn_ratio)
        elif 0 < unlearn_ratio <= 1:
            # 如果unlearn_ratio是一个处于1和0之间的浮点数
            num_unlearning_nodes = int(unlearn_ratio * num_trainset_nodes)
        else:
            raise ValueError("Invalid unlearn_ratio value")

        # 随机选择unlearning_set
        unlearning_set = random.sample(v_trainset, num_unlearning_nodes)

        # 输出产生的遗忘请求的节点数量
        self.logger.info(f"Number of nodes in unlearning request: {num_unlearning_nodes}")

        # 3. 获取遗忘请求节点所在的所有社区
        unlearning_communities = set()
        for community, nodes in self.c2n.items():
            if any(node in unlearning_set for node in nodes):
                unlearning_communities.add(community)

        # 4. 计算需要重计算的社区编号和社区对
        features_communities = set()
        labels_communities = set()
        sim_communities_pairs = set()

        for community in unlearning_communities:
            features_communities.add(community)

            # 检查是否需要重计算标签
            if any(node in selected_nodes for node in self.c2n.get(community, [])):
                labels_communities.add(community)

        # 计算需要重计算sim的社区对
        for (comm1, comm2), _ in self.sim.items():
            if comm1 in unlearning_communities or comm2 in unlearning_communities:
                sim_communities_pairs.add((comm1, comm2))

        # 记录受影响的内容信息
        self.logger.info(f"Communities needing feature recalculation: {features_communities}")
        self.logger.info(f"Communities needing label recalculation: {labels_communities}")
        self.logger.info(f"Community pairs needing sim recalculation: {sim_communities_pairs}")

        # 5. 返回受影响的内容
        influence_dict = {
            'features_communities': features_communities,
            'labels_communities': labels_communities,
            'sim_communities_pairs': sim_communities_pairs
        }

        return influence_dict

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

    def unlearning(self, influence_dict):
        """
        Update features, labels, and similarity calculations for the affected communities.

        :param influence_dict: dict, contains information about communities needing recalculations
        """
        # Extract affected communities from the influence_dict
        features_communities = influence_dict['features_communities']
        labels_communities = influence_dict['labels_communities']
        sim_communities_pairs = influence_dict['sim_communities_pairs']

        # 1. Recalculate features for affected communities
        new_feats = self.recalculate_features_pca(features_communities)

        # 2. Recalculate labels for affected communities
        new_labels = self.recalculate_labels_th1(labels_communities, new_feats)

        # 3. Recalculate similarity for affected community pairs
        new_sim = self.recalculate_sim_rubost(sim_communities_pairs)


        # Save the updated results
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
        """Recalculate features using PCA for specified communities."""
        in_feats = self.graph.ndata['feat'].shape[1]
        X = self.graph.ndata['feat'].numpy()
        np_new_feats = np.zeros((len(features_communities), in_feats), dtype='float32')

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
        """Recalculate labels using threshold-based majority vote for specified communities."""
        Y = self.graph.ndata['label'].numpy()
        np_new_labels = np.zeros(len(labels_communities), dtype='int64')

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

    def recalculate_sim_rubost(self, sim_communities_pairs):
        """
        Recalculate similarity scores between pairs of communities using edge robustness methods.
        Update and return the new similarity for the affected community pairs.

        Parameters:
            sim_communities_pairs: list of tuples, where each tuple contains a pair of communities (i, j)
                                   whose similarity needs to be recalculated.
            community_set: dict, where each key is a community, and each value is a set of its member nodes.

        Returns:
            np_new_sim: dict, updated similarity values for the affected community pairs.
        """
        test_edge_method = self.args['test_edge_method']
        np_new_sim = {}

        community_set = {}

        edge_counts = self.calculate_edge_counts(self.n2c, self.c2n)

        # Calculate community sets
        for community in self.c2n:
            community_set[community] = set(self.c2n[community])

        for (i, j) in sim_communities_pairs:
            if i in community_set and j in community_set:
                edge_count = edge_counts.get((i, j), 0)
                if edge_count > 0:
                    out_degree_A = sum(edge_counts.get((i, k), 0) for k in range(len(self.c2n)) if k != i)
                    in_degree_B = sum(edge_counts.get((k, j), 0) for k in range(len(self.c2n)) if k != j)

                    if out_degree_A > 0 and in_degree_B > 0:
                        if test_edge_method == 0:
                            # Method 1: (A到B的边数/A的出度) * (A到B的边数/B的入度)
                            robustness_A2B = (edge_count / np.sqrt(out_degree_A)) * (edge_count / np.sqrt(in_degree_B))
                        elif test_edge_method == 1:
                            # Method 2: log(1 + A到B的边数) / (log(1 + A的出度) + log(1 + B的入度))
                            robustness_A2B = np.log1p(edge_count) / (np.log1p(out_degree_A) + np.log1p(in_degree_B))
                        elif test_edge_method == 2:
                            # Method 3: (log(1 + 边数量_A_to_B) / sqrt(log(1 + 社区A的出度))) * (log(1 + 边数量_A_to_B) / sqrt(log(1 + 社区B的入度)))
                            log_edge_count = np.log1p(edge_count)
                            log_out_degree_A = np.log1p(out_degree_A)
                            log_in_degree_B = np.log1p(in_degree_B)
                            robustness_A2B = (log_edge_count / np.sqrt(log_out_degree_A)) * (
                                        log_edge_count / np.sqrt(log_in_degree_B))
                        elif test_edge_method == 3:
                            robustness_A2B = 0
                        else:
                            raise ValueError("Invalid test_edge_method. Choose from 0, 1, 2.")

                        # Update similarity for the community pair (i, j)
                        if robustness_A2B != 0:
                            np_new_sim[(i, j)] = robustness_A2B
                            np_new_sim[(j, i)] = robustness_A2B

                    else:
                        # If degrees are zero, set similarity to zero
                        np_new_sim[(i, j)] = 0
                        np_new_sim[(j, i)] = 0

        return np_new_sim
