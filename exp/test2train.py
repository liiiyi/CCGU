import time

import torch

from exp.exp import Exp
import networkx as nx
import numpy as np
import logging
import dgl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from lib_gnn_model.node_classifier import NodeClassifier
from lib_gnn_model.node_classifier_dgl import NodeClassifierDGL
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import os
import random


class TrainModel(Exp):
    def __init__(self, args):
        super(TrainModel, self).__init__(args)
        self.nucleus_graph = None
        self.logger = logging.getLogger('ExpTrainNucleusGraph')
        self.load_data()

        num_nodes_original = self.graph.number_of_nodes()
        # self.test_mask_original = self.graph.ndata['test_mask']
        # labels_original = self.graph.ndata['label']

        regen_graph = True
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
        n2c = load_communities_data['n2c']
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
        train_mask_, val_mask_, test_mask_ = self.gen_masks(available_communities, self.num_communities,
                                                            test_size=test_size)

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['train_lr'],
                                     weight_decay=self.args['train_weight_decay'])
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
