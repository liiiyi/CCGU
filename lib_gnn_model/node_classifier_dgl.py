import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from lib_gnn_model.GCN import GCNNet
from lib_gnn_model.GAT import GATNet
from lib_gnn_model.GraphSAGE import GraphSAGENet
import logging
from sklearn.metrics import f1_score, accuracy_score
import time

class NodeClassifierDGL(nn.Module):
    def __init__(self, num_feats, num_classes, args):
        super(NodeClassifierDGL, self).__init__()
        self.args = args
        self.logger = logging.getLogger('Node_classifier')
        model = self.args['target_model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Generating target model: {model}')

        if self.args['target_model'] == 'GCN':
            self.model = GCNNet(num_feats, num_classes).to(self.device)
        elif self.args['target_model'] == 'GAT':
            self.model = GATNet(num_feats, num_classes).to(self.device)
        elif self.args['target_model'] == 'SAGE':
            self.model = GraphSAGENet(num_feats, num_classes).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.args['target_model']}")

    def forward(self, x, g, edge_weight):
        return self.model(g, x, edge_weight)

    def reset_parameters(self):
        self.model.reset_parameters()