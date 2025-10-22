import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import logging
import time
from sklearn.metrics import f1_score, accuracy_score

class GCNNet(nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = dglnn.GraphConv(num_feats, 16)
        self.conv2 = dglnn.GraphConv(16, num_classes)
        self.dropout_rate = 0.5

    def forward(self, g, x, edge_weight=None):
        x = F.relu(self.conv1(g, x, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(g, x, edge_weight=edge_weight)
        return x