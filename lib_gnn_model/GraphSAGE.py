import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

class GraphSAGENet(nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GraphSAGENet, self).__init__()
        self.conv1 = dglnn.SAGEConv(num_feats, 16, 'mean')
        self.conv2 = dglnn.SAGEConv(16, num_classes, 'mean')
        self.dropout_rate = 0.5

    def forward(self, g, x, edge_weight=None):
        x = F.relu(self.conv1(g, x, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(g, x, edge_weight=edge_weight)
        return x