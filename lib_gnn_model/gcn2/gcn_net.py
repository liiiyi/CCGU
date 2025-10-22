import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GCNNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_feats, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
