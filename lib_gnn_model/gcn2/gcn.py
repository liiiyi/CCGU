import os
import logging

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.gcn.gcn_net import GCNNet
import config


class GCN(GNNBase):
    def __init__(self, num_feats, num_classes, data=None, lr=0.01, weight_decay=5e-4):
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCNNet(num_feats, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.data = data
        self.logger = logging.getLogger('GCN')

    def train_model(self, epochs=100):
        self.model.train()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            train_acc = self.evaluate(self.data.train_mask)
            test_acc = self.evaluate(self.data.test_mask)
            self.logger.info(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            pred = out[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc


if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    gcn = GCN(dataset.num_features, dataset.num_classes, data)
    gcn.train_model()