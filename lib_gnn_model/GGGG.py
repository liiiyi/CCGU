import logging
import os

import torch
import torch.nn.functional as F
import dgl
from dgl.data import RedditDataset
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
import config

class GATNet(torch.nn.Module):
    def __init__(self, in_feats, n_classes):
        super(GATNet, self).__init__()
        self.layer1 = GATConv(in_feats, 8, num_heads=8, feat_drop=0.6, attn_drop=0.6)
        self.layer2 = GATConv(8 * 8, n_classes, num_heads=1, feat_drop=0.6, attn_drop=0.6)

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = F.elu(h.flatten(1))
        h = self.layer2(graph, h).mean(1)
        return F.log_softmax(h, dim=1)

class GNNBase:
    def __init__(self):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class GAT(GNNBase):
    def __init__(self, num_feats, num_classes, graph, features, labels, train_mask, test_mask, note=None):
        super(GAT, self).__init__()
        self.logger = logging.getLogger('gat')

        self.model = GATNet(num_feats, num_classes)
        self.graph = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.test_mask = test_mask

    def train_model(self, num_epoch=200):
        self.model.train()
        self.model.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.graph = self.graph.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.001)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))

            optimizer.zero_grad()
            output = self.model(self.graph, self.features)[self.train_mask]
            loss = F.nll_loss(output, self.labels[self.train_mask])
            loss.backward()
            optimizer.step()

            train_acc, test_acc = self.evaluate_model()
            self.logger.info('train acc: %s, test acc: %s' % (train_acc, test_acc))

        self.logger.info(f"Final F1 Score:{self.f1_score[-1]}")

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.graph = self.graph.to(self.device)
        self.f1_score = []

        logits = self.model(self.graph, self.features)
        pred_train = logits[self.train_mask].max(1)[1]
        train_acc = pred_train.eq(self.labels[self.train_mask]).sum().item() / self.train_mask.sum().item()

        pred_test = logits[self.test_mask].max(1)[1]
        test_acc = pred_test.eq(self.labels[self.test_mask]).sum().item() / self.test_mask.sum().item()
        f1 = self.compute_f1_score(logits, self.labels[self.test_mask])
        self.f1_score.append(f1)

        return train_acc, test_acc

    def compute_f1_score(self, logits, labels):
        pred = logits.max(1)[1]
        true_labels = labels
        f1 = f1_score(true_labels.cpu().numpy(), pred.cpu().numpy(), average='micro')
        return f1

if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset = RedditDataset(raw_dir='GIF-torch-main/' + config.DGL_PATH)
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']

    gat = GAT(features.shape[1], dataset.num_classes, graph, features, labels, train_mask, test_mask)
    gat.train_model()