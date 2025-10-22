import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EdgeWeightOptimizer(nn.Module):
    def __init__(self, num_communities, initial_edge_counts, new_feats, args, alpha=1.0, beta=1.0):
        super(EdgeWeightOptimizer, self).__init__()
        self.device = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.edge_weights = nn.Parameter(torch.zeros(num_communities, num_communities).to(self.device))
        self.initial_edge_counts = initial_edge_counts
        # self.new_feats = new_feats.to(self.device)
        self.new_feats = torch.tensor(new_feats, device=self.device, dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta


        # 计算社区之间的节点特征相似度
        self.similarity_matrix = self.calculate_similarity_matrix(self.new_feats).to(self.device)

    def calculate_similarity_matrix(self, new_feats):
        similarity_matrix = torch.zeros(len(new_feats), len(new_feats)).to(self.device)
        for i in range(len(new_feats)):
            for j in range(len(new_feats)):
                if i != j:
                    similarity_matrix[i, j] = F.cosine_similarity(new_feats[i], new_feats[j], dim=0)
        return similarity_matrix

    def forward(self):
        edge_loss = 0.0
        similarity_loss = 0.0

        for (u, v), count in self.initial_edge_counts.items():
            edge_loss += (self.edge_weights[u, v] - count) ** 2
            similarity_loss += (self.edge_weights[u, v] - self.similarity_matrix[u, v]) ** 2

        total_loss = self.alpha * edge_loss + self.beta * similarity_loss
        return total_loss

    def calculate_sim(self):
        trained_edge_weights = self.edge_weights.detach().cpu().numpy()
        sim = {}
        for i in range(trained_edge_weights.shape[0]):
            for j in range(trained_edge_weights.shape[1]):
                if trained_edge_weights[i, j] != 0:
                    sim[(i, j)] = trained_edge_weights[i, j]
        return sim