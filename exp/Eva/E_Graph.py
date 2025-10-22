import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class GraphAutoencoder(nn.Module):
    def __init__(self, in_feats, hidden_size, embedding_size):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.embedding = GraphConv(hidden_size, embedding_size, allow_zero_in_degree=True)
        self.decoder = GraphConv(embedding_size, in_feats, allow_zero_in_degree=True)

    def forward(self, g, features):
        h = F.relu(self.encoder(g, features))
        z = self.embedding(g, h)
        recon = self.decoder(g, z)
        return z, recon

class GraphRepresentativenessEvaluator:
    def __init__(self, original_graph, subgraph=None, community_nodes=None):
        self.original_graph = original_graph
        if subgraph is not None:
            self.subgraph = subgraph
        elif community_nodes is not None:
            self.subgraph = self.create_subgraph_from_community(community_nodes)
        else:
            raise ValueError("Either subgraph or community_nodes must be provided")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_subgraph_from_community(self, community_nodes):
        if isinstance(self.original_graph, dgl.DGLGraph):
            subgraph = self.original_graph.subgraph(community_nodes)
            return subgraph
        else:
            raise TypeError("Original graph must be a DGL graph")

    def compute_graph_embedding(self, graph, features, hidden_size=64, embedding_size=32, epochs=200):
        in_feats = features.shape[1]
        model = GraphAutoencoder(in_feats, hidden_size, embedding_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

        graph = graph.to(self.device)
        features = features.to(self.device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, recon = model(graph, features)
            loss = F.mse_loss(recon, features)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embedding, recon = model(graph, features)
        return embedding.detach().cpu().numpy(), recon.detach().cpu().numpy()

    def evaluate_representativeness(self, hidden_size=64, embedding_size=32, epochs=200):
        if isinstance(self.original_graph, dgl.DGLGraph):
            original_features = self.original_graph.ndata['feat']
        else:
            raise TypeError("Original graph must be a DGL graph with node features")

        if isinstance(self.subgraph, dgl.DGLGraph):
            subgraph_features = self.subgraph.ndata['feat']
        else:
            raise TypeError("Subgraph must be a DGL graph with node features")

        original_embedding, original_recon = self.compute_graph_embedding(self.original_graph, original_features, hidden_size, embedding_size, epochs)
        subgraph_embedding, subgraph_recon = self.compute_graph_embedding(self.subgraph, subgraph_features, hidden_size, embedding_size, epochs)

        # Graph Embedding Similarity
        similarity = cosine_similarity(original_embedding, subgraph_embedding.mean(axis=0, keepdims=True))
        representativeness_score = similarity.mean()

        # Information Retention
        info_retention = cosine_similarity(original_recon, subgraph_recon.mean(axis=0, keepdims=True)).mean()

        return representativeness_score, info_retention

    def compute_wl_kernel(self, g1, g2, h=1):
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_subgraph_hashes
        if not isinstance(g1, nx.Graph):
            g1 = dgl.to_networkx(g1.cpu())
        if not isinstance(g2, nx.Graph):
            g2 = dgl.to_networkx(g2.cpu())
        wl_kernel = weisfeiler_lehman_subgraph_hashes(g1, h) == weisfeiler_lehman_subgraph_hashes(g2, h)
        return wl_kernel

    def evaluate_all(self, hidden_size=64, embedding_size=32, epochs=200):
        rep_score, info_retention = self.evaluate_representativeness(hidden_size, embedding_size, epochs)
        # wl_kernel = self.compute_wl_kernel(self.original_graph, self.subgraph)
        return {
            "representativeness_score": rep_score,
            "information_retention": info_retention,
            # "wl_kernel": wl_kernel
        }