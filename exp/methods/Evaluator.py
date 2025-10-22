import dgl
import networkx as nx
import numpy as np
from collections import defaultdict


class CommunityEvaluator:
    def __init__(self, graph, c2n):
        self.graph = graph
        self.c2n = self._filter_valid_nodes(graph, c2n)
        self.G = self.graph.to_networkx().to_undirected()

    def _filter_valid_nodes(self, graph, c2n):
        valid_nodes = set(graph.nodes().numpy())
        filtered_c2n = {}
        for community, nodes in c2n.items():
            filtered_nodes = [node for node in nodes if node in valid_nodes]
            if filtered_nodes:
                filtered_c2n[community] = filtered_nodes
        return filtered_c2n

    def _compute_modularity_matrix(self):
        num_nodes = self.G.number_of_nodes()
        A = nx.adjacency_matrix(self.G).todense()
        degrees = np.array([self.G.degree(n) for n in self.G.nodes()])
        total_edges = np.sum(degrees) / 2
        Q = np.zeros((num_nodes, num_nodes))
        for i, nodes_i in enumerate(self.c2n.values()):
            for j, nodes_j in enumerate(self.c2n.values()):
                if i != j:
                    cut_size = sum(A[u, v] for u in nodes_i for v in nodes_j)
                    volume_i = sum(degrees[u] for u in nodes_i)
                    volume_j = sum(degrees[v] for v in nodes_j)
                    Q[i, j] = cut_size / total_edges - (volume_i * volume_j) / (4 * total_edges ** 2)
        modularity = np.sum(Q) / (2 * total_edges)
        return modularity

    def calculate_modularity(self):
        return self._compute_modularity_matrix()

    def calculate_average_clustering(self):
        # Calculate clustering coefficient for multi-graph
        clustering_coeffs = {}
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            k = len(neighbors)
            if k < 2:
                clustering_coeffs[node] = 0.0
                continue
            edges_between_neighbors = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if self.G.has_edge(neighbors[i], neighbors[j]):
                        edges_between_neighbors += 1
            max_edges = k * (k - 1) / 2
            clustering_coeffs[node] = edges_between_neighbors / max_edges

        avg_clustering = np.mean(list(clustering_coeffs.values()))
        return avg_clustering

    def calculate_edge_density(self):
        edge_counts = self._calculate_edge_counts()
        total_possible_edges = sum([len(nodes) * (len(nodes) - 1) / 2 for nodes in self.c2n.values()])
        total_edges = sum(edge_counts.values()) / 2  # Each edge counted twice
        edge_density = total_edges / total_possible_edges
        return edge_density

    def _calculate_edge_counts(self):
        edge_counts = defaultdict(int)
        src, dst = self.graph.edges()
        src = src.numpy()
        dst = dst.numpy()
        for u, v in zip(src, dst):
            for comm_u in self.c2n.values():
                if u in comm_u:
                    for comm_v in self.c2n.values():
                        if v in comm_v and comm_u != comm_v:
                            edge_counts[(tuple(comm_u), tuple(comm_v))] += 1
        return edge_counts

    def calculate_conductance(self):
        conductance_values = []
        total_edges = self.G.number_of_edges()
        for nodes in self.c2n.values():
            cut_size = sum(1 for u in nodes for v in self.G.neighbors(u) if v not in nodes) / 2
            volume = sum(self.G.degree(n) for n in nodes)
            if volume > 0:
                conductance = cut_size / min(volume, 2 * total_edges - volume)
            else:
                conductance = 0
            conductance_values.append(conductance)
        return np.mean(conductance_values)

    def evaluate(self):
        modularity = self.calculate_modularity()
        avg_clustering = self.calculate_average_clustering()
        edge_density = self.calculate_edge_density()
        conductance = self.calculate_conductance()
        return {
            'modularity': modularity,
            'avg_clustering': avg_clustering,
            'edge_density': edge_density,
            'conductance': conductance
        }