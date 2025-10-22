import time
import logging
import torch
import dgl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import networkx as nx


class NIKM:
    def __init__(self, graph, embeddings, args):
        self.logger = logging.getLogger('NIKM')
        self.graph = graph.to(torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu'))
        self.args = args
        self.embeddings = embeddings
        self.copy_node_ratio = args.get('copy_node_ratio', 0.1)  # 默认值
        self.nearest_communities = args.get('nearest_communities', 3)  # 默认值
        self.embedding_path = args.get('embedding_path', 'embeddings.npy')

    def cluster_graph(self, embeddings):
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        """使用KMeans进行图聚类"""
        kmeans = KMeans(n_clusters=300, random_state=10).fit(embeddings_normalized)
        c2n = {i: np.where(kmeans.labels_ == i)[0].tolist() for i in range(kmeans.n_clusters)}
        return c2n

    def copy_node_requestions(self, c2n):
        """根据节点度信息进行节点复制"""
        self.logger.info('Copying nodes based on degree...')
        g = self.graph
        num_nodes = g.number_of_nodes()
        degrees = g.in_degrees() + g.out_degrees()
        top_nodes = torch.topk(degrees, int(num_nodes * self.copy_node_ratio)).indices.tolist()

        n2c = {node: [] for node in range(num_nodes)}
        for c, nodes in c2n.items():
            for node in nodes:
                n2c[node].append(c)

        for node in tqdm(top_nodes, desc="Copying nodes"):
            neighbors = g.successors(node).tolist() + g.predecessors(node).tolist()
            neighbors_degrees = {n: degrees[n].item() for n in neighbors}
            sorted_neighbors = sorted(neighbors_degrees.items(), key=lambda x: x[1], reverse=True)

            copied = 0
            for neighbor, _ in sorted_neighbors:
                if copied >= self.nearest_communities:
                    break
                neighbor_communities = n2c[neighbor]
                for community in neighbor_communities:
                    if len(c2n[community]) > 1:
                        c2n[community].append(node)
                        n2c[node].append(community)
                        copied += 1
                        break
                if copied >= self.nearest_communities:
                    break

        return c2n, n2c

    def copy_node_requestions_by_pagerank(self, c2n):
        """根据PageRank进行节点复制"""
        self.logger.info('Copying nodes based on PageRank...')
        g = self.graph.cpu().to_networkx().to_undirected()
        pagerank_scores = nx.pagerank(g)
        top_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:int(len(pagerank_scores) * self.copy_node_ratio)]

        n2c = {node: [] for node in range(self.graph.number_of_nodes())}
        for c, nodes in c2n.items():
            for node in nodes:
                n2c[node].append(c)

        for node in tqdm(top_nodes, desc="Copying nodes"):
            neighbors = list(g.neighbors(node))
            neighbors_pagerank = {n: pagerank_scores[n] for n in neighbors}
            sorted_neighbors = sorted(neighbors_pagerank.items(), key=lambda x: x[1], reverse=True)

            copied = 0
            for neighbor, _ in sorted_neighbors:
                if copied >= self.nearest_communities:
                    break
                neighbor_communities = n2c[neighbor]
                for community in neighbor_communities:
                    if len(c2n[community]) > 1:
                        c2n[community].append(node)
                        n2c[node].append(community)
                        copied += 1
                        break
                if copied >= self.nearest_communities:
                    break

        return c2n, n2c

    def run(self):
        start_time = time.time()
        embeddings = self.embeddings
        c2n = self.cluster_graph(embeddings)
        co_re = 'pagerank'
        if co_re == 'degree':
            c2n, n2c = self.copy_node_requestions(c2n)
        else:
            c2n, n2c = self.copy_node_requestions_by_pagerank(c2n)
        total_time = time.time() - start_time
        self.logger.info(f"Total time for computation: {total_time:.2f} seconds")
        return n2c, c2n, len(c2n), total_time