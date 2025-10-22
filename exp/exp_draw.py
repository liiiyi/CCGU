import math

from exp.exp import Exp
import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import dgl
from exp.methods.Evaluator import CommunityEvaluator as CE
import pickle
import config
from collections import Counter

class ExpDraw(Exp):
    def __init__(self, args):
        super(ExpDraw, self).__init__(args)
        self.logger = logging.getLogger('ExpEvaluate Class')
        self.args = args
        self.load_data()
        # self.draw()
        # self.draw_dgl_graph(self.graph, self.c2n)
        # self.evaluate_c()
        a = 0
        if a == 1:
            g = self.data_store.load_nucleus_graph()
        else:
            g= self.graph
        self.analyze_class_distribution(g)

    def analyze_class_distribution(self, g):

        labels = g.ndata['label'].cpu().numpy()
        label_counts = Counter(labels)
        total_nodes = len(labels)

        class_probs = np.array([count / total_nodes for count in label_counts.values()])
        num_classes = len(class_probs)

        entropy = -np.sum(class_probs * np.log(class_probs))
        max_entropy = math.log((num_classes))

        imbalance_ratio = np.min(class_probs) / np.max(class_probs)

        gini_coeff = np.sum([abs(p1 - p2) for p1 in class_probs for p2 in class_probs]) / (2 * num_classes * np.sum(class_probs))

        self.logger.info(f"Total Nodes: {total_nodes}")
        self.logger.info(f"Class Counts: {label_counts}")
        self.logger.info(f"Class Probabilities: {class_probs}")
        self.logger.info(f"Entropy: {entropy:.4f} (Max En: {max_entropy:.4f}")
        self.logger.info(f"Imbalance Ratio: {imbalance_ratio}")
        self.logger.info(f"Gini: {gini_coeff:.4f}")







    def load_data(self):
        # load_communities_data = self.data_store.load_communities_info()
        # self.c2n = load_communities_data['c2n']
        self.graph = self.data_store.load_graph_from_dgl()
        # address = config.COM_PATH + 'community_' + self.args['dataset_name']
        # self.c2n = pickle.load(open(address, 'rb'))

    def draw(self):
        c2n = self.c2n
        src, dst = self.graph.edges()
        edges = list(zip(src.numpy(), dst.numpy()))

        # 创建NetworkX图
        G = nx.Graph()
        G.add_edges_from(edges)

        # 添加节点到图中，并记录每个节点所属的社区
        node_community = {}
        for community, nodes in c2n.items():
            for node in nodes:
                node_community[node] = community

        # 获取社区数量并生成颜色映射
        communities = list(c2n.keys())
        num_communities = len(communities)
        colors = cm.get_cmap('hsv', num_communities)  # 使用hsv颜色图，颜色更多

        # 分配节点颜色
        node_color_list = [
            colors(communities.index(node_community[node])) if node in node_community else colors(num_communities) for
            node in G.nodes]

        # 绘制图形
        plt.figure(figsize=(14, 12))

        # 使用其他布局方法，比如Fruchterman-Reingold force-directed algorithm
        pos = nx.spectral_layout(G)  # 使用spring布局进行可视化

        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=1)  # 缩小节点大小
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')

        # 隐藏节点标号
        nx.draw_networkx_labels(G, pos, labels={})

        # 添加边框和背景
        ax = plt.gca()
        ax.set_facecolor('white')  # 设置背景颜色
        ax.collections[0].set_edgecolor("#444444")  # 设置边颜色

        plt.title("Community Distribution in Graph")
        plt.show()

    def draw_dgl_graph(self, graph, c2n):
        # Convert DGL graph to NetworkX graph
        nx_graph = graph.to_networkx().to_undirected()

        # Apply spectral layout
        pos = nx.spectral_layout(nx_graph)

        # Get a list of colors
        num_communities = len(c2n)
        colors = plt.cm.tab20(range(num_communities))

        # Assign a color to each community
        node_colors = {}
        for community, nodes in c2n.items():
            color = colors[community % len(colors)]
            for node in nodes:
                node_colors[node] = color

        # Draw nodes with color based on community
        for community, nodes in c2n.items():
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodes, node_color=[node_colors[node] for node in nodes],
                                   node_size=1)

        # Draw edges with different colors for intra-community and inter-community edges
        intra_edges = []
        inter_edges = []
        for u, v in nx_graph.edges():
            if any(u in nodes and v in nodes for nodes in c2n.values()):
                intra_edges.append((u, v))
            else:
                inter_edges.append((u, v))

        nx.draw_networkx_edges(nx_graph, pos, edgelist=intra_edges, edge_color=[node_colors[u] for u, v in intra_edges],
                               alpha=0.5)
        nx.draw_networkx_edges(nx_graph, pos, edgelist=inter_edges, edge_color='black', alpha=0.5)

        plt.axis('off')
        plt.show()

    def evaluate_c(self):
        ce = CE(self.graph, self.c2n)
        print(ce.evaluate())