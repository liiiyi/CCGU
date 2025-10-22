import logging
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import dgl
import numpy as np

class Louvain:
    def __init__(self, graph, args, conductance_threshold=0.1):
        self.logger = logging.getLogger('Louvain')
        self.graph = graph
        self.conductance_threshold = conductance_threshold
        self.args = args

    def louvain_partition(self):
        start_time = time.time()
        import networkx as nx
        from cdlib.algorithms import louvain

        g = dgl.to_networkx(self.graph)
        communities = louvain(g, resolution=2.0)

        n2c = defaultdict(list)
        c2n = defaultdict(list)

        for i, community in enumerate(communities):
            for node in community:
                n2c[node].append(i)
                c2n[i].append(node)

        elapsed_time = time.time() - start_time
        self.logger.info(f'Louvain operation time: {elapsed_time:.2f} seconds')

        return n2c, c2n, elapsed_time

    def calculate_conductance(self, c2n):
        conductance_values = {}
        total_edges = self.graph.num_edges()
        for community_id, nodes in c2n.items():
            cut_size = sum(1 for u in nodes for v in self.graph.successors(u) if v not in nodes) / 2
            volume = sum(self.graph.in_degrees(nodes).tolist())
            if volume > 0:
                conductance = cut_size / min(volume, 2 * total_edges - volume)
            else:
                conductance = 0
            conductance_values[community_id] = conductance
        return conductance_values

    def refine_with_oslom(self, c2n):
        refined_n2c = defaultdict(list)
        refined_c2n = defaultdict(list)
        oslom = OSLOM(self.graph, self.args)

        for community_id in tqdm(c2n, desc="Refining communities with OSLOM"):
            community_nodes = c2n[community_id]
            subgraph = self.graph.subgraph(community_nodes)
            subgraph_n2c, subgraph_c2n, _, _ = oslom.OSLOM_partition(subgraph)

            for node, sub_communities in subgraph_n2c.items():
                for sub_community_id in sub_communities:
                    refined_n2c[node].append(f"{community_id}_{sub_community_id}")
                    refined_c2n[f"{community_id}_{sub_community_id}"].append(node)

        return refined_n2c, refined_c2n

    def execute(self):
        n2c, c2n, elapsed_time = self.louvain_partition()
        conductance_values = self.calculate_conductance(c2n)

        for community_id, conductance in conductance_values.items():
            if conductance > self.conductance_threshold:
                self.logger.info(f"Community {community_id} has high conductance ({conductance:.4f}), refining with OSLOM.")
                refined_n2c, refined_c2n = self.refine_with_oslom({community_id: c2n[community_id]})
                n2c.update(refined_n2c)
                c2n.update(refined_c2n)

        return n2c, c2n, len(c2n), elapsed_time


class OSLOM:
    def __init__(self, graph, args, max_iterations=100, threshold=0.2):
        self.logger = logging.getLogger('OSLOM')
        self.graph = graph
        self.max_iterations = max_iterations
        self.threshold = threshold

        # Configuring CUDA and CPU threads
        torch.set_num_threads(args["num_threads"])
        torch.cuda.set_device(args["cuda"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Check if CUDA is available and set device accordingly
        self.device = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.graph = self.graph.to(self.device)

    def OSLOM_partition(self, subgraph=None):
        if subgraph is None:
            subgraph = self.graph

        start_time = time.time()

        nodes = subgraph.nodes().to(self.device)
        labels = {node.item(): [node.item()] for node in nodes.cpu()}

        for iteration in tqdm(range(self.max_iterations), desc="OSLOM iterations"):
            for node in nodes:
                neighbors = subgraph.successors(node).tolist()
                if not neighbors:
                    continue
                received_labels = []
                for neighbor in neighbors:
                    received_labels.extend(labels[neighbor])
                most_common_label = max(set(received_labels), key=received_labels.count)
                labels[node.item()].append(most_common_label)

        n2c = defaultdict(list)
        c2n = defaultdict(list)

        for node, label_list in labels.items():
            label_count = defaultdict(int)
            for label in label_list:
                label_count[label] += 1
            sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels:
                if count / len(label_list) >= self.threshold:
                    n2c[node].append(label)
                    c2n[label].append(node)

        end_time = time.time()
        total_time = end_time - start_time

        return n2c, c2n, len(c2n), total_time

# Example usage:
# graph = ... # load your DGL graph here
# args = {"num_threads": 4, "cuda": 0}
# louvain = Louvain(graph, args)
# n2c, c2n, num_communities, total_time = louvain.execute()
