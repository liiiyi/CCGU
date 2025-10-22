import logging
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import dgl

class LPA:
    def __init__(self, graph, args, max_iterations=100):
        self.logger = logging.getLogger('LPA')
        self.graph = graph
        self.max_iterations = max_iterations
        self.device = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.graph = self.graph.to(self.device)

        # Log the configuration parameters
        config_str = (
            f"########################### Nucleus Graph Parameters ############################\n"
            f"Aggregation Feature Method: {args['agg_feat']}\n"
            f"Aggregation Label Method: {args['agg_label']}\n"
            f"Aggregation Edge Method: {args['agg_edge']}\n"
            f"Partition Method: {args['partition']}\n"
            f"Threshold Similarity to Edge: {args['th_sim2edge']}\n"
            f"Test Edge Method: {args['test_edge_method']}\n"
            f"Use Edge Weight: {args['use_edge_weight']}\n"
            f"Dataset Name: {args['dataset_name']}"
        )
        self.logger.info(config_str)

    def lpa_partition(self):
        start_time = time.time()
        nodes = self.graph.nodes().to(self.device)
        labels = {node.item(): node.item() for node in nodes.cpu()}

        for iteration in tqdm(range(self.max_iterations), desc="LPA iterations"):
            for node in nodes:
                neighbors = self.graph.successors(node).tolist()
                if not neighbors:
                    continue
                neighbor_labels = [labels[neighbor.item()] for neighbor in neighbors]
                most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                labels[node.item()] = most_common_label

        n2c = defaultdict(list)
        c2n = defaultdict(list)

        for node, label in labels.items():
            n2c[node].append(label)
            c2n[label].append(node)

        elapsed_time = time.time() - start_time
        self.logger.info(f'LPA operation time: {elapsed_time:.2f} seconds')

        return n2c, c2n, len(c2n), elapsed_time

# Example usage:
# args = {'cuda': 0, 'agg_feat': 'pca', 'agg_label': 'th', 'agg_edge': 'rubost', 'partition': 'lpa', 'th_sim2edge': -1, 'test_edge_method': 2, 'use_edge_weight': False, 'dataset_name': 'reddit'}
# graph = ...  # Load your DGL graph here
# lpa = LPA(graph, args)
# n2c, c2n, num_communities, elapsed_time = lpa.lpa_partition()