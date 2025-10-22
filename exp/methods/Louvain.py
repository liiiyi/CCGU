import logging
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import dgl

class Louvain:
    def __init__(self, graph, args):
        self.logger = logging.getLogger('Louvain')
        self.graph = graph

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

        return n2c, c2n, len(c2n), elapsed_time

# Example usage:
# args = {'cuda': 0, 'agg_feat': 'pca', 'agg_label': 'th', 'agg_edge': 'rubost', 'partition': 'louvain', 'th_sim2edge': -1, 'test_edge_method': 2, 'use_edge_weight': False, 'dataset_name': 'reddit'}
# graph = ...  # Load your DGL graph here
# louvain = Louvain(graph, args)
# n2c, c2n, num_communities, elapsed_time = louvain.louvain_partition()