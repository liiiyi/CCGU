import logging
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import dgl

class Louvain:
    def __init__(self, graph, args=None):
        self.logger = logging.getLogger('Louvain')
        self.graph = graph
        args = args or {}
        self.seed = int(args.get('random_seed', 0))
        # Its own flag, not ccp_resolution: this wrapper has always used 2.0, and
        # the CLI always supplies --ccp_resolution (default 1.0), which would
        # silently halve the legacy resolution.
        self.resolution = float(args.get('louvain_resolution', 2.0))

    def louvain_partition(self):
        start_time = time.time()
        import networkx as nx

        g = dgl.to_networkx(self.graph).to_undirected()
        try:
            from cdlib.algorithms import louvain

            communities = louvain(g, resolution=self.resolution).communities
        except ImportError:
            # cdlib is not part of the pinned environment; networkx ships the same
            # algorithm and takes an explicit seed, which the pipeline needs anyway.
            from networkx.algorithms.community import louvain_communities

            self.logger.info(
                'cdlib is not installed, using networkx louvain_communities '
                '(resolution %.3f, seed %d)', self.resolution, self.seed
            )
            communities = louvain_communities(
                g, resolution=self.resolution, seed=self.seed
            )

        n2c = defaultdict(list)
        c2n = defaultdict(list)

        for i, community in enumerate(communities):
            for node in community:
                n2c[int(node)].append(i)
                c2n[i].append(int(node))

        elapsed_time = time.time() - start_time
        self.logger.info(f'Louvain operation time: {elapsed_time:.2f} seconds')

        return n2c, c2n, len(c2n), elapsed_time

# Example usage:
# args = {'cuda': 0, 'agg_feat': 'pca', 'agg_label': 'th', 'agg_edge': 'rubost', 'partition': 'louvain', 'th_sim2edge': -1, 'test_edge_method': 2, 'use_edge_weight': False, 'dataset_name': 'reddit'}
# graph = ...  # Load your DGL graph here
# louvain = Louvain(graph, args)
# n2c, c2n, num_communities, elapsed_time = louvain.louvain_partition()