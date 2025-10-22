import logging

import dgl
import networkx as nx
from cdlib import algorithms
import time
from infomap import Infomap as Ifp

class Infomap:
    def __init__(self, graph, args, trials=100, markov_time=1.0, seed=42, silent=True, directed=True):
        self.logger = logging.getLogger('Infomap')
        self.graph = graph
        self.trials = trials
        self.markov_time = markov_time
        self.seed = seed
        self.silent = silent
        self.directed = directed

    def partition(self):
        start_time = time.time()

        # nx_graph = dgl.to_networkx(self.graph)
        communities = self.run_infomap(graph=self.graph)
        c2n = {i: community for i, community in enumerate(communities.values())}

        # 构建节点到社区的映射 (node to community, n2c)
        n2c = {}
        for community_id, nodes in c2n.items():
            for node in nodes:
                n2c[node] = community_id

        total_time = time.time() - start_time
        return n2c, c2n, len(c2n), total_time



    def run_infomap(self, graph):
        infomap_args = f"--two-level --directed --num-trials {self.trials} --markov-time {self.markov_time} --seed {self.seed} --silent"
        infomap = Ifp("--two-level --directed --num-trials 100 --seed 42 --silent")

        # Add edges to Infomap
        if isinstance(graph, nx.Graph):
            for edge in graph.edges(data=True):
                if 'weight' in edge[2]:
                    infomap.addLink(edge[0], edge[1], edge[2]['weight'])
                else:
                    infomap.addLink(edge[0], edge[1])
        elif isinstance(graph, dgl.DGLGraph):
            src, dst = graph.edges()
            weights = graph.edata.get('weight', None)
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                if weights is not None:
                    infomap.addLink(s, d, weights[i].item())
                else:
                    infomap.addLink(s, d)
        else:
            raise TypeError("Graph must be either NetworkX or DGL graph")

        # Run the algorithm
        infomap.run()

        # Extract communities
        communities = {}

        '''
        for node in infomap.nodes:
            module_id = infomap.getModules()[node.nodeId]
            if module_id not in communities:
                communities[module_id] = []
            communities[module_id].append(node.nodeId)
            '''

        for node in infomap.tree:
            if node.is_leaf:
                c_id = node.module_id - 1
                if c_id not in communities:
                    communities[c_id] = []
                communities[c_id].append(node.node_id)

        return communities


