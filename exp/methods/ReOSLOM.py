import logging
import torch
import os
from tqdm import tqdm
from collections import defaultdict
from exp.methods import OSLOM
import dgl

class ReOSLOM:
    def __init__(self, graph, args, theta=20):
        self.logger = logging.getLogger('ReOSLOM')
        self.graph = graph
        self.args = args
        self.theta = theta

        torch.set_num_threads(args["num_threads"])
        torch.cuda.set_device(args["cuda"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.device = torch.device(f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu')
        self.graph = self.graph.to(self.device)

    def refine(self, n2c_louvain):
        c2n_louvain = defaultdict(list)
        for node, comms in n2c_louvain.items():
            for comm in comms:
                c2n_louvain[comm].append(node)

        new_n2c = {}
        new_c2n = {}
        community_id_counter = 0

        for comm_id, nodes in tqdm(c2n_louvain.items(), desc="Refining large communities with OSLOM"):
            if len(nodes) <= self.theta:
                # Keep small communities unchanged
                new_c2n[community_id_counter] = nodes
                for node in nodes:
                    new_n2c[node] = [community_id_counter]
                community_id_counter += 1
            else:
                # Apply OSLOM to large community
                subgraph = dgl.node_subgraph(self.graph, torch.tensor(nodes, device=self.device))
                old_to_new = {old: i for i, old in enumerate(nodes)}
                new_to_old = {i: old for i, old in enumerate(nodes)}

                # Run OSLOM on the subgraph
                oslom = OSLOM(subgraph, self.args)
                sub_n2c, sub_c2n, _, _ = oslom.OSLOM_partition()

                for sub_comm_nodes in sub_c2n.values():
                    real_nodes = [new_to_old[sub_node] for sub_node in sub_comm_nodes]
                    new_c2n[community_id_counter] = real_nodes
                    for node in real_nodes:
                        new_n2c.setdefault(node, []).append(community_id_counter)
                    community_id_counter += 1

        return new_n2c, new_c2n
