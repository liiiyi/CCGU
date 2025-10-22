import logging
import torch
import os
import time
from tqdm import tqdm
from collections import defaultdict

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

    def OSLOM_partition(self):
        """
        Perform OSLOM on the given DGL graph.

        Parameters:
        - graph: a DGL graph
        - max_iterations: maximum number of iterations for the optimization process
        - threshold: the threshold for post-processing to determine the final community membership

        Returns:
        - node2community: dictionary mapping each node to the list of communities it belongs to
        - community2node: dictionary mapping each community to the list of nodes it contains
        - total_time: total time taken for the OSLOM operation
        """
        start_time = time.time()  # Start time

        nodes = self.graph.nodes().to(self.device)
        labels = {node.item(): [node.item()] for node in nodes.cpu()}

        # Main optimization phase
        for iteration in tqdm(range(self.max_iterations), desc="OSLOM iterations"):
            for node in nodes:
                neighbors = self.graph.successors(node).tolist()
                if not neighbors:
                    continue
                received_labels = []
                for neighbor in neighbors:
                    received_labels.extend(labels[neighbor])
                most_common_label = max(set(received_labels), key=received_labels.count)
                labels[node.item()].append(most_common_label)

        n2c = defaultdict(list)
        c2n = defaultdict(list)

        # Post-processing phase to determine final community membership
        for node, label_list in labels.items():
            label_count = defaultdict(int)
            for label in label_list:
                label_count[label] += 1
            sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels:
                if count / len(label_list) >= self.threshold:
                    n2c[node].append(label)
                    c2n[label].append(node)

        # Assign nodes without communities to individual communities
        self.logger.info('Num of communities found by OSLOM = {} | Num of nodes with community = {}'.format(len(c2n), len(n2c.keys())))
        self.logger.info('Assigning lonely nodes to communities of themselves...')
        n_nodes = self.graph.num_nodes()
        for node in tqdm(range(n_nodes)):
            if node not in n2c:
                n2c[node] = [node]
                c2n[node].append(node)

        self.logger.info(f'Num of communities after assigning = {len(c2n)}')

        # Normalize community IDs to be continuous
        community_mapping = {old_label: new_label for new_label, old_label in enumerate(c2n.keys())}
        c2n = {community_mapping[old_label]: nodes for old_label, nodes in c2n.items()}
        n2c = {node: [community_mapping[old_label] for old_label in labels] for node, labels in n2c.items()}

        end_time = time.time()  # End time
        total_time = end_time - start_time

        return n2c, c2n, len(c2n), total_time

# Example usage:
# graph = ... # load your DGL graph here
# args = {"num_threads": 4, "cuda": 0}
# oslom = OSLOM(graph, args)
# n2c, c2n, num_communities, total_time = oslom.OSLOM_partition()