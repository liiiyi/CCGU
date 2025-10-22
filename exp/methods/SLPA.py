import logging
from collections import defaultdict
import time
from tqdm import tqdm


class SLPA:
    def __init__(self, graph, max_iterations=100, threshold=0.2, max_communities_per_node=5):
        self.logger = logging.getLogger('SLPA')
        self.graph = graph
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.max_c_per_n = max_communities_per_node

    def SLPA_partition(self):
        """
        Perform SLPA on the given DGL graph.

        Parameters:
        - graph: a DGL graph
        - max_iterations: maximum number of iterations for label propagation
        - threshold: the threshold for post-processing to determine the final community membership

        Returns:
        - node2community: dictionary mapping each node to the list of communities it belongs to
        - community2node: dictionary mapping each community to the list of nodes it contains
        """
        start_time = time.time()
        nodes = self.graph.nodes()
        labels = {node.item(): [node.item()] for node in nodes}

        # Label propagation phase
        for iteration in tqdm(range(self.max_iterations), desc="SLPA iterations"):
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
            for label, count in sorted_labels[:self.max_c_per_n]:
                if count / len(label_list) >= self.threshold:
                    n2c[node].append(label)
                    c2n[label].append(node)

        # Assign nodes without communities to individual communities
        self.logger.info('Num of communities found by SLPA = {} | Num of nodes with community = {}'.format(len(c2n), len(n2c.keys())))
        self.logger.info('Assigning lonely nodes to communities of themselves...')
        n_nodes = self.graph.num_nodes()
        for node in tqdm(range(n_nodes)):
            if node not in n2c:
                n2c[node] = [node]
                c2n[node].append(node)

        self.logger.info(f'Num of communities after assigning = {len(c2n)}')

        over_time = time.time() - start_time

        # Normalize community IDs to be continuous
        community_mapping = {old_label: new_label for new_label, old_label in enumerate(c2n.keys())}
        c2n = {community_mapping[old_label]: nodes for old_label, nodes in c2n.items()}
        n2c = {node: [community_mapping[old_label] for old_label in labels] for node, labels in n2c.items()}

        return n2c, c2n, len(c2n), over_time