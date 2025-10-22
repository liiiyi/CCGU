import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def calculate_jaccard_similarity(community_a, community_b):
    set_a, set_b = set(community_a), set(community_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def calculate_edge_count(graph, community_a, community_b):
    src, dst = graph.edges()
    src, dst = src.numpy(), dst.numpy()
    count = 0
    for i in range(len(src)):
        if src[i] in community_a and dst[i] in community_b:
            count += 1
        if src[i] in community_b and dst[i] in community_a:
            count += 1
    return count

def calculate_node_similarity(node_features, nodes_a, nodes_b, method='cosine'):
    vec_a = np.mean(node_features[nodes_a], axis=0)
    vec_b = np.mean(node_features[nodes_b], axis=0)
    if method == 'cosine':
        sim = cosine_similarity([vec_a], [vec_b])[0][0]
    elif method == 'euclidean':
        sim = -euclidean_distances([vec_a], [vec_b])[0][0]  # 距离越小，相似度越高，取负值
    return sim

def calculate_combined_similarity(graph, node_features, node2community, community2node, method='cosine'):
    combined_sim = {}
    for comm_a in community2node.keys():
        for comm_b in community2node.keys():
            if comm_a < comm_b:
                jaccard_sim = calculate_jaccard_similarity(community2node[comm_a], community2node[comm_b])
                edge_count = calculate_edge_count(graph, community2node[comm_a], community2node[comm_b])
                center_sim = calculate_node_similarity(node_features, community2node[comm_a], community2node[comm_b], method)
                combined_sim[(comm_a, comm_b)] = 0.3 * jaccard_sim + 0.3 * edge_count + 0.4 * center_sim  # 这里权重可以通过优化算法来自动调整
    return combined_sim