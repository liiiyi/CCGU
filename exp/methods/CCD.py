import torch
import dgl
import numpy as np
import hnswlib
import random
import logging
import time
import math
from collections import defaultdict
from exp.methods.GetEmbed import GetEmbed

class CentroidCommunityDetection:
    def __init__(self, graph, args):
        """
        初始化社区检测类
        Initialize the community detection class

        参数:
        - graph: DGLGraph 对象，表示图
        - embeddings: 节点嵌入，形状为 (num_nodes, num_features)
        - args: 包含参数信息的字典，例如分辨率参数
        """
        self.graph = graph
        emb = GetEmbed(graph, args)
        self.embeddings = emb.generate_embeddings()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = graph.number_of_nodes()
        self.num_features = self.embeddings.shape[1]
        self.resolution = args['resolution'] if 'resolution' in args else 1.0

        # 初始化日志
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CentroidCommunityDetection')

        # 初始化 HNSW 索引
        # Initialize HNSW index
        self.index = hnswlib.Index(space='l2', dim=self.num_features)
        self.index.init_index(max_elements=self.num_nodes, ef_construction=200, M=16)
        self.index.add_items(self.embeddings.cpu().numpy())

        # 初始化社区和节点字典
        # Initialize community and node dictionaries
        self.C = {i: {'nodes': [i], 'centroid': self.embeddings[i].clone(), 'max_initial_distance': 0} for i in range(self.num_nodes)}
        self.N = {i: i for i in range(self.num_nodes)}

        # 计算初始最大距离以用于停止条件
        # Compute initial max distance for stopping condition
        for node in range(self.num_nodes):
            neighbors = self.query_neighbors(node, k=len(graph.successors(node).tolist()))
            furthest_neighbor = neighbors[-1]
            self.C[node]['max_initial_distance'] = torch.norm(self.embeddings[node] - self.embeddings[furthest_neighbor]).item()

    def query_neighbors(self, node_id, k=1):
        """
        查询节点的最近邻
        Query the nearest neighbors of a node

        参数:
        - node_id: 节点ID
        - k: 查询的邻居数量

        返回:
        - 邻居ID列表
        """
        return self.index.knn_query(self.embeddings[node_id].cpu().numpy(), k=k)[0]

    def calculate_centroid(self, community):
        """
        计算社区质心
        Calculate the centroid of a community

        参数:
        - community: 包含节点列表的社区字典

        返回:
        - 社区质心
        """
        nodes = community['nodes']
        return torch.mean(self.embeddings[nodes], dim=0)

    def calculate_centroid_change(self, old_centroid, new_centroid):
        """
        计算质心变化量
        Calculate the change in centroid

        参数:
        - old_centroid: 旧的质心
        - new_centroid: 新的质心

        返回:
        - 质心变化量
        """
        return torch.norm(new_centroid - old_centroid).item()

    def should_stop(self, community, new_node):
        """
        判断是否应停止合并
        Determine whether to stop merging

        参数:
        - community: 当前社区
        - new_node: 待加入的节点

        返回:
        - 布尔值，表示是否应停止
        """
        old_centroid = self.calculate_centroid(community)
        new_community_nodes = community['nodes'] + [new_node]
        new_centroid = self.calculate_centroid({'nodes': new_community_nodes})
        centroid_change = self.calculate_centroid_change(old_centroid, new_centroid)
        threshold = community['max_initial_distance']
        return centroid_change > threshold

    def merge_communities(self, node, neighbor):
        """
        合并两个社区
        Merge two communities

        参数:
        - node: 节点ID
        - neighbor: 邻居节点ID
        """
        community = self.C[node]['nodes'] + self.C[neighbor]['nodes']
        new_centroid = self.calculate_centroid({'nodes': community})
        self.C[node] = {'nodes': community, 'centroid': new_centroid, 'max_initial_distance': self.C[node]['max_initial_distance']}
        del self.C[neighbor]
        self.N[neighbor] = node

    def simulated_annealing(self):
        """
        实现模拟退火过程
        Implement the simulated annealing process

        返回:
        - 更新后的社区字典
        """
        T = 1.0
        alpha = 0.99
        max_iter = 1000
        current_solution = self.C
        current_energy = self.calculate_energy(current_solution)

        for i in range(max_iter):
            new_solution = self.perturb_solution(current_solution)
            new_energy = self.calculate_energy(new_solution)

            if self.accept_solution(current_energy, new_energy, T):
                current_solution = new_solution
                current_energy = new_energy

            T *= alpha
            if T < 1e-10:
                break

        return current_solution

    def accept_solution(self, current_energy, new_energy, temperature):
        """
        判断是否接受新的解
        Determine whether to accept the new solution

        参数:
        - current_energy: 当前能量
        - new_energy: 新能量
        - temperature: 当前温度

        返回:
        - 布尔值，表示是否接受新的解
        """
        if new_energy < current_energy:
            return True
        else:
            return random.uniform(0, 1) < math.exp((current_energy - new_energy) / temperature)

    def perturb_solution(self, solution):
        """
        扰动当前解
        Perturb the current solution

        参数:
        - solution: 当前社区字典

        返回:
        - 新的社区字典
        """
        node = random.choice(list(solution.keys()))
        neighbors = self.query_neighbors(node, k=len(self.graph.successors(node).tolist()))
        nearest_neighbor = neighbors[0]
        new_solution = solution.copy()
        self.merge_communities(node, nearest_neighbor)
        return new_solution

    def calculate_energy(self, solution):
        """
        计算当前解的能量
        Calculate the energy of the current solution

        参数:
        - solution: 当前社区字典

        返回:
        - 能量值
        """
        energy = 0
        for community in solution.values():
            centroid = community['centroid']
            nodes = community['nodes']
            energy += torch.sum(torch.norm(self.embeddings[nodes] - centroid, dim=1)).item()
        return energy

    def run(self):
        """
        运行社区检测算法
        Run the community detection algorithm

        返回:
        - n2c: 节点到社区的映射
        - c2n: 社区到节点的映射
        - time_consumption: 算法运行时间
        """
        start_time = time.time()
        initial_phase = True

        while not self.convergence_criteria_met():
            if initial_phase:
                for node in list(self.C.keys()):
                    neighbors = self.query_neighbors(node, k=len(self.graph.successors(node).tolist()))
                    furthest_neighbor = neighbors[-1]
                    if not self.should_stop(self.C[node], furthest_neighbor):
                        self.merge_communities(node, furthest_neighbor)
                initial_phase = False
            else:
                self.C = self.simulated_annealing()

        end_time = time.time()
        n2c = {node: self.N[node] for node in range(self.num_nodes)}
        c2n = defaultdict(list)
        for node, community in self.N.items():
            c2n[community].append(node)

        time_consumption = end_time - start_time
        return n2c, c2n, time_consumption

    def convergence_criteria_met(self):
        """
        判断是否满足收敛条件
        Determine whether the convergence criteria are met

        返回:
        - 布尔值，表示是否收敛
        """
        for node in self.C.keys():
            neighbors = self.query_neighbors(node, k=len(self.graph.successors(node).tolist()))
            for neighbor in neighbors:
                if not self.should_stop(self.C[node], neighbor):
                    return False
        return True