import networkx as nx
from cdlib import algorithms, evaluation

# 创建一个示例图
G = nx.karate_club_graph()

# 定义一个函数来评估不同分辨率参数下的社区划分结果
def evaluate_resolution(G, resolutions):
    results = []
    for resolution in resolutions:
        louvain_communities = algorithms.louvain(G, resolution=resolution)
        modularity_score = evaluation.newman_girvan_modularity(G, louvain_communities).score
        results.append((resolution, modularity_score, louvain_communities.communities))
    return results

# 定义不同的分辨率参数进行测试
resolutions = [0.5, 1.0, 1.5, 2.0, 2.5]

# 评估不同分辨率参数下的社区划分结果
results = evaluate_resolution(G, resolutions)

# 打印结果
for resolution, modularity_score, communities in results:
    print(f"分辨率: {resolution}, 模块度得分: {modularity_score}, 社区数量: {len(communities)}")

# 选择最佳的分辨率参数
best_resolution, best_score, best_communities = max(results, key=lambda x: x[1])
print(f"最佳分辨率: {best_resolution}, 最佳模块度得分: {best_score}, 最佳社区数量: {len(best_communities)}")
