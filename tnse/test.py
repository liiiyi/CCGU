import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from node2vec import Node2Vec
import networkx as nx

from torch.nn import Linear

import config

# 加载数据集
dataset = Planetoid(config.RAW_DATA_PATH, name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# 定义可视化函数
def visualize(embed, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    embed_2d = tsne.fit_transform(embed)

    plt.figure(figsize=(6, 6))
    plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=labels, cmap='jet', s=15, alpha=0.8)

    # 设置正方形比例，去掉边框和坐标轴
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')  # 保证正方形比例

    # 添加标题
    plt.text(0.5, -0.1, title, ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)

    # 保存图像
    plt.savefig(f"{title}.png", bbox_inches='tight', pad_inches=0)
    plt.show()


# 定义模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def pca_embedding(data):
    pca = PCA(n_components=50, random_state=42)
    return pca.fit_transform(data.x.numpy())

# 2. 深度游走 (Node2Vec)
def node2vec_embedding(data):
    # 将 edge_index 转换为 networkx.Graph
    edge_index = data.edge_index.t().numpy()  # 转置为 (num_edges, 2)
    G = nx.Graph()
    G.add_edges_from(edge_index)

    # 使用 Node2Vec 生成嵌入
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=80, workers=2)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # 获取节点嵌入
    embeddings = model.wv
    return torch.tensor([embeddings[str(i)] for i in range(data.num_nodes)])



# 训练函数
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 测试函数
def get_embedding(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
    return out

# 可视化原始数据
visualize(data.x.numpy(), data.y.numpy(), "Original Data")
pca_embed = pca_embedding(data)
visualize(pca_embed, data.y.numpy(), "PCA")
node2vec_embed = node2vec_embedding(data)
visualize(node2vec_embed.numpy(), data.y.numpy(), "Node2Vec")

# 设置模型和优化器
models = {
    "GCN": GCN(dataset.num_features, dataset.num_classes),
    "GAT": GAT(dataset.num_features, dataset.num_classes),
    "SAGE": SAGE(dataset.num_features, dataset.num_classes)
}

for model_name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(100):
        train(model, data, optimizer)
    embed = get_embedding(model, data).numpy()
    visualize(embed, data.y.numpy(), f"{model_name} after 100 epochs")
