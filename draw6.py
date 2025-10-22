import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置Seaborn样式
sns.set(style="darkgrid")

datasets = ['Cora', 'Citeseer', 'CS', 'Reddit']
gnns = ['SAGE', 'GAT', 'GCN']

f1_cge_oslom = {
    'Cora': [0.8745, 0.7463, 0.7586],
    'Citeseer': [0.7170, 0.7473, 0.7082],
    'CS': [0.8460, 0.7764, 0.7806],
    'Reddit': [0.9451, 0.9138, 0.9302]
}
f1_slpa = {
    'Cora': [0.8345, 0.7363, 0.7486],
    'Citeseer': [0.7070, 0.7333, 0.7012],
    'CS': [0.8260, 0.7364, 0.7606],
    'Reddit': [0, 0, 0]
}
f1_infomap = {
    'Cora': [0.7925, 0.6763, 0.6686],
    'Citeseer': [0.6370, 0.6933, 0.6612],
    'CS': [0.5160, 0.5564, 0.5806],
    'Reddit': [0.2361, 0.3444, 0.2910]
}

f1_data = []
for dataset in datasets:
    for i, gnn in enumerate(gnns):
        f1_data.append({
            'Dataset': dataset,
            'GNN': gnn,
            'Score': f1_cge_oslom[dataset][i],
            'Method': 'OSLOM'
        })
        f1_data.append({
            'Dataset': dataset,
            'GNN': gnn,
            'Score': f1_slpa[dataset][i],
            'Method': 'SLPA'
        })
        f1_data.append({
            'Dataset': dataset,
            'GNN': gnn,
            'Score': f1_infomap[dataset][i],
            'Method': 'Infomap'
        })

f1_df = pd.DataFrame(f1_data)

# 图 - F1 Scores
plt.figure(figsize=(18, 11))

# 设置柱子宽度
bar_width = 0.3

# 绘制OSLOM的柱子
sns.barplot(
    x='Dataset', y='Score', hue='GNN', data=f1_df[f1_df['Method'] == 'OSLOM'],
    palette=['#808080', '#A9A9A9', '#C0C0C0'], dodge=True,
    width=bar_width, alpha=0.4
)

# 绘制SLPA的柱子
sns.barplot(
    x='Dataset', y='Score', hue='GNN', data=f1_df[f1_df['Method'] == 'SLPA'],
    palette=['#4682B4', '#5F9EA0', '#87CEFA'], dodge=True,
    width=bar_width, alpha=0.4
)

# 绘制Infomap的柱子
sns.barplot(
    x='Dataset', y='Score', hue='GNN', data=f1_df[f1_df['Method'] == 'Infomap'],
    palette=['#8B0000', '#B22222', '#DC143C'], dodge=True,
    width=bar_width, alpha=0.4
)

# 手动添加图例标签
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [
    'SAGE (OSLOM)', 'GAT (OSLOM)', 'GCN (OSLOM)',
    'SAGE (SLPA)', 'GAT (SLPA)', 'GCN (SLPA)',
    'SAGE (Infomap)', 'GAT (Infomap)', 'GCN (Infomap)',
]

# 替换图例标签
plt.legend(handles=handles[:3] + handles[3:6] + handles[6:9], labels=new_labels, loc='upper right', fontsize=40)

plt.legend(handles=handles[:3] + handles[3:6] + handles[6:9], labels=new_labels,
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=3, fontsize=25)

# 设置y轴范围
plt.ylim(0, 1)
plt.ylabel('Macro F1', fontsize=40)
plt.xticks(fontsize=40)

# 手动移除x轴描述
plt.xlabel('')

plt.yticks(fontsize=40)
plt.grid(True)
plt.tight_layout()
plt.savefig('cdf1.pdf')
plt.savefig('cdf1.png')
plt.show()
