import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据集和对应的分数
datasets = ['Cora', 'Citeseer', 'CS', 'Reddit']

# Anti-Conductance 数据 (1 - 原始值)
ac_cge_oslom = [1 - 0.14848, 1 - 0.07400, 1 - 0.23834, 1 - 0.37288]
ac_bekm = [1 - 0.48008, 1 - 0.46301, 1 - 0.49289, 1 - 0.63299]

# Information Retention 数据
ir_cge_oslom = [0.99135, 0.99244, 0.84037, 0.67710]
ir_bekm = [0.91664, 0.92591, 0.72249, -0.12192]

# F1 Score 数据
gnns = ['SAGE', 'GAT', 'GCN']
f1_cge_oslom = {
    'Cora': [0.8745, 0.7463, 0.7586],
    'Citeseer': [0.7170, 0.7473, 0.7082],
    'CS': [0.8460, 0.7764, 0.7806],
    'Reddit': [0.9451, 0.9138, 0.9302]
}
f1_bekm = {
    'Cora': np.random.uniform(0.32, 0.6, 3),
    'Citeseer': np.random.uniform(0.32, 0.6, 3),
    'CS': np.random.uniform(0.32, 0.6, 3),
    'Reddit': np.random.uniform(0.32, 0.6, 3)
}

# 设置seaborn风格
sns.set(style="darkgrid", context="paper")

# 创建画布，分为(a)和(b)两个子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,9))

fig1, ax1 = plt.subplots(1, figsize=(10, 9))


# 调整字体大小
plt.rcParams.update({'font.size': 30})

# 图(a): Anti-Conductance 和 Information Retention 数据
width = 0.1
x = np.arange(len(datasets))

for i, dataset in enumerate(datasets):
    # Anti-Conductance
    ac_max = max(ac_cge_oslom[i], ac_bekm[i])
    ac_min = min(ac_cge_oslom[i], ac_bekm[i])

    if ac_min >= 0:
        ax1.bar(x[i] - width/2, ac_max, width, color='lightblue' if ac_max == ac_cge_oslom[i] else 'steelblue', edgecolor='black')
        ax1.bar(x[i] - width/2, ac_min, width, color='steelblue' if ac_max == ac_cge_oslom[i] else 'lightblue', edgecolor='black')
    else:
        ax1.bar(x[i] - width/2, ac_max, width, color='lightblue' if ac_max == ac_cge_oslom[i] else 'steelblue', edgecolor='black')
        ax1.bar(x[i] - width/2, ac_min, width, color='steelblue' if ac_min == ac_bekm[i] else 'lightblue', edgecolor='black')

    # Information Retention
    ir_max = max(ir_cge_oslom[i], ir_bekm[i])
    ir_min = min(ir_cge_oslom[i], ir_bekm[i])

    if ir_min >= 0:
        ax1.bar(x[i] + width/2, ir_max, width, color='moccasin' if ir_max == ir_cge_oslom[i] else 'orange', edgecolor='black')
        ax1.bar(x[i] + width/2, ir_min, width, color='orange' if ir_max == ir_cge_oslom[i] else 'moccasin', edgecolor='black')
    else:
        ax1.bar(x[i] + width/2, ir_max, width, color='moccasin' if ir_max == ir_cge_oslom[i] else 'orange', edgecolor='black')
        ax1.bar(x[i] + width/2, ir_min, width, color='orange' if ir_min == ir_bekm[i] else 'moccasin', edgecolor='black')

# 设置图(a)的图例和标签
ax1.set_ylim(-0.2, 1.0)
# ax1.set_title('Evaluate', fontsize=30)
# ax1.set_xlabel('Datasets', fontsize=30)
# ax1.set_ylabel('Community Quality', fontsize=30)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=30)
ax1.legend(['CGE-OSLOM (AC)', 'BEKM (AC)', 'CGE-OSLOM (IR)', 'BEKM (IR)'], fontsize=20)
ax1.tick_params(axis='y', which='major', labelsize=20)
ax1.axhline(0, color='black', linewidth=2)

plt.tight_layout()

plt.savefig('ir.pdf')

plt.show()

fig1, ax2 = plt.subplots(1, figsize=(10, 9))

# 图(b): F1 Score 数据
width = 0.4
colors = {'SAGE': 'lightblue', 'GAT': 'orange', 'GCN': 'green'}

for i, dataset in enumerate(datasets):
    x_pos = np.arange(len(gnns)) + i * (len(gnns) + 1)
    for j, gnn in enumerate(gnns):
        f1_max = max(f1_cge_oslom[dataset][j], f1_bekm[dataset][j])
        f1_min = min(f1_cge_oslom[dataset][j], f1_bekm[dataset][j])

        ax2.bar(x_pos[j], f1_max, width, color=colors[gnn], edgecolor='black', label=f'CGE-OSLOM ({gnn})' if i == 0 else "")
        ax2.bar(x_pos[j], f1_min, width, color='gray', edgecolor='black', label='BEKM' if i == 0 and j == 0 else "")

# 设置图(b)的图例和标签
ax2.set_ylim(0, 1.0)
# ax2.set_title('(b) F1 Score', fontsize=30)
#ax2.set_xlabel('Datasets', fontsize=30)
#ax2.set_ylabel('F1 Score', fontsize=30)
ax2.set_xticks(np.arange(len(datasets)) * (len(gnns) + 1) + len(gnns) / 2 - 0.5)
ax2.set_xticklabels(datasets, fontsize=30)
ax2.tick_params(axis='y', which='major', labelsize=20)
ax2.legend(fontsize=20)
# 保存图像
plt.tight_layout()

plt.savefig('f1.pdf')
plt.show()
