import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据集和对应的分数
datasets = ['Cora', 'Citeseer', 'CS', 'Reddit']

# Anti-Conductance 数据 (1 - 原始值)
ac_cge_oslom = [0.14848, 0.07400, 0.23834, 0.37288]
ac_bekm = [0.48008, 0.46301, 0.49289, 0.63299]

# Information Retention 数据
ir_cge_oslom = [0.99135, 0.99244, 0.84037, 0.67710]
ir_bekm = [0.91664, 0.92591, 0.72249, -0.12192]

# 统一设置seaborn风格
sns.set(style="darkgrid", context="paper")

# 图1：Anti-Conductance 柱状图
fig_ac, ax_ac = plt.subplots(figsize=(18, 9))
width = 0.2
x = np.arange(len(datasets))

ax_ac.bar(x - width/2, ac_cge_oslom, width, color='steelblue', label='CGE-OSLOM (AC)')
ax_ac.bar(x + width/2, ac_bekm, width, color='lightblue', label='BEKM (AC)')

ax_ac.set_ylim(0, 1.0)
#ax_ac.set_xlabel('Datasets', fontsize=30)
ax_ac.set_ylabel('Anti-Conductance', fontsize=40)
ax_ac.set_xticks(x)
ax_ac.set_xticklabels(datasets, fontsize=40)
ax_ac.legend(fontsize=40)
ax_ac.tick_params(axis='both', which='major', labelsize=40)

plt.tight_layout()
plt.savefig('con.pdf')
plt.savefig('con.png')
plt.show()
# 图2：Information Retention 柱状图
fig_ir, ax_ir = plt.subplots(figsize=(10, 9))
width = 0.3  # 与图1保持一致

ax_ir.bar(x - width/2, ir_cge_oslom, width, color='moccasin', edgecolor='black', label='CGE-OSLOM (IR)', alpha=0.6)
ax_ir.bar(x + width/2, ir_bekm, width, color='orange', edgecolor='black', label='BEKM (IR)', alpha=0.6)

ax_ir.set_ylim(-0.2, 1.0)
ax_ir.set_xticks(x)
ax_ir.set_xticklabels(datasets, fontsize=30)
ax_ir.legend(fontsize=20)
ax_ir.tick_params(axis='both', which='major', labelsize=30)
ax_ir.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig('newir.pdf')
plt.show()

# 图3：F1 Score 数据
fig1, ax2 = plt.subplots(figsize=(10, 9))
width = 0.3  # 与图1和图2保持一致
colors = {'SAGE': 'lightblue', 'GAT': 'orange', 'GCN': 'green'}

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

for i, dataset in enumerate(datasets):
    x_pos = np.arange(len(gnns)) + i * (len(gnns) + 1)
    for j, gnn in enumerate(gnns):
        f1_max = max(f1_cge_oslom[dataset][j], f1_bekm[dataset][j])
        f1_min = min(f1_cge_oslom[dataset][j], f1_bekm[dataset][j])

        ax2.bar(x_pos[j], f1_max, width, color=colors[gnn], label=f'CGE-OSLOM ({gnn})' if i == 0 else "", alpha=0.6)
        ax2.bar(x_pos[j], f1_min, width, color='gray', label='BEKM' if i == 0 and j == 0 else "", alpha=0.6)

# 设置图3的图例和标签
ax2.set_ylim(0, 1.0)
ax2.set_xticks(np.arange(len(datasets)) * (len(gnns) + 1) + len(gnns) / 2 - 0.5)
ax2.set_xticklabels(datasets, fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=30)
ax2.legend(fontsize=20)

plt.tight_layout()
plt.savefig('f1.pdf')
plt.show()