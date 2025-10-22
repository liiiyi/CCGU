import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据集和对应的分数
datasets = ['Cora', 'Citeseer', 'CS', 'Reddit']

# Information Retention 数据
ir_cge_oslom = [0.99135, 0.99244, 0.84037, 0.67710]
ir_bekm = [0.91664, 0.92591, 0.72249, -0.12192]

# 设置seaborn风格
sns.set(style="darkgrid", context="paper")

# 图2：Information Retention 柱状图
fig_ir, ax_ir = plt.subplots(figsize=(10, 9))

width = 0.3
x = np.arange(len(datasets))

# 添加柱体并调整柱体的起始位置，以确保负值部分显示出来
ax_ir.bar(x - width/2, ir_cge_oslom, width, color='moccasin', edgecolor='black', label='CGE-OSLOM (IR)', bottom=-1)
ax_ir.bar(x + width/2, ir_bekm, width, color='orange', edgecolor='black', label='BEKM (IR)', bottom=-1)

ax_ir.set_ylim(-1, 1.0)  # 设置y轴范围
ax_ir.set_xlabel('Datasets', fontsize=30)
ax_ir.set_ylabel('Information Retention', fontsize=30)
ax_ir.set_xticks(x)
ax_ir.set_xticklabels(datasets, fontsize=30)
ax_ir.legend(fontsize=20)
ax_ir.tick_params(axis='both', which='major', labelsize=30)

plt.tight_layout()
plt.show()