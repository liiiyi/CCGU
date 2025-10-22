import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

# Dataset-CS
y_grapheraser_cs = np.array([0, 19, 33, 41, 43, 48, 53, 57, 58, 60, 62])
y_guide_cs = np.array([0, 21, 32, 43, 50, 53, 55, 57, 61, 63, 66])
y_cge_cs = np.array([0, 5.2, 6.1, 6.7, 7.3, 7.9, 8.2, 8.5, 8.9, 9, 9.3])

# Dataset-Reddit
y_grapheraser_reddit = np.array([0, 2060, 2626, 2780, 2810, 2840, 2890, 2923, 2973, 3066, 3128])
y_guide_reddit = np.array([0, 2150, 2740, 2920, 3010, 2940, 3081, 3198, 3277, 3331, 3411])
y_cge_reddit = np.array([0, 37, 37.7, 38.1, 39.8, 41, 43.2, 43.3, 44, 45.7, 47])

# Set seaborn style
sns.set(style="darkgrid")
sns.set_theme(context="paper")

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 9))

# Plot Dataset-CS
sns.lineplot(ax=axs[0], x=x, y=y_grapheraser_cs, label='GraphEraser', color='red', marker='^', markersize=10, linewidth=2)
sns.lineplot(ax=axs[0], x=x, y=y_guide_cs, label='GUIDE', color='green', marker='o', markersize=10, linewidth=2)
sns.lineplot(ax=axs[0], x=x, y=y_cge_cs, label='CGE', color='blue', marker='s', markersize=10, linewidth=2)

# axs[0].set_title('CS', fontsize=40)
axs[0].set_xlabel('Unlearning Ratio on CS (%)', fontsize=40)
axs[0].set_ylabel('Time Consumption (s)', fontsize=40)
axs[1].tick_params(axis='x', labelsize=32)
axs[1].tick_params(axis='y', labelsize=32)
axs[0].legend(fontsize=30)
axs[0].grid(True)

# Plot Dataset-Reddit
sns.lineplot(ax=axs[1], x=x, y=y_grapheraser_reddit, label='GraphEraser', color='red', marker='^', markersize=10, linewidth=2)
sns.lineplot(ax=axs[1], x=x, y=y_guide_reddit, label='GUIDE', color='green', marker='o', markersize=10, linewidth=2)
sns.lineplot(ax=axs[1], x=x, y=y_cge_reddit, label='CGE', color='blue', marker='s', markersize=10, linewidth=2)

# axs[1].set_title('Reddit', fontsize=40)
axs[1].set_xlabel('Unlearning Ratio on Reddit (%)', fontsize=40)
# axs[1].set_ylabel('Time Consumption', fontsize=32)
axs[0].tick_params(axis='x', labelsize=32)
axs[0].tick_params(axis='y', labelsize=32)
axs[1].legend(fontsize=30)
axs[1].grid(True)

# Save plots
plt.tight_layout()
plt.savefig('drawplt.pdf')
plt.savefig('drawplt.png', dpi=300)

# Show plot
plt.show()
