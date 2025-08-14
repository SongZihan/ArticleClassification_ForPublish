# Y轴：F1,X轴：k-means，Abstract-based,Key-insight-based,voting.

import os
import matplotlib.pyplot as plt
import os, json
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

import scienceplots

plt.style.use(['science', 'no-latex'])


x = ["KMC", "ABC", "KBC", "CWV"]

y = [0.441, 0.604, 0.703, 0.820]

x_pos = np.arange(len(x))

# 画柱状图
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
bars = ax.bar(x_pos, y, color=['#564592', '#724cf9', '#ca7df9', '#f896d8'],hatch=['', '\\', '//', 'xx'],
              width=0.6, edgecolor='white', alpha=1)


# 加粗坐标轴外框
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=20)

# 设置X/Y轴和标题
ax.yaxis.set_ticks_position('left')   # y轴刻度只在左侧
ax.xaxis.set_ticks_position('bottom') # x轴刻度只在下侧

ax.set_xticks(x_pos)
ax.set_xticklabels(x,  ha='center', fontsize=20)
ax.set_ylabel('Macro F1-Score', fontsize=20)
ax.tick_params(axis='y',which='major',width=2,direction="out",length=4, labelsize=20)
ax.tick_params(axis='y',which='minor',width=1,direction="out",length=2)

ax.tick_params(axis='x',which='major',width=2,direction="out",length=4)

ax.set_ylim(0, 1)
# ax.set_title('Performance Comparison of Classification Methods', fontsize=14, pad=14)

plt.tight_layout()

plt.savefig(r"C:\Users\24033\Nutstore\1\我的坚果云\文献聚类论文\Result figure 2B.png", dpi=100)  # dpi×figsize = 像素大小

plt.show()