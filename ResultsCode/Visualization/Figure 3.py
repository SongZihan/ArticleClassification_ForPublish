import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from upsetplot import UpSet, from_memberships

import os
import plotly.graph_objects as go
import os, json
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# 对同一测试集，记录每种方法判错的论文 ID（或样本）。

total_k_means = []
total_abstract = []
total_key_insight = []
total_voting = []

round_number = 0
space = 1000

for review_name in os.listdir(
        r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification"):
    # 对于单个review 计算各个方法的性能
    # k-means
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\TraditionalMethod\JustKmeans\{review_name}",
            "rb") as f:
        this_k_means = pickle.load(f)

    diff_indexes = [i + space*round_number for i, (a, b) in enumerate(zip(this_k_means['y_true'], this_k_means['y_pred'])) if a != b]
    total_k_means += diff_indexes

    # abstract
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification\{review_name}",
            "rb") as f:
        this_model_result = pickle.load(f)
    diff_indexes = [i + space*round_number for i, (a, b) in enumerate(zip(this_model_result['y_true'], this_model_result['y_pred'])) if a != b]
    total_abstract += diff_indexes

    # key_insight
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\GPT-4o\key-insights-direction\{review_name}",
            "rb") as f:
        this_model_result = pickle.load(f)
    diff_indexes = [i+ space*round_number for i, (a, b) in enumerate(zip(this_model_result['y_true'], this_model_result['y_pred'])) if a != b]
    total_key_insight += diff_indexes

    # voting
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Voting\result_per_review.json",
            "r") as f:
        this_model_result = json.load(f)['key-insights-direction'][f"{review_name}"]
    diff_indexes = [i+ space*round_number for i, (a, b) in enumerate(zip(this_model_result['y_true'], this_model_result['y_pred'])) if a != b]
    total_voting += diff_indexes

    round_number += 1


import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations

# ==== 新方法名 & 颜色 ====
methods = ["KMC", "ABC", "KBC", "CWV"]
method_colors = {
    "KMC": "#564592",
    "ABC": "#724cf9",
    "KBC": "#ca7df9",
    "CWV": "#f896d8"
}

# ==== 数据准备 ====
set_kmc = set(total_k_means)      # KMC
set_abc = set(total_abstract)     # ABC
set_kbc = set(total_key_insight)  # KBC
set_cwv = set(total_voting)       # CWV
sets = [set_kmc, set_abc, set_kbc, set_cwv]

def all_combinations(items):
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)+1))

comb_counts = {}
for comb in all_combinations(range(len(methods))):
    inters = set.intersection(*(sets[i] for i in comb))
    comb_counts[comb] = len(inters)

# ==== 样式 ====
plt.rcParams.update({
    "font.size": 15,
    "font.family": "Times New Roman",  # 全局字体
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ==== 布局 ====
fig = plt.figure(figsize=(13, 8))
ax_bar = plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=2)  # 上方交集柱
ax_matrix = plt.subplot2grid((3, 4), (2, 1), colspan=3)          # 下方布尔矩阵
ax_sets = plt.subplot2grid((3, 4), (2, 0), rowspan=2)             # 左方法总数（这次改反向）

# ==== 左侧总数条 ====
totals = [len(s) for s in sets]
bars = ax_sets.barh(
    range(len(methods)), totals,
    color=[method_colors[m] for m in methods],
    hatch=['', '\\', '//', 'xx'],
    edgecolor='white'
)

# 添加方法标签（居右放到条形左端外边）
# for i, m in enumerate(methods):
#     ax_sets.text(-max(totals)*0.05, i, m, va='center', ha='right', color='black', fontsize=12)

# 在条形右端（靠近左侧）加数量
for i, v in enumerate(totals):
    ax_sets.text(v - max(totals)*0.02 + 60, i, str(v), va='center', ha='left', fontsize=15, color='black')

# 反转 x 轴方向
ax_sets.invert_xaxis()
ax_sets.invert_yaxis()
# 去掉所有坐标轴 & 刻度
ax_sets.axis('off')


# ==== 上方交集柱形 ====
sorted_combs = sorted(comb_counts.items(), key=lambda x: (-x[1], x[0]))
bar_heights = [count for _, count in sorted_combs]
x_pos = np.arange(len(bar_heights))
ax_bar.bar(x_pos, bar_heights, color="gray")
ax_bar.set_ylabel("Intersection Size")
ax_bar.set_xticks([])
ax_bar.set_xlim(-0.5, len(bar_heights)-0.5)

for i, v in enumerate(bar_heights):
    ax_bar.text(i, v + max(bar_heights)*0.01, str(v),
                ha='center', va='bottom', fontsize=15)

# ==== 下方布尔矩阵 ====
ax_matrix.set_xlim(-0.5, len(bar_heights)-0.5)
ax_matrix.set_ylim(-0.5, len(methods)-0.5)
ax_matrix.set_yticks(range(len(methods)))
ax_matrix.set_yticklabels(methods)

ax_matrix.set_xticks([])
ax_matrix.set_xlabel("")
ax_matrix.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

for xi, (comb, count) in zip(x_pos, sorted_combs):
    ys = []
    for mi in range(len(methods)):
        if mi in comb:
            ax_matrix.plot(xi, mi, 'o', color=method_colors[methods[mi]], markersize=8)
            ys.append(mi)
        else:
            ax_matrix.plot(xi, mi, 'o', color="lightgray", markersize=5)
    if len(ys) > 1:
        ax_matrix.plot([xi, xi], [min(ys), max(ys)], color="black", linewidth=1)

ax_matrix.invert_yaxis()

# ==== 清理边框 ====
for ax in [ax_bar, ax_matrix]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(r"C:\Users\24033\Nutstore\1\我的坚果云\文献聚类论文\Result figure 3.png", dpi=300)
plt.show()