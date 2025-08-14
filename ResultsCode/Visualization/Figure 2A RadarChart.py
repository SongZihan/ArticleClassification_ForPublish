import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import os, json
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score
import scienceplots

# plt.style.use(['science', 'no-latex'])

key_insight_results = []
directClassification_results = []

base_path = r"C:\Users\24033\PycharmProjects\ArticleClassification\Results"
for model in ['Grok','Gemini','GPT-4o','Claude','Deepseek','Doubao']:
    this_model_path = os.path.join(base_path, model)

    this_model_direct_results = []
    this_model_key_insight_results = []

    direct_classification_review_path = os.path.join(this_model_path, 'DirectClassification')
    key_insight_reviews_path = os.path.join(this_model_path, 'key-insights-direction')
    for review_name in os.listdir(r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification"):
        if review_name == "Review1":
            continue
        this_direct_classfication_review_path = os.path.join(direct_classification_review_path, review_name)
        this_key_insight_reviews_path = os.path.join(key_insight_reviews_path, review_name)

        with open(this_direct_classfication_review_path,"rb") as f:
            this_model_direct_result = pickle.load(f)
        with open(this_key_insight_reviews_path,"rb") as f:
            this_model_key_insight_result = pickle.load(f)

        this_model_direct_results.append(f1_score(this_model_direct_result['y_true'], this_model_direct_result['y_pred'],average='macro'))
        this_model_key_insight_results.append(f1_score(this_model_key_insight_result['y_true'], this_model_key_insight_result['y_pred'],average='macro'))

    key_insight_results.append(np.mean(this_model_key_insight_results))
    directClassification_results.append(np.mean(this_model_direct_results))

# 加入vote的结果
# key_insight_results.append(0.76)
# directClassification_results.append(0.67)

plt.rcParams['font.family'] = 'Times New Roman'

# 属性和数据
labels = ['Grok', 'Gemini', 'GPT', 'Claude', 'Deepseek', 'Doubao']
data1 = key_insight_results
data2 = directClassification_results
num_vars = len(labels)

# 角度设置
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
data1 += data1[:1]
data2 += data2[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), dpi=100, subplot_kw=dict(polar=True))

# 第一组
ax.plot(angles, data1, color='blue', linewidth=2, label='KBC',linestyle='--', alpha=0.7)
ax.fill(angles, data1, color='blue', alpha=0.15)

# 第二组
ax.plot(angles, data2, color='red', linewidth=2, label='ABC',linestyle=':')
ax.fill(angles, data2, color='red', alpha=0.15)



ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels,fontsize=25)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],fontsize=20)

leg = ax.legend(loc='upper right', bbox_to_anchor=(0.15, 1.17), fontsize=20,fancybox=False)
leg.get_frame().set_edgecolor('black')   # 设置边框颜色为黑色
leg.get_frame().set_linestyle('-')       # 设置边框为实线
leg.get_frame().set_linewidth(2)       # 可选，设置边框宽度，增加可见性


plt.savefig(r"C:\Users\24033\Nutstore\1\我的坚果云\文献聚类论文\Result figure 2A.png", dpi=100)  # dpi×figsize = 像素大小

plt.show()