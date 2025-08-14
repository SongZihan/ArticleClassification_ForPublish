import os
import plotly.graph_objects as go
import os, json
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from scipy.stats import wilcoxon


k_means_results = []
directClassification_results = []
key_insight_results = []
voting_results_direct_classification_results = []
voting_results_key_insight_classification_results = []

for review_name in os.listdir(
        r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification"):
    # 对于单个review 计算各个方法的性能
    # k-means
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\TraditionalMethod\JustKmeans\{review_name}",
            "rb") as f:
        this_k_means = pickle.load(f)
    k_means_results.append(f1_score(this_k_means['y_true'], this_k_means['y_pred'], average='macro'))

    # direct classification
    total_this_score_per_model = []
    for model in ['Claude', 'Deepseek', 'Grok', 'GPT-4o', 'Doubao', 'Gemini']:
        with open(
                rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\{model}\DirectClassification\{review_name}",
                "rb") as f:
            this_model_result = pickle.load(f)
        total_this_score_per_model.append(
            f1_score(this_model_result['y_true'], this_model_result['y_pred'], average='macro'))
    directClassification_results.append(np.mean(total_this_score_per_model))

    # key_insight result
    total_this_score_per_model = []
    for model in ['Claude', 'Deepseek', 'Grok', 'GPT-4o', 'Doubao', 'Gemini']:
        with open(
                rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\{model}\key-insights-direction\{review_name}",
                "rb") as f:
            this_model_result = pickle.load(f)
        total_this_score_per_model.append(
            f1_score(this_model_result['y_true'], this_model_result['y_pred'], average='macro'))
    key_insight_results.append(np.mean(total_this_score_per_model))

    # voting method
    with open(r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Voting\result_per_review.json", "r") as f:
        voting_result = json.load(f)

    voting_results_direct_classification_results.append(
        f1_score(voting_result['DirectClassification'][review_name]['y_true'],
                 voting_result['DirectClassification'][review_name]['y_pred'],
                 average='macro'))
    voting_results_key_insight_classification_results.append(
        f1_score(voting_result['key-insights-direction'][review_name]['y_true'],
                 voting_result['key-insights-direction'][review_name]['y_pred'], average='macro'))




# 执行Wilcoxon Signed-Rank Test
statistic, k_means2Direct = wilcoxon(k_means_results, directClassification_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"k_means2Direct, Statistic: {statistic:.4f}, p-value: {k_means2Direct:.4f}")
statistic, kmeans2keyinsight = wilcoxon(k_means_results, key_insight_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"kmeans2keyinsight, Statistic: {statistic:.4f}, p-value: {kmeans2keyinsight:.4f}")
statistic, kmeans2voting = wilcoxon(k_means_results, voting_results_key_insight_classification_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"kmeans2voting, Statistic: {statistic:.4f}, p-value: {kmeans2keyinsight:.4f}")
statistic, direct2keyinsight = wilcoxon(directClassification_results, key_insight_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"direct2keyinsight, Statistic: {statistic:.4f}, p-value: {direct2keyinsight:.4f}")
statistic, direct2voting = wilcoxon(directClassification_results, voting_results_key_insight_classification_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"direct2voting, Statistic: {statistic:.4f}, p-value: {direct2voting:.4f}")
statistic, keyinsight2voting = wilcoxon(key_insight_results, voting_results_key_insight_classification_results, alternative='two-sided')  # 可改为 'greater' 或 'less'
print(f"keyinsight2voting, Statistic: {statistic:.4f}, p-value: {keyinsight2voting:.4f}")