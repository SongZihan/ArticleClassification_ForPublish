import os
import plotly.graph_objects as go
import os, json
import pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

x = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'] # review 1 用来评价了


models = ['Claude', 'Deepseek', 'Grok', 'GPT-4o', 'Doubao', 'Gemini']
methods = ['Kmeans', 'DirectClassification', 'KeyInsight', 'VotingDirectClassification', 'VotingKeyInsight']

# 按方法-模型-评测指标组织结果
k_means_result = []

abstract_results = {model:[] for model in models}
key_insight_results = {model:[] for model in models}

for review_name in os.listdir(
        r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification"):
    # 1. k-means
    with open(
            rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\TraditionalMethod\JustKmeans\{review_name}",
            "rb") as f:
        this_k_means = pickle.load(f)
    # 提取所有指标
    y_true = this_k_means['y_true']
    y_pred = this_k_means['y_pred']

    k_means_result.append([accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='macro', zero_division=0),
                           recall_score(y_true, y_pred, average='macro', zero_division=0), f1_score(y_true, y_pred, average='macro')])


    for model in models:
        with open(
                rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\{model}\DirectClassification\{review_name}",
                "rb") as f:
            this_model_result = pickle.load(f)
        y_true = this_model_result['y_true']
        y_pred = this_model_result['y_pred']
        abstract_results[model].append([
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='macro', zero_division=0),
            recall_score(y_true, y_pred, average='macro', zero_division=0),
            f1_score(y_true, y_pred, average='macro')
        ])

    # 3. key_insight
    for model in models:
        with open(
                rf"C:\Users\24033\PycharmProjects\ArticleClassification\Results\{model}\key-insights-direction\{review_name}",
                "rb") as f:
            this_model_result = pickle.load(f)
        y_true = this_model_result['y_true']
        y_pred = this_model_result['y_pred']
        key_insight_results[model].append([
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='macro', zero_division=0),
            recall_score(y_true, y_pred, average='macro', zero_division=0),
            f1_score(y_true, y_pred, average='macro')
        ])

    # 4. voting method



    # voting method
    # with open(r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Voting\result_per_review.json", "r") as f:
    #     voting_result = json.load(f)
    #
    # voting_results_direct_classification_results.append(
    #     f1_score(voting_result['DirectClassification'][review_name]['y_true'],
    #              voting_result['DirectClassification'][review_name]['y_pred'],
    #              average='macro'))
    # voting_results_key_insight_classification_results.append(
    #     f1_score(voting_result['key-insights-direction'][review_name]['y_true'],
    #              voting_result['key-insights-direction'][review_name]['y_pred'], average='macro'))

final_k_means = {
        'accuracy': np.mean(np.array(k_means_result)[:, 0]),
        'precision': np.mean(np.array(k_means_result)[:, 1]),
        'recall': np.mean(np.array(k_means_result)[:, 2]),
        'f1': np.mean(np.array(k_means_result)[:, 3])
    }

for key,value in abstract_results.items():
    print(f"""For model {key}, overall score is {
    {
        'accuracy': np.mean(np.array(value)[:, 0]),
        'precision': np.mean(np.array(value)[:, 1]),
        'recall': np.mean(np.array(value)[:, 2]),
        'f1': np.mean(np.array(value)[:, 3])
    }
    }""")

print("===================")
for key,value in key_insight_results.items():
    print(f"""For model {key}, overall score is {
    {
        'accuracy': np.mean(np.array(value)[:, 0]),
        'precision': np.mean(np.array(value)[:, 1]),
        'recall': np.mean(np.array(value)[:, 2]),
        'f1': np.mean(np.array(value)[:, 3])
    }
    }""")