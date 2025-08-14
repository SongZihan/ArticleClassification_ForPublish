import os,pickle, json
from sklearn.metrics import classification_report, accuracy_score, f1_score

from collections import Counter

def most_common_element(total_vote):
    model_weight = [0.817,0.848,0.799,0.800]
    # 统计每个label的权重和
    label_weight_sum = {}
    for label, weight in zip(total_vote, model_weight):
        if label not in label_weight_sum:
            label_weight_sum[label] = 0
        label_weight_sum[label] += weight

    # 找到权重和最大的label
    max_label = max(label_weight_sum, key=label_weight_sum.get)
    return max_label
result = {
    "DirectClassification":{},
    "key-insights-direction":{}
}

for file_name in os.listdir(r'C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\DirectClassification'):

    for i in ['DirectClassification','key-insights-direction']:
        gpt_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\GPT-4o\{i}\{file_name}"
        claude_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Claude\{i}\{file_name}"
        grok_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Grok\{i}\{file_name}"
        gemini_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Gemini\{i}\{file_name}"
        doubao_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Doubao\{i}\{file_name}"
        deepseek_path = fr"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Deepseek\{i}\{file_name}"

        with open(gpt_path,'rb') as f:
            gpt_result = pickle.load(f)
        # with open(claude_path,'rb') as f:
        #     claude_result = pickle.load(f)
        with open(grok_path,'rb') as f:
            grok_result = pickle.load(f)
        with open(gemini_path,'rb') as f:
            gemini_result = pickle.load(f)
        with open(doubao_path,'rb') as f:
            doubao_result = pickle.load(f)
        # with open(deepseek_path,'rb') as f:
        #     deepseek_result = pickle.load(f)


        y_true = gpt_result['y_true']
        y_pred = []
        for k in range(len(y_true)):
            y_pred.append(most_common_element([gpt_result['y_pred'][k],grok_result['y_pred'][k],gemini_result['y_pred'][k],doubao_result['y_pred'][k]]))

        result[i][file_name] = {}
        result[i][file_name]['y_pred'] = y_pred
        result[i][file_name]['y_true'] = y_true
# 写入json文件
with open(r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\Voting\result_per_review.json", 'w') as f:
    json.dump(result,f,indent=4)