import os,pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

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



def get_this_performance_list(path):
    y_true_list = []
    y_pred_list = []

    for file in os.listdir(path):
        try:
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"<UNK>{path} + {file}<UNK>")
            raise e

        # Assuming data is a dictionary with keys 'y_true' and 'y_pred'
        y_true = data.get('y_true', [])
        y_pred = data.get('y_pred', [])

        y_true_list += y_true
        y_pred_list += y_pred

    return y_true_list,y_pred_list


target_model = ['Grok','GPT-4o','Claude','Gemini']
base_path = r"C:\Users\24033\PycharmProjects\ArticleClassification\Results"



multi_model_result = {"direct_classification":{},"key_insight":{}}

for this_folder in target_model:
    this_path = os.path.join(base_path, this_folder)


    direct_classification_performance_path = os.path.join(this_path, 'DirectClassification')
    direct_y_true,direct_y_pred = get_this_performance_list(direct_classification_performance_path)
    multi_model_result["direct_classification"][this_folder] = {"y_true":direct_y_true,"y_pred":direct_y_pred}


    key_insight_performance_path = os.path.join(this_path, 'key-insights-direction')
    key_insight_y_true,key_insight_y_pred = get_this_performance_list(key_insight_performance_path)
    multi_model_result['key_insight'][this_folder] = {"y_true":key_insight_y_true,"y_pred":key_insight_y_pred}

"""
计算direct_classification情况下多智能体投票的性能
"""
# grok_direct_y_true,grok_direct_y_pred = multi_model_result['direct_classification']['Grok']['y_true'],multi_model_result['direct_classification']['Grok']['y_pred']
# gpt4o_direct_y_true,gpt4o_direct_y_pred = multi_model_result['direct_classification']['GPT-4o']['y_true'],multi_model_result['direct_classification']['GPT-4o']['y_pred']
# # Doubao_direct_y_true,Doubao_direct_y_pred = multi_model_result['direct_classification']['Doubao']['y_true'],multi_model_result['direct_classification']['Doubao']['y_pred']
# Claude_direct_y_true,Claude_direct_y_pred = multi_model_result['direct_classification']['Claude']['y_true'],multi_model_result['direct_classification']['Claude']['y_pred']
# Gemini_direct_y_true,Gemini_direct_y_pred = multi_model_result['direct_classification']['Gemini']['y_true'],multi_model_result['direct_classification']['Gemini']['y_pred']
# # Deepseek_direct_y_true,Deepseek_direct_y_pred = multi_model_result['direct_classification']['Deepseek']['y_true'],multi_model_result['direct_classification']['Deepseek']['y_pred']
#
#
# total_direct_y_pred = []
# for i in range(len(grok_direct_y_pred)):
#     total_vote = [grok_direct_y_pred[i],gpt4o_direct_y_pred[i],Claude_direct_y_pred[i],Gemini_direct_y_pred[i]]
#     # 选择占比最高的作为当前选择
#     this_selection = most_common_element(total_vote)
#     total_direct_y_pred.append(this_selection)
#
# # 投票模式下的整体准确率
# print(f"Abstrct classification multi vote performance: {accuracy_score(total_direct_y_pred,Claude_direct_y_true)}")

grok_key_insight_y_true,grok_key_insight_y_pred = multi_model_result['key_insight']['Grok']['y_true'],multi_model_result['key_insight']['Grok']['y_pred']
gpt4o_key_insight_y_true,gpt4o_key_insight_y_pred = multi_model_result['key_insight']['GPT-4o']['y_true'],multi_model_result['key_insight']['GPT-4o']['y_pred']
# Doubao_key_insight_y_true,Doubao_key_insight_y_pred = multi_model_result['key_insight']['Doubao']['y_true'],multi_model_result['key_insight']['Doubao']['y_pred']
Claude_key_insight_y_true,Claude_key_insight_y_pred = multi_model_result['key_insight']['Claude']['y_true'],multi_model_result['key_insight']['Claude']['y_pred']
Gemini_key_insight_y_true,Gemini_key_insight_y_pred = multi_model_result['key_insight']['Gemini']['y_true'],multi_model_result['key_insight']['Gemini']['y_pred']
# Deepseek_key_insight_y_true,Deepseek_key_insight_y_pred = multi_model_result['key_insight']['Deepseek']['y_true'],multi_model_result['key_insight']['Deepseek']['y_pred']

print(grok_key_insight_y_true==gpt4o_key_insight_y_true==Claude_key_insight_y_true==Gemini_key_insight_y_true)

total_direct_y_pred = []
for i in range(len(grok_key_insight_y_pred)):
    total_vote = [grok_key_insight_y_pred[i],gpt4o_key_insight_y_pred[i],Claude_key_insight_y_true[i],Gemini_key_insight_y_pred[i]]
    this_selection = most_common_element(total_vote)
    total_direct_y_pred.append(this_selection)

# 投票模式下的整体准确率
print(f"key-insight classification multi vote performance: "
      f"""{[
            accuracy_score(total_direct_y_pred, Gemini_key_insight_y_true),
            precision_score(total_direct_y_pred, Gemini_key_insight_y_true, average='macro', zero_division=0),
            recall_score(total_direct_y_pred, Gemini_key_insight_y_true, average='macro', zero_division=0),
            f1_score(total_direct_y_pred, Gemini_key_insight_y_true, average='macro')
        ]
      }""")


