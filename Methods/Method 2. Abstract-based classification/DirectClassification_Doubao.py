"""
直接使用LLM模型进行分类的代码
"""
import json, os
from difflib import SequenceMatcher

from openai import OpenAI
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from volcenginesdkarkruntime import Ark


# 从环境变量中读取您的方舟API Key
client = Ark(api_key="")


def GPT4oPromptGenerate(system_prompt, user_prompt):
    completion = client.chat.completions.create(model="doubao-1.5-pro-32k-250115", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],response_format={"type": "json_object"},
    temperature=0.7, top_p=0.95
                                                )
    return completion.choices[0].message.content

def find_most_similar_string(target, string_list):
    """
    从字符串列表中找出与目标字符串最相似的字符串

    参数:
        target (str): 目标字符串
        string_list (list): 待比较的字符串列表

    返回:
        tuple: (最相似的字符串, 相似度分数)
    """
    if not string_list:
        return None, 0.0

    # 计算每个字符串与目标字符串的相似度
    similarities = [
        (s, SequenceMatcher(None, target, s).ratio())
        for s in string_list
    ]

    # 找出相似度最高的字符串
    most_similar = max(similarities, key=lambda x: x[1])

    return most_similar[0]


with open(r"C:\Users\Administrator\PythonProjects\ArticleClassification\Data\Summary_insight\key_dict.json", 'r',
          encoding='utf-8') as f:
    class_description = json.load(f)

for review_file in os.listdir(
        r"C:\Users\Administrator\PythonProjects\ArticleClassification\Data\Abstract\AbstractData"):
    # if review_file.split(".")[0] + ".pkl" in os.listdir("../../Results/GPT-4o/DirectClassification"):
    #     continue
    # if review_file in ['Review1.json']:
    #     continue
    print(f"Processing file: {review_file}")

    result = {}

    target_file = fr"C:\Users\Administrator\PythonProjects\ArticleClassification\Data\Abstract\AbstractData/{review_file}"
    with open(target_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keys = list(data.keys())
    # 将keys映射为数字字典，方便后续处理
    keys_dict = {key: i for i, key in enumerate(keys)}

    result['keys_dict'] = keys_dict
    result['original_data'] = data

    # 制作y_true
    y_true = []
    y_pred = []
    for key in keys:
        for article, abstract in data[key].items():
            y_true.append(keys_dict[key])

    system_prompt = """
    You are a meticulous literature researcher. You possess strong thematic analysis capabilities and can determine the category of a paper based on its abstract. Your responsibility is to classify papers according to the abstracts and title provided by users.
    """

    instruction = """
    Please judge which category this article belongs to based on the article title and abstract below.
    Below are the category candidates: %s
    Below are the descriptions of each category: %s

    Article title: %s
    Article abstract: %s

    Return your answer in JSON format. Here is an example of JSON format:
    {
    "Choice": "Your choice",
    "Reason": "The reason you made your choice"
    }
    """

    for key in tqdm(keys, desc="Processing keys"):
        for article, abstract in data[key].items():

            user_prompt = instruction % (
                "; ".join(keys),  # 将keys的值作为分类候选项
                "; ".join([f"{k}: {class_description[review_file[:-5]][k]}" for k in keys]),  # 添加分类描述
                article,
                abstract
            )
            while True:
                try:
                    model_result = GPT4oPromptGenerate(system_prompt, user_prompt)
                    if "```json" in model_result:
                        model_result = model_result.replace("```json", "")
                        model_result = model_result.replace("```", "")
                    # 判断模型选择属于哪一类
                    model_choice = json.loads(model_result)['Choice']
                    # 将模型选择映射回数字字典
                    model_choice_index = keys_dict[model_choice]
                    # 将模型选择的索引添加到y_pred中
                    # if model_choice == key:
                    #     print(f"=====Right=====")
                    # else:
                    #     print(f"=====Wrong===== Model Choice: {model_choice}, right choice: {key}")
                    y_pred.append(model_choice_index)
                    break
                except KeyError:
                    most_similar_key = find_most_similar_string(model_choice, list(keys_dict.keys()))
                    model_choice_index = keys_dict[most_similar_key]
                    y_pred.append(model_choice_index)
                    break
                except Exception as e:
                    # 没有完全一致, 匹配最相似的键
                    print(f"Error processing article {article}: {e}")

    result['y_true'] = y_true
    result['y_pred'] = y_pred

    # 存储结果
    output_file = fr"C:\Users\Administrator\PythonProjects\ArticleClassification\Results\Claude\DirectClassification/{review_file.split('.')[0]}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    # 1. 核心指标计算
    print("分类报告：\n", classification_report(y_true, y_pred, zero_division=0))
    print("准确率：", accuracy_score(y_true, y_pred))
    print("宏平均F1：", f1_score(y_true, y_pred, average='macro'))
    print("加权F1：", f1_score(y_true, y_pred, average='weighted'))
