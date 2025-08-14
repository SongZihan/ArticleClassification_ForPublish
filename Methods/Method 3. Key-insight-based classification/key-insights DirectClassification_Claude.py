"""
直接使用LLM模型进行分类的代码
"""
import json, os
from openai import OpenAI
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# 从环境变量中读取您的方舟API Key
client = OpenAI(base_url="https://openrouter.ai/api/v1",
    api_key="")


def GPT4oPromptGenerate(system_prompt, user_prompt):
    completion = client.chat.completions.create(model="anthropic/claude-sonnet-4", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],response_format={"type": "json_object"}
                                                # ,temperature=0.0
                                                )
    return completion.choices[0].message.content

with open(r"C:\Users\Administrator\PythonProjects\ArticleClassification\Data\Summary_insight\key_dict.json", 'r', encoding='utf-8') as f:
    class_description = json.load(f)


with open(r"C:\Users\Administrator\PythonProjects\ArticleClassification\Data\Summary_insight\Summary_insight.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

    for review_paper_name in data.keys():
        # if review_paper_name in ['Review2','Review3','Review4','Review5','Review6','Review7','Review8','Review9']:
        #     continue
        print(f"Processing file: {review_paper_name}")
        result = {}

        this_review = data[review_paper_name]

        keys = list(this_review.keys())
        # 将keys映射为数字字典，方便后续处理
        keys_dict = {key: i for i, key in enumerate(keys)}
        result['keys_dict'] = keys_dict
        result['original_data'] = this_review

        # 制作y_true
        y_true = []
        y_pred = []
        for key in keys:
            for article, abstract in this_review[key].items():
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
            for article, abstract in this_review[key].items():

                user_prompt = instruction % (
                    "; ".join(keys),  # 将keys的值作为分类候选项
                    "; ".join([f"{k}: {class_description[review_paper_name][k]}" for k in keys]),  # 添加分类描述
                    article,
                    abstract
                )
                while True:
                    try:
                        model_result = GPT4oPromptGenerate(system_prompt, user_prompt)
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
                    except Exception as e:
                        # 没有完全一致, 匹配最相似的键
                        print(f"Error processing article {article}: {e}")

        result['y_true'] = y_true
        result['y_pred'] = y_pred

        # 存储结果
        output_file = fr"C:\Users\Administrator\PythonProjects\ArticleClassification\Results\Claude\key-insights-direction/{review_paper_name}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        # 1. 核心指标计算
        print("分类报告：\n", classification_report(y_true, y_pred, zero_division=0))
        print("准确率：", accuracy_score(y_true, y_pred))
        print("宏平均F1：", f1_score(y_true, y_pred, average='macro'))
        print("加权F1：", f1_score(y_true, y_pred, average='weighted'))

