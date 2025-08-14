import json, os
from cProfile import label

from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def map_clusters_to_labels(true_labels, cluster_labels):
    """使用匈牙利算法匹配聚类簇到真实标签"""
    conf_mat = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(conf_mat, maximize=True)  # 最大化匹配
    return {cluster: true_label for cluster, true_label in zip(col_ind, row_ind)}

accuracy_list = []

base_path = "../../Data/Abstract/AbstractEmebedding"
for file in os.listdir(base_path):
    with open(os.path.join(base_path, file), 'rb') as f:
        data = pickle.load(f)

    # 将keys映射为数字字典，方便后续处理
    keys_dict = {key: i for i, key in enumerate(list(data.keys()))}

    label_list = []
    value_list = []
    title_list = [] # 根据顺序存储的论文标题列表
    for class_label in data.keys():
        for title, abstract_embedding in data[class_label].items():
            label_list.append(keys_dict[class_label])
            value_list.append(abstract_embedding)
            title_list.append(title)

    # PCA降维（保留90%主成分）
    pca = PCA(n_components=0.9, random_state=1)  # 设置n_components=0.9表示保留90%的方差
    value_list_pca = pca.fit_transform(value_list)  # 对value_list进行PCA降维

    print(f"原始维度: {len(value_list[0])}")
    print(f"降维后维度: {pca.n_components_}")  # 显示实际保留的主成分数量
    print(f"保留的方差比例: {sum(pca.explained_variance_ratio_):.2%}")  # 显示实际保留的方差比例

    kmeans_model = KMeans(n_clusters=len(keys_dict), random_state=1, n_init='auto').fit(value_list_pca)
    cluster_labels = kmeans_model.fit_predict(value_list_pca)

    # 将聚类结果与标题对应
    # clustered_titles = {i: [] for i in range(len(keys_dict))}
    # for title, label in zip(title_list, cluster_labels):
    #     clustered_titles[label].append(title)

    # 根据重合度判断每个cluster数据所属类别
    mapping = map_clusters_to_labels(label_list, cluster_labels)
    # 将聚类标签映射到真实标签
    remapping_cluster_labels = [mapping[label] for label in cluster_labels]


    y_true = label_list
    y_pred = remapping_cluster_labels
    # 评价分类准确性
    accuracy_list.append(accuracy_score(y_true, y_pred))
    print("准确率：", accuracy_score(y_true, y_pred))
    print("宏平均F1：", f1_score(y_true, y_pred, average='macro'))
    print("===================")

    result = {}
    result['y_true'] = y_true
    result['y_pred'] = y_pred
    result['keys_dict'] = keys_dict
    result['original_data'] = data
    output_file = f"./Results\JustKmeans/{file}"
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

print("所有文件的平均准确率：", sum(accuracy_list) / len(accuracy_list))

