import json, os
import PyPDF2
import numpy as np
from tqdm import tqdm

base_dir = "../Data/PDF"


this_review_class = []
per_class_article_length = []
for review_folder_name in tqdm(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, review_folder_name)
    this_review_class.append(len(os.listdir(folder_path)))

    for cluster_folder_name in os.listdir(folder_path):
        cluster_folder_path = os.path.join(folder_path, cluster_folder_name)
        this_cluster_article_length = len(os.listdir(cluster_folder_path))
        per_class_article_length.append(this_cluster_article_length)

print(f"Total number of review review class: {sum(this_review_class)}")
print(f"Average article number per review class: {np.mean(per_class_article_length)}")