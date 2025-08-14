import re
from typing import Tuple, Optional
import json, os
import PyPDF2
import numpy as np
from tqdm import tqdm

def is_arxiv_article(filename: str) -> bool:
    """
    判断文件名是否是arXiv文章

    Args:
        filename: 文件名字符串

    Returns:
        bool: 如果是arXiv文章返回True，否则返回False
    """
    # 移除文件扩展名
    base_name = filename.lower().strip()
    if '.' in base_name:
        base_name = '.'.join(base_name.split('.')[:-1])

    # arXiv ID的几种常见模式
    patterns = [
        # 新格式: YYMM.NNNNN 或 YYMM.NNNNNvN (2007年4月后)
        r'^\d{4}\.\d{4,5}(v\d+)?$',

        # 旧格式: archive.subject/YYMMNNN 或 archive.subject/YYMMNNNvN
        r'^[a-z\-]+(\.[a-z\-]+)?\/\d{7}(v\d+)?$',

        # 也匹配包含 "arxiv" 关键词的情况
        r'arxiv[\-_\.]?\d{4}\.\d{4,5}(v\d+)?',
        r'arxiv[\-_\.]?[a-z\-]+(\.[a-z\-]+)?\/\d{7}(v\d+)?',
    ]

    for pattern in patterns:
        if re.search(pattern, base_name):
            return True

    # 检查是否包含arXiv URL
    if 'arxiv.org' in filename.lower():
        return True

    return False

base_dir = "../Data/PDF"

arxiv_number = 0
non_arxiv_number = 0
for review_folder_name in tqdm(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, review_folder_name)
    for cluster_folder_name in os.listdir(folder_path):
        cluster_folder_path = os.path.join(folder_path, cluster_folder_name)
        for file_name in os.listdir(cluster_folder_path):
            if not file_name.endswith('.pdf'):
                continue
            if is_arxiv_article(file_name):
                arxiv_number += 1
            else:
                non_arxiv_number += 1

print(f"Arxiv rate is {arxiv_number / (non_arxiv_number+arxiv_number) * 100}%")