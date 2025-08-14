import json
import arxiv
from semanticscholar import SemanticScholar
from difflib import SequenceMatcher
import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import ssl
import copy
import time

# # 自定义 SSL 上下文以解决 SSL 握手问题
# class CustomSSLAdapter(HTTPAdapter):
#     def init_poolmanager(self, *args, **kwargs):
#         context = ssl.create_default_context()
#         context.set_ciphers('DEFAULT:@SECLEVEL=1')  # 降低安全级别以兼容旧服务器
#         kwargs['ssl_context'] = context
#         return super(CustomSSLAdapter, self).init_poolmanager(*args, **kwargs)
#
# # 创建会话并添加自定义适配器
# session = requests.Session()
# session.mount('https://', CustomSSLAdapter())
#
# try:
#     response = session.get('https://example.com', timeout=10)
#     # print(response.status_code)
#     # print(response.text)
# except requests.exceptions.SSLError as e:
#     print(f"SSL Error: {e}")
# except requests.exceptions.RequestException as e:
#     print(f"Request failed: {e}")


def get_first_arxiv_id_by_title(title,original_title):
    """
    根据论文标题获取最相关的arXiv论文摘要

    参数:
        title (str): 论文标题

    返回:
        dict: 包含论文标题、摘要、URL等信息的字典
        None: 如果未找到结果
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=f"ti:{title}",  # ti: 表示标题搜索
        max_results=1,  # 只获取最相关的一个结果
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)

    while True:
        try:
            result = list(results)
        except Exception as e:
            time.sleep(5)
            print("Error fetching arXiv ID:", e)
            continue
        break

    if len(result) > 0:
        # 比对标题一致性
        ratio = SequenceMatcher(None, result[0].title.lower(), original_title.lower()).ratio()
        if ratio >= 0.8:
            return result[0]
        else:
            return None
    else:
        return None

base_path = "W:\PythonDoc\ArticleClassification\Data\PDF\Review11_checklist.json"
target_path = "W:\PythonDoc\ArticleClassification\Data\PDF\Review11"
with open(base_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

result = copy.deepcopy(data)
for category, papers in data.items():
    for title, body in papers.items():
        if ":" in title:
            after_title = title.replace(":", "")  # 只取冒号前的部分作为标题
        else:
            after_title = title
        if body == "OK":
            continue

        arxiv_paper = get_first_arxiv_id_by_title(after_title,title)
        if arxiv_paper is not None:
            if not os.path.exists(f"{target_path}/{category}"):
                os.makedirs(f"{target_path}/{category}")

            try:
                arxiv_paper.download_pdf(dirpath = f"{target_path}/{category}")
            except Exception as e:
                print(f"Failed to download PDF for {arxiv_paper.title}, {e}")
                continue
            else:
                print(f"Downloaded: {arxiv_paper.title}.pdf")
            result[category][title] = "OK"
        else:
            print(f"No results found for title: {title}")
            result[category][title] = "Not Found"

        # 保存更新后的JSON文件
        with open(base_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)



