import json, os
import PyPDF2
import numpy as np
from tqdm import tqdm

base_dir = "../Data/PDF"

def extract_text_from_pdf(pdf_path):
    # return extract_text(pdf_path)
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

article_length_before = []
article_length = []
for review_folder_name in tqdm(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, review_folder_name)
    for cluster_folder_name in os.listdir(folder_path):
        cluster_folder_path = os.path.join(folder_path, cluster_folder_name)
        for file_name in os.listdir(cluster_folder_path):
            if not file_name.endswith('.pdf'):
                continue

            pdf_path = os.path.join(cluster_folder_path, file_name)
            try:
                text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print("Error reading PDF file:", pdf_path, e)
                continue
            article_length_before.append(len(text))
            # 正则匹配reference, 移除其后面的所有内容
            if 'reference' in text.lower():
                text = text.lower()
                text = text.split('reference')[:-1]
                text = ' '.join(text)
            article_length.append(len(text))


    print(f"Average length of article is {np.mean(article_length_before)}")
    print(f"Std of length of article is {np.std(article_length_before)}")
    print("====================")
    print(f"Average length of article is {np.mean(article_length)}")
    print(f"Std of length of article is {np.std(article_length)}")

print("Overall===================")
print(f"Average length of article is {np.mean(article_length_before)}")
print(f"Std of length of article is {np.std(article_length_before)}")
print("====================")
print(f"Average length of article is {np.mean(article_length)}")
print(f"Std of length of article is {np.std(article_length)}")