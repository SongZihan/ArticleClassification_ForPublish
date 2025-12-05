import json, os, sys
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import pickle
import json_repair
from PyPDF2 import PdfReader

from volcenginesdkarkruntime import Ark

# 从环境变量中读取您的方舟API Key
client = Ark(api_key="14195777-fbc3-4414-9eb9-3af208af84be")

def DoubaoPromptGenerate(system_prompt, user_prompt):
    completion = client.chat.completions.create(model="doubao-1.5-pro-32k-250115", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],response_format={"type": "json_object"}
                                                # ,temperature=0.0
                                                )
    return completion.choices[0].message.content


user_prompt = """
You are an expert at parsing academic papers from plain text. Your task is to extract the title and abstract from the following paper text. 
- Identify the title: It is usually the first line or a bold/large heading at the very beginning.
- Identify the abstract: It is typically a section labeled "Abstract", "Summary", or similar, following the title and authors. Extract the full abstract text until it ends (before the introduction or next section).
- If the abstract is missing or unclear, output an empty string for it.
- Do not include authors, keywords, or any other sections.
- Respond ONLY with a valid JSON object in this exact format: 
  {
    "title": "extracted title here",
    "abstract": "extracted abstract text here"
  }
- Ensure the JSON is properly formatted and contains no extra text, explanations, or markdown.
Paper text:
"""

review_files = ["Review14","Review15","Review16","Review17"]
for review_file in tqdm(review_files):
    print(f"Now processing {review_file}=============================")
    target_review_folder = "./PDF/" + review_file
    output_Path = f"./Abstract/AbstractData/{review_file}.json"

    result = {}

    for this_category in os.listdir(target_review_folder):
        result[this_category] = {}
        this_category_path = os.path.join(target_review_folder, this_category)
        for this_file in os.listdir(this_category_path):

            this_file_path = os.path.join(this_category_path, this_file)

            # 创建 PDF 读取器
            if not this_file_path.endswith(".pdf"):
                print(f"{this_file} is not a PDF")
                continue
            try:
                reader = PdfReader(this_file_path)
            except Exception as e:
                print(f"{this_file} is not a valid PDF")
                continue

            # 提取所有页面的文本
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"  # 添加换行符以分隔页面

            used_content = text_content[:2500]

            resp = DoubaoPromptGenerate("You are a helpful assistant.",user_prompt=user_prompt+used_content)
            title_abstract = json_repair.loads(resp)
            title = title_abstract["title"]
            abstract = title_abstract["abstract"]

            result[this_category][title] = abstract

        with open(output_Path,"w") as f:
            json.dump(result, f)
        print(f"{this_category} has been processed~")
