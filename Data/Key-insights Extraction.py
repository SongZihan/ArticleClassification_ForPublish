# 提取key-insight
import json, os
from openai import OpenAI
from tqdm import tqdm
import logging
from pdfminer.high_level import extract_text
import PyPDF2

# client = OpenAI(  base_url="https://openrouter.ai/api/v1",
#     api_key="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")
#
# def GPT4oPromptGenerate(system_prompt, user_prompt):
#     completion = client.chat.completions.create(model="openai/gpt-4o", messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ],
#     temperature=0.7, top_p=0.95
#                                                 )
#     return completion.choices[0].message.content


from volcenginesdkarkruntime import Ark
client = Ark(api_key="14195777-fbc3-4414-9eb9-3af208af84be")

def GPT4oPromptGenerate(system_prompt, user_prompt):
    completion = client.chat.completions.create(model="doubao-1.5-pro-32k-250115", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],response_format={"type": "json_object"}
                                                # ,temperature=0.0
                                                )
    return completion.choices[0].message.content



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


"""
Prompt1: 提取key-insight
"""

# 你是一位细致的文献研究者。你拥有很强的主题分析能力，能够准确地提取关键信息和核心观点。你的职责是根据用户需求从文献中总结相关信息。
system_prompt = """
You are a meticulous literature researcher. You possess strong thematic analysis skills, enabling you to accurately extract key information and core points. """

instruction = """
Start with the file name in quotes, followed by a short overview paragraph (2–4 sentences) describing the paper’s topic, main contribution, and approach.
After the overview, add the heading ### Key Points:.
Present the key insights as a numbered list.
Each numbered item should have:
A bolded title describing the section (e.g., Motivation and Background, Methodology, Experimental Results, Ablation Studies and Analysis, Limitations, Conclusion).
Several bullet points or sentences elaborating on that aspect in your own words (avoid direct quotations from the paper).
Cover at least the following aspects:
Motivation and Background: Why the research was done, context, prior work, and the problem addressed.
Methodology: How the proposed method works and what makes it unique.
Experimental Results: Key benchmarks, datasets, model performance, and comparisons.
Ablation Studies and Analysis: Additional experiments, observations, and explanations.
Limitations: Weaknesses, assumptions, or untested scenarios.
Conclusion: Final takeaways, contributions, and potential future work.
Keep the writing concise, factual, and clear. Summarize complex ideas in plain English without omitting crucial technical details.
Research manuscript: [Research manuscript]
"""

base_dir = "../Data/PDF"
out_file = "../Data/Summary_insight/Summary_insight.json"

# 读取out_file
with open(out_file, 'r', encoding='utf-8') as f:
    result = json.load(f)


for review_folder_name in tqdm(os.listdir(base_dir)):
    if review_folder_name not in ["Review13","Review14","Review15","Review16","Review17"]:
        continue

    folder_path = os.path.join(base_dir, review_folder_name)
    if review_folder_name not in result:
        result[review_folder_name] = {}
    for cluster_folder_name in os.listdir(folder_path):
        cluster_folder_path = os.path.join(folder_path, cluster_folder_name)
        if cluster_folder_name not in result[review_folder_name]:
            result[review_folder_name][cluster_folder_name] = {}

        for file_name in os.listdir(cluster_folder_path):
            if not file_name.endswith('.pdf'):
                continue
            if file_name in result[review_folder_name][cluster_folder_name]:
                continue


            pdf_path = os.path.join(cluster_folder_path, file_name)
            try:
                text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print("Error reading PDF file:", pdf_path, e)
                continue
            # 正则匹配reference, 移除其后面的所有内容
            if 'reference' in text.lower():
                text = text.lower()
                text = text.split('reference')[:-1]
                text = ' '.join(text)

            try:
                print(f"Now extracting {file_name} in {cluster_folder_name} in {review_folder_name}")
                user_prompt = instruction + text
                response = GPT4oPromptGenerate(system_prompt, user_prompt)
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                continue

            result[review_folder_name][cluster_folder_name][file_name] = response


            # 将结果写入out_file
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)









