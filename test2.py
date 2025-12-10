import os,json


base_path = r"C:\Users\24033\PycharmProjects\ArticleClassification_ForPublish\Data\PDF"

abstract_data_path = "./Data/Abstract/AbstractData"


summary_insight = json.load(open(r"C:\Users\24033\PycharmProjects\ArticleClassification_ForPublish\Data\Summary_insight\Summary_insight.json",encoding="utf-8"))

for review in os.listdir(base_path):
    this_review_path = os.path.join(base_path, review)
    this_title_abstract_path = os.path.join(abstract_data_path, review) + ".json"

    this_title_abstract = json.load(open(this_title_abstract_path,encoding="utf-8"))


    total_category = []
    total_pdf_article = 0
    total_abstract=0
    total_key_insight=0

    for catorory in os.listdir(this_review_path):

        total_abstract += len(this_title_abstract[catorory])
        this_catorory_path = os.path.join(this_review_path, catorory)
        this_category_number = os.listdir(this_catorory_path)
        total_pdf_article += len(os.listdir(this_catorory_path))
        total_category.append(catorory)

        try:
            total_key_insight += len(summary_insight[review][catorory] )
        except KeyError as e:
            continue

    print(f"For {review}, total pdf: {total_pdf_article}, total abstract: {total_abstract},key_insight: {total_key_insight}")
