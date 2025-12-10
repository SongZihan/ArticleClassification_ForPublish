import os


base_path = r"C:\Users\24033\PycharmProjects\ArticleClassification_ForPublish\Data\PDF"

for review in os.listdir(base_path):
    this_review_path = os.path.join(base_path, review)
    total_category = []
    total_pdf_article = 0

    for catorory in os.listdir(this_review_path):
        this_catorory_path = os.path.join(this_review_path, catorory)
        this_category_number = os.listdir(this_catorory_path)
        total_pdf_article += len(os.listdir(this_catorory_path))
        total_category.append(catorory)
    print(f"For {review}, {total_pdf_article}, {', '.join(total_category)}")