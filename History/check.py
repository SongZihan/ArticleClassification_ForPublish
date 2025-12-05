import os, json

for i in range(1,18):
    this_review_file = f"./Abstract/AbstractData/Review{i}.json"
    this_pdf_folder = f"./PDF/Review{i}"

    with open(this_review_file,'r',encoding="utf-8") as f:
        data = json.load(f)


    for category in data.keys():
        this_category_number = len(data[category])
        this_pdf_number = len(os.listdir(f"{this_pdf_folder}/{category}"))
        if this_pdf_number != this_category_number:
            print(f"In Review{i}-{category}, number is not match, pdf number is {this_pdf_number}, category number is {this_category_number}")


