import os, pickle
from sklearn.metrics import classification_report, accuracy_score, f1_score

base_path = r"/Results/Deepseek/key-insights-direction"

accuracy_list = []

for file in os.listdir(base_path):
    with open(os.path.join(base_path, file), 'rb') as f:
        data = pickle.load(f)

    # Assuming data is a dictionary with keys 'y_true' and 'y_pred'
    y_true = data.get('y_true', [])
    y_pred = data.get('y_pred', [])


    # print("分类报告：\n", classification_report(y_true, y_pred, zero_division=0))
    print(file)
    accuracy_list.append(accuracy_score(y_true, y_pred))
    print("准确率：", accuracy_score(y_true, y_pred))
    print("宏平均F1：", f1_score(y_true, y_pred, average='macro'))
    print("===================")
    # print("加权F1：", f1_score(y_true, y_pred, average='weighted'))

print("所有文件的平均准确率：", sum(accuracy_list) / len(accuracy_list))