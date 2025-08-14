
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_this_performance_list(path):
    file = "Review1.pkl"
    with open(os.path.join(path, file), 'rb') as f:
        data = pickle.load(f)
    y_true = data.get('y_true', [])
    y_pred = data.get('y_pred', [])
    return y_true, y_pred

target_model = ['Grok','GPT-4o','Doubao','Deepseek','Gemini','Claude']
base_path = r"C:\Users\24033\PycharmProjects\ArticleClassification\Results"

result_list = []

for model in target_model:
    ki_path = os.path.join(base_path, model, 'key-insights-direction')

    y_true, y_pred = get_this_performance_list(ki_path)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    result_list.append({
        'Model': model,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    })

df = pd.DataFrame(result_list)
df.to_excel(r"C:\Users\24033\PycharmProjects\ArticleClassification\Results\key_insights_direction_performance.xlsx", index=False)
print(df)