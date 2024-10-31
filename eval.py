import argparse
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def load_jsonl_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df['text_input'] = df['headline'] + '\n' + df['short_description']
    return df


def predict_categories(model, tokenizer, df, model_path):
    le = LabelEncoder()
    le.classes_ = np.load(model_path / 'classes.npy', allow_pickle=True)
    predicted_category = []
    texts = df['text_input'].tolist()
    output_logits = None
    model.eval()
    with torch.no_grad():
        for text in texts:
            logits = model(**tokenizer(text, return_tensors='pt')).logits
            if output_logits is None:
                output_logits = logits
            else:
                output_logits = torch.cat((output_logits, logits), dim=0)
            predicted_category.append(le.classes_[(torch.argmax(logits, dim=-1))])
    df['predicted_category'] = predicted_category
    output_logits = [el.tolist() for el in output_logits.numpy()]
    df['logits'] = output_logits
    return df, le


def plot_conf_matrix(conf_mat, le):
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main(model_path, input_data):
    df = load_jsonl_data(input_data)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    df, le = predict_categories(model, tokenizer, df, model_path)

    # Compute confusion matrix
    y_true = df['category'].tolist()
    y_pred = df['predicted_category'].tolist()
    conf_mat = confusion_matrix(y_true, y_pred)

    y_true_idx = le.transform(y_true)
    y_pred_logits = df['logits'].tolist()
    top_5_acc = top_k_accuracy_score(y_true_idx, y_pred_logits, k=5)
    top_1_acc = accuracy_score(y_true, y_pred)

    print('Top 1 accuracy: ', top_1_acc)
    print('Top 5 accuracy: ', top_5_acc)

    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0.0)

    results_df = pd.DataFrame({'Category': le.classes_, 'Precision': precision, 'Recall': recall, 'F-score': f_score})
    print(results_df.to_markdown(index=False))

    # Plot confusion matrix
    plot_conf_matrix(conf_mat, le)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()

    main(args.model, args.data)
