import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_jsonl_data, predict_categories


def plot_conf_matrix(conf_mat: np.ndarray, le: LabelEncoder) -> None:
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def main(model_path: Path, input_data: Path) -> None:
    df = load_jsonl_data(input_data)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    df, le = predict_categories(model, tokenizer, df, model_path, include_logits=True)

    # Compute confusion matrix
    y_true = df['category'].tolist()
    y_pred = df['predicted_category'].tolist()
    conf_mat = confusion_matrix(y_true, y_pred)

    # Use logits to compute top-k accuracy
    y_true_idx = le.transform(y_true)
    y_pred_logits = df['logits'].tolist()
    top_5_acc = top_k_accuracy_score(y_true_idx, y_pred_logits, k=5)
    top_1_acc = accuracy_score(y_true, y_pred)

    print('Top 1 accuracy: ', top_1_acc)
    print('Top 5 accuracy: ', top_5_acc)

    # Print precision, recall, f_score for each class
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
