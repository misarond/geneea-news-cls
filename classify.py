import argparse
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def load_jsonl_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df['text_input'] = df['headline'] + '\n' + df['short_description']
    return df


def predict_categories(model, tokenizer, df, model_path):
    le = LabelEncoder()
    le.classes_ = np.load(model_path / 'classes.npy', allow_pickle=True)
    predicted_category = []
    texts = df['text_input'].tolist()
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts):
            logits = model(**tokenizer(text, return_tensors='pt')).logits
            predicted_category.append(le.classes_[(torch.argmax(logits, dim=-1))])
        df['category'] = predicted_category
    return df


def main(model_path, input_data):
    df = load_jsonl_data(input_data)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    df = predict_categories(model, tokenizer, df, model_path)
    df.to_json(input_data, lines=True, orient='records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()

    main(args.model, args.data)
