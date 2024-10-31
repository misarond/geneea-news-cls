import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer


def predict_categories(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       df: pd.DataFrame,
                       model_path: Path,
                       include_logits: bool = False) -> (pd.DataFrame, LabelEncoder):
    le = LabelEncoder()
    le.classes_ = np.load(model_path / 'classes.npy', allow_pickle=True)
    predicted_category = []
    texts = df['text_input'].tolist()
    output_logits = None
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts):
            logits = model(**tokenizer(text, return_tensors='pt')).logits
            if output_logits is None:
                output_logits = logits
            else:
                output_logits = torch.cat((output_logits, logits), dim=0)
            predicted_category.append(le.classes_[(torch.argmax(logits, dim=-1))])
    df['predicted_category'] = predicted_category
    if include_logits:
        output_logits = [el.tolist() for el in output_logits.numpy()]
        df['logits'] = output_logits
    return df, le


def load_jsonl_data(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df['text_input'] = df['headline'] + '\n' + df['short_description']
    return df
