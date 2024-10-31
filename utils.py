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
    """
    Predict categories for the given text inputs using a pre-trained model.

    Args:
        model (PreTrainedModel): Pre-trained model to use for predictions.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the pre-trained model.
        df (pd.DataFrame): DataFrame containing the text inputs.
        model_path (Path): Path to the directory containing the model and label encoder classes.
        include_logits (bool, optional): Whether to include logits in the output DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with predicted categories (and logits if include_logits is True).
        LabelEncoder: LabelEncoder used to transform the categories.
    """
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
    """
    Load data from a JSONL file into a DataFrame.

    Args:
        path (Path): Path to the JSONL file.

    Returns: pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_json(path, lines=True)
    df['text_input'] = df['headline'] + '\n' + df['short_description']
    return df
