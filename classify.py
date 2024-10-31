import argparse
from pathlib import Path
from utils import load_jsonl_data, predict_categories
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def main(model_path: Path, input_data: Path) -> None:
    """
    Main function to load data, predict categories, and save the labeled data.

    Args:
        model_path (Path): Path to the pre-trained model and tokenizer.
        input_data (Path): Path to the input data in JSONL format.
    """
    df = load_jsonl_data(input_data)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    df, _ = predict_categories(model, tokenizer, df, model_path)
    # Change column name to category
    df = df.rename(columns={'predicted_category': 'category'})
    df.to_json(input_data, lines=True, orient='records')
    print('Data successfully labeled and saved in ', input_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for predicting categories using a trained DistilBERT "
                                                 "model.")
    parser.add_argument('model', type=Path, help="Path to the pre-trained model and tokenizer.")
    parser.add_argument('data', type=Path, help="Path to the input data in JSONL format.")
    args = parser.parse_args()

    main(args.model, args.data)
