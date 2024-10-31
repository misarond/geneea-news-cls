import argparse
from pathlib import Path
from utils import load_jsonl_data, predict_categories
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def main(model_path: Path, input_data: Path) -> None:
    df = load_jsonl_data(input_data)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    df, _ = predict_categories(model, tokenizer, df, model_path)
    df.to_json(input_data, lines=True, orient='records')
    print('Data successfully labeled and saved in ', input_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()

    main(args.model, args.data)
