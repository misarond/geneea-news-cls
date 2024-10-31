from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import yaml


def load_jsonl_data(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # Keep only important columns (category, headline, short_description)
    df = df[['category', 'headline', 'short_description']]
    df['text_input'] = df['headline'] + '\n' + df['short_description']
    return df


def preprocess_data(df: pd.DataFrame, le: LabelEncoder = None) -> (pd.DataFrame, LabelEncoder):
    if le is None:
        le = LabelEncoder()
        df['labels'] = le.fit_transform(df['category'])
    else:
        df['labels'] = le.transform(df['category'])
    dataset = Dataset.from_pandas(df[['text_input', 'labels']])
    return dataset, le


# Ensure the dataset has the features the model expects
def prepare_dataset(dataset, tokenizer):

    # Pre-processing function
    def preprocess_function(examples):
        return tokenizer(examples['text_input'], truncation=True, padding=True)

    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.with_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


def train_model(model, train_dataset, val_dataset, tokenizer, hyperparameters):
    # Load hyperparameter values if they were provided
    if hyperparameters:
        with open(hyperparameters, 'r') as f:
            params = yaml.safe_load(f)
        # Training Arguments
        training_args = TrainingArguments(output_dir='./results', evaluation_strategy="epoch", **params)
    else:
        # Training Arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=2,
            weight_decay=0.05,
        )

    # Create Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    return trainer


def compute_metrics(eval_pred):
    # Compute metrics
    metric = load("accuracy")
    logits, labels = eval_pred
    logits_tensor = torch.tensor(logits)
    predictions = torch.argmax(logits_tensor, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(train_data: Path, val_data: Path, hyperparameters: Path) -> None:
    # Check if paths exist
    if not train_data.exists():
        raise FileNotFoundError(train_data)
    if val_data and not val_data.exists():
        warnings.warn("Warning: Inserted path to validation data does not exist! Skipping validation.")
        val_data = None
    if hyperparameters and not hyperparameters.exists():
        warnings.warn("Warning: Inserted path to hyperparameters data does not exist! Skipping validation.")
        hyperparameters = None

    # Load datasets
    df_train = load_jsonl_data(train_data)
    df_val = load_jsonl_data(val_data) if val_data else None

    train_dataset, le = preprocess_data(df_train)
    val_dataset, _ = preprocess_data(df_val) if val_data else None

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    val_dataset = prepare_dataset(val_dataset, tokenizer) if val_data else None

    # Model
    num_classes = len(le.classes_)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_classes)

    trainer = train_model(model, train_dataset, val_dataset, tokenizer, hyperparameters)

    # Evaluate the model
    trainer.evaluate()

    trainer.save_model("./trained_model")
    np.save('trained_model/classes.npy', le.classes_)

    # Predict labels for validation dataset
    predictions, labels, _ = trainer.predict(val_dataset)

    # Get predicted class labels
    predicted_labels = torch.argmax(torch.tensor(predictions), dim=-1)

    # Compute confusion matrix
    conf_mat = confusion_matrix(labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("train_data_path", type=Path)
    parser.add_argument("-vdp", "--validation_data_path", default=None, type=Path)
    parser.add_argument("-hv", "--hyperparameters_values", default=None, type=Path)

    args = parser.parse_args()
    main(args.train_data_path, args.validation_data_path, args.hyperparameters_values)
