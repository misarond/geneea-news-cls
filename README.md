# Geneea assignment - News classification

This repository contains scripts to train, evaluate and use AI model to classify news categories based on their heading and short description in text form.

## Approach
To solve this task decided to use a pretrained transformer model, to extract features from raw text. Particularly I chose [DistilBERT base model (uncased)](https://huggingface.co/distilbert/distilbert-base-uncased), which is distilled version of [BERT model](https://huggingface.co/google-bert/bert-base-uncased). This transformer model is followed by classification head (linear layer), which is trained for our specific task.

## Installation
In order to run the scripts you need to clone the repository first.

    git clone git@github.com:misarond/geneea-news-cls.git

Navigate yourself to the directory where you cloned the repository and create new virtual environment (this isn't necessary, but it is highly recommended to prevent possible problems with python package compatibilities).

    python -m venv venv

And activating it

    source venv/bin/activate

Now install all necessary packages using requirements.txt file

    pip install -r requirements.txt

## Running the scripts
### Training
First you need to train the model with our data, which is done with `train.py` script. This script requires one argument at first position, which is path to training data in JSONL format. Then you can specify two optional arguments:

* Path to validation data, which can show you how well is the model working on 'not-seen' data during the training process
* Path to `yaml` file with hyperparameters. In this file you can choose your custom hyperparameters for training. See the file `train_hyperparameters.yaml` in repository root, for correct format. It contains few hyperparameters such as learning rate or weight decay, full list of tunable parameters can be found in [documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

Example use

    python train.py data/train.jsonl -vdp data/dev.jsonl -hv train_hyperparameters.yaml

This script prints out path to saved model as an output.


### Evaluation
Then you can evaluate the trained model with validation or test data using `eval.py` script. It requires two arguments
1. Path to model, which you'll get from the output of the `train.py` script.
2. Path to validation/test data in JSONL format.

Note that it is necessary to save also some additional config files to correctly load the trained model, therefore the final model is saved in a folder, and you need to pass a path to the folder to the `eval.py` script, not only the weights in binary format.

This script prints out, top-1 and top-5 accuracy of the model on the provided data and also precision, recall, f-score for each individual category and also plots confusion matrix.

Example use

    python eval.py trained_model data/test.jsonl

### Classification
The script `classify.py` classifies given data and outputs a label (category) for each new. It requires the same arguments as the `eval.py` script, so path to a trained model and path to data in JSONL format. It adds the category labels directly to the provided JSONL file at the input and only prints out confirmation of successful finish of classification process.

Example use

    python classify.py trained_model data/test_no_categories.jsonl.jsonl