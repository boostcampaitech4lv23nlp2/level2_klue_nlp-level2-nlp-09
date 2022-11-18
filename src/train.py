import os
import pickle as pickle

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    HfArgumentParser,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from data_loader.data_loader import REDataset, data_loader
from model.metric import compute_metrics
from utils.util import DataTrainingArguments, ModelArguments, get_training_args, label_to_num, set_mlflow_logger


def train():
    # Using HfArgumentParser we can turn this class into argparse arguments to be able to specify them on the command line.
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = get_training_args()

    # load dataset
    train_raw_dataset = data_loader(data_args.train_file_path)
    # dev_raw_dataset = data_loader(data_args.validation_file_path) # validation용 데이터는 따로 만드셔야 합니다.

    # label
    train_label = label_to_num(train_raw_dataset["label"].values)
    # dev_label = label_to_num(dev_raw_dataset["label"].values)

    # setting model hyperparameter
    num_labels = len(set(train_label))
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=model_config)

    model.parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Preprocessing the raw_datasets.
    # make dataset for pytorch.
    train_dataset = REDataset(train_raw_dataset, tokenizer, train_label)
    # dev_dataset = REDataset(dev_raw_dataset, tokenizer, dev_label)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")


def main():
    set_mlflow_logger("", "", 0)
    train()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    main()
