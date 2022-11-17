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

from data_loader.load_data import REDataset, load_data
from mlflow_logger import set_mlflow_logger
from model.metric import compute_metrics
from utils.util import DataTrainingArguments, ModelArguments, label_to_num


def train():
    # Using HfArgumentParser we can turn this class into argparse arguments to be able to specify them on the command line.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load dataset
    train_raw_dataset = load_data(data_args.train_file_path)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_raw_dataset["label"].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=30)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=model_config)
    print(model.config)

    model.parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)

    # Preprocessing the raw_datasets
    # make dataset for pytorch.
    train_dataset = REDataset(train_raw_dataset, tokenizer, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=1,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )
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
    main()
