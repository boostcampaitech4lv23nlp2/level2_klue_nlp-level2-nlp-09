import pickle as pickle

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer

from src.data_loader import REDataset, data_loader
from src.model import compute_metrics
from src.utils import get_train_valid_split, label_to_num, save_model_remote, set_mlflow_logger, set_seed
from src.utils.custom_trainer import CustomTrainer


def train(model_args, data_args, training_args):
    # Using HfArgumentParser we can turn this class into argparse arguments to be able to specify them on the command line.
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(data_args.seed)

    # load dataset
    train_raw_dataset = data_loader(data_args.train_file_path)
    # dev_raw_dataset = data_loader(data_args.validation_file_path) # validation용 데이터는 따로 만드셔야 합니다.

    train_raw_dataset, valid_raw_dataset = get_train_valid_split(train_raw_dataset, valid_size=0.1)
    valid_raw_dataset.to_csv(data_args.validation_file_path, index=False)

    # label
    train_label = label_to_num(train_raw_dataset["label"].values)
    valid_label = label_to_num(valid_raw_dataset["label"].values)

    # setting model hyperparameter
    num_labels = len(set(train_label))
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=model_config)

    new_tokens = pd.read_csv("src/new_tokens.csv").columns.tolist()
    new_special_tokens = pd.read_csv("src/special_tokens.csv").columns.tolist()
    special_tokens_dict = {"additional_special_tokens": new_special_tokens}
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model.parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Preprocessing the raw_datasets.
    # make dataset for pytorch.
    train_dataset = REDataset(train_raw_dataset, tokenizer, train_label)
    valid_dataset = REDataset(valid_raw_dataset, tokenizer, valid_label)

    trainer = CustomTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    special_word = data_args.task_name
    tracking_uri = ""
    experiment_name = ""
    logging_step = 100

    model_id = set_mlflow_logger(special_word, tracking_uri, experiment_name, logging_step)
    trainer.train()
    model.save_pretrained(data_args.best_model_dir_path)
    save_model_remote(experiment_name, model_id)
