import pickle as pickle

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer

from src.data_loader import REDataset, data_loader
from src.model import compute_metrics
from src.utils import get_train_valid_split, label_to_num, save_model_remote, set_mlflow_logger, set_seed


def train(model_args, data_args, training_args):
    # Using HfArgumentParser we can turn this class into argparse arguments to be able to specify them on the command line.
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_mlflow_logger()

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

    model.parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Preprocessing the raw_datasets.
    # make dataset for pytorch.
    train_dataset = REDataset(train_raw_dataset, tokenizer, train_label)
    valid_dataset = REDataset(valid_raw_dataset, tokenizer, valid_label)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(data_args.best_model_dir_path)
    save_model_remote(special_word=data_args.task_name)
