import pickle as pickle

import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    modeling_outputs,
)

from src.data_loader import REDataset, data_loader
from src.model import compute_metrics
from src.utils import get_train_valid_split, label_to_num, save_model_remote, set_mlflow_logger, set_seed
from src.utils.custom_trainer import CustomTrainer


class CombineModels(nn.Module):
    """
    edit by 이요한_T2166
    """

    def __init__(self):
        super(CombineModels, self).__init__()

        c1 = AutoConfig.from_pretrained("klue/roberta-large", num_labels=2)
        c2 = AutoConfig.from_pretrained("klue/roberta-large", num_labels=29)
        c3 = AutoConfig.from_pretrained("klue/roberta-large", num_labels=30)

        self.roberta1 = AutoModelForSequenceClassification.from_pretrained("src/best_model/2_relations", config=c1)
        self.roberta2 = AutoModelForSequenceClassification.from_pretrained("src/best_model/29_relations", config=c2)
        self.roberta3 = AutoModelForSequenceClassification.from_pretrained("src/best_model/30_relations", config=c3)

        for p in self.roberta1.parameters():
            p.requires_grad = False
        for p in self.roberta2.parameters():
            p.requires_grad = False
        for p in self.roberta3.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(2, 768)
        self.fc2 = nn.Linear(29, 768)
        self.fc3 = nn.Linear(30, 768)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768 * 3, 768, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, 30, bias=True),
        )

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        logits_1 = self.roberta1(input_ids.clone(), attention_mask=attention_mask).get("logits")
        logits_2 = self.roberta2(input_ids.clone(), attention_mask=attention_mask).get("logits")
        logits_3 = self.roberta3(input_ids.clone(), attention_mask=attention_mask).get("logits")

        logits_1 = self.fc1(logits_1)
        logits_2 = self.fc2(logits_2)
        logits_3 = self.fc3(logits_3)

        self.labels = labels

        concatenated_vectors = torch.cat((logits_1, logits_2, logits_3), dim=-1)

        output = self.classifier(concatenated_vectors)
        outputs = modeling_outputs.SequenceClassifierOutput(logits=output)
        return outputs  # WARNING!!!! supposed to be outputs


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
    # num_labels = len(set(train_label))
    # model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # model
    # model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=model_config)
    model = CombineModels()

    # new_tokens = pd.read_csv("src/new_tokens.csv").columns.tolist()
    # new_special_tokens = pd.read_csv("src/special_tokens.csv").columns.tolist()
    # special_tokens_dict = {"additional_special_tokens": new_special_tokens}
    # tokenizer.add_tokens(new_tokens)
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

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
    trainer.save_model("src/best_model")
    # model.save_pretrained(data_args.best_model_dir_path)
    torch.save(model.state_dict(), "model.pt")
    save_model_remote(experiment_name, model_id)
