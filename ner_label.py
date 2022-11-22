from typing import List

import re

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from src.data_loader import REDataset, data_loader
from src.utils import DataTrainingArguments, ModelArguments, get_train_valid_split, get_training_args
from src.utils.representation import extraction


def ner_label(dataset, tokenizer):

    ner_label = []
    for idx, data in tqdm(dataset.iterrows(), total=len(dataset)):

        _, _, subject, subject_entity = extraction(data["subject_entity"])
        _, _, object, object_entity = extraction(data["object_entity"])

        data["sentence"] = data["sentence"].replace("(", "<<")
        data["sentence"] = data["sentence"].replace(")", ">>")

        data["sentence"] = data["sentence"].replace("*", "")

        data["sentence"] = data["sentence"].replace("[", "<<<")
        data["sentence"] = data["sentence"].replace("]", ">>>")

        tokenized_sentence = tokenizer.tokenize(data["sentence"])
        tokenized_sentence = tokenizer.tokenize(data["sentence"])
        tokenized_subject = tokenizer.tokenize(subject, add_special_tokens=False)
        tokenized_object = tokenizer.tokenize(object, add_special_tokens=False)

        labels = []
        for token in tokenized_sentence:

            if re.match(token, subject):
                labels.append(f"B-{subject_entity}")
            elif re.match(token, object):
                labels.append(f"B-{object_entity}")

            elif token in tokenized_subject:
                labels.append(f"I-{subject_entity}")
            elif token in tokenized_object:
                labels.append(f"I-{object_entity}")
            elif token not in tokenized_subject and token not in tokenized_object:
                labels.append("O")

        ner_label.append(labels)
    return ner_label


def get_ner_label():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    train_raw_dataset = data_loader(data_args.train_file_path)
    train_raw_dataset, valid_raw_dataset = get_train_valid_split(train_raw_dataset, valid_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    train_ner_label = ner_label(train_raw_dataset, tokenizer)
    valid_ner_label = ner_label(valid_raw_dataset, tokenizer)

    train_raw_dataset["ner_label"] = train_ner_label
    valid_raw_dataset["ner_label"] = valid_ner_label

    train_raw_dataset.to_csv("./dataset/train/train_ner_label.csv", index=False)
    valid_raw_dataset.to_csv("./dataset/train/valid_ner_label.csv", index=False)


get_ner_label()
