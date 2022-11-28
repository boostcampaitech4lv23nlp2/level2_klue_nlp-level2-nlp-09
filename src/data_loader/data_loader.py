import pickle as pickle

import pandas as pd
import torch

from src.utils import representation


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    return dataset


def data_loader(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


class REDataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, dataset, tokenizer, labels):
        self.labels = labels
        self.pair_dataset = self.tokenized_dataset(dataset, tokenizer)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def tokenized_dataset(self, dataset, tokenizer):
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02, sentence in zip(dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]):
            temp = representation(
                e01,
                e02,
                sentence,
                entity_method=None,
                is_replace=False,
                translation_methods=[],
            )
            concat_entity.append(temp)
        tokenized_sentences = tokenizer(
            concat_entity,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences
