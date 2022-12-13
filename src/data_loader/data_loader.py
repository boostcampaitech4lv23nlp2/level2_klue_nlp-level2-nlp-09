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
        self.tokenizer = tokenizer
        self.pair_dataset = self.tokenized_dataset(dataset)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_entity_mask(self, input_id, start_token_id, end_token_id):
        start_idx = list(input_id).index(start_token_id)
        end_idx = list(input_id).index(end_token_id)
        entity_mask = [0] * len(input_id)
        for idx in range(start_idx, end_idx + 1):
            entity_mask[idx] = 1
        return entity_mask

    def tokenized_dataset(self, dataset):
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02, sentence in zip(dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]):
            temp = representation(
                e01,
                e02,
                sentence,
                entity_method="typed_entity_marker_punct_custom",
                is_replace=True,
                translation_methods=[None],
            )
            concat_entity.append(temp)

        (
            subject_start_token_id,
            subject_end_token_id,
            object_start_token_id,
            object_end_token_id,
        ) = self.tokenizer.encode("<S> </S> <O> </O>", add_special_tokens=False)

        tokenized_sentences = self.tokenizer(
            concat_entity,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )

        subject_mask_list = []
        object_mask_list = []
        for input_id in tokenized_sentences["input_ids"]:
            subject_entity_mask = self.get_entity_mask(input_id, subject_start_token_id, subject_end_token_id)
            object_entity_mask = self.get_entity_mask(input_id, object_start_token_id, object_end_token_id)
            subject_mask_list.append(subject_entity_mask)
            object_mask_list.append(object_entity_mask)
        tokenized_sentences["subject_mask"] = torch.tensor(subject_mask_list)
        tokenized_sentences["object_mask"] = torch.tensor(object_mask_list)

        return tokenized_sentences
