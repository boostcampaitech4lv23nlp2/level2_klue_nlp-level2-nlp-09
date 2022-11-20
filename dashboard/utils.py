import json
import os
import pickle as pickle

import numpy as np
import pandas as pd
import pysftp
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DICT_NUM_TO_LABEL_PATH = "dashboard/dict_num_to_label.pkl"
with open(DICT_NUM_TO_LABEL_PATH, "rb") as f:
    dict_num_to_label = pickle.load(f)


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = [100 for i in range(len(test_dataset))]
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], tokenized_test, test_label


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_topn_probs(probs, n=3):
    """_summary_

    Args:
        probs (_type_): _description_
        n (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    pairs = []
    top_n_idxs = list(reversed(np.array(probs).argsort()))[:n]
    for idx in top_n_idxs:
        pairs.append((dict_num_to_label[idx], probs[idx]))
    return pairs


def get_entity_word(row):
    """_summary_

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    row = row.replace("'", '"')
    return json.loads(row)["word"]


def get_filtered_result(new_df, test_df):
    """_summary_

    Args:
        new_df (_type_): _description_
        test_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_df["sentence"] = test_df["sentence"]
    new_df["answer"] = test_df["label"]
    new_df["subject"] = test_df["subject_entity"].apply(get_entity_word)
    new_df["object"] = test_df["object_entity"].apply(get_entity_word)
    new_df["probs"] = new_df["probs"].apply(get_topn_probs)
    new_df = new_df.loc[new_df["pred_label"] != new_df["answer"]]
    new_df = new_df[["sentence", "subject", "object", "pred_label", "answer", "probs"]]
    return new_df


def test(args):
    """Perform a test using model of model_dir

    Returns:
        _type_: pd.DataFrame
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    test_id, test_dataset, test_label = load_test_dataset(args.valid_data_path, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    pred_answer, output_prob = inference(model, Re_test_dataset, device)
    pred_answer = num_to_label(pred_answer)
    output = pd.DataFrame(
        {
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    return output


# 이하 후다닥 만든 데모용 SFTP 모듈들. 다듬을 예정.
def connect_remote():
    model_list = []

    with open("src/config/sftp_config.yml") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
        host = config_data["host"]
        port = config_data["port"]
        username = config_data["username"]
        password = config_data["password"]

        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None

        with pysftp.Connection(host, port=port, username=username, password=password, cnopts=cnopts) as sftp:
            print("connected!!")
            files = sftp.listdir_attr("/mlflow_models/Boostcamp KLUE Contest/")
            # print(files)
            for f in files:
                print(f.filename)
                model_list.append(f.filename)
    sftp.close()

    return model_list


def download_model(model_name):
    progressDict = {}
    progressEveryPercent = 10

    for i in range(0, 101):
        if i % progressEveryPercent == 0:
            progressDict[str(i)] = ""

    def printProgressDecimal(x, y):
        """A callback function for show sftp progress log.
           Source: https://stackoverflow.com/questions/24278146/how-do-i-monitor-the-progress-of-a-file-transfer-through-pysftp

        Args:
            x (String): Represent for to-do-data size(e.g. remained file size of pulling of getting)
            y (String): Represent for total file size
        """
        if (
            int(100 * (int(x) / int(y))) % progressEveryPercent == 0
            and progressDict[str(int(100 * (int(x) / int(y))))] == ""
        ):
            print("{}% ({} Transfered(B)/ {} Total File Size(B))".format(str("%.2f" % (100 * (int(x) / int(y)))), x, y))
            progressDict[str(int(100 * (int(x) / int(y))))] = "1"

    with open("src/config/sftp_config.yml") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
        host = config_data["host"]
        port = config_data["port"]
        username = config_data["username"]
        password = config_data["password"]

        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None

        with pysftp.Connection(host, port=port, username=username, password=password, cnopts=cnopts) as sftp:
            print("connected!!")
            # 일단 기본 모델 명을 따랐음
            sftp.get(
                "/mlflow_models/Boostcamp KLUE Contest/" + model_name + "/pytorch_model.bin",
                localpath="dashboard/download_model/pytorch_model.bin",
                callback=lambda x, y: printProgressDecimal(x, y),
            )
            sftp.get(
                "/mlflow_models/Boostcamp KLUE Contest/" + model_name + "/config.json",
                localpath="dashboard/download_model/config.json",
                callback=lambda x, y: printProgressDecimal(x, y),
            )

    sftp.close()
