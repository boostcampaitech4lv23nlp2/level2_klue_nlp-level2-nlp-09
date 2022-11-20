import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

PROJECT_ROOT_DIR = os.path.dirname((os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(PROJECT_ROOT_DIR)
from src.data_loader import REDataset, data_loader
from src.model import compute_metrics
from src.utils import label_to_num, set_seed

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


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """

    return dict_num_to_label[label]


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
    entity = row[1:-1].split(",")[0].split(":")[1]
    entity = entity.replace("'", "").strip()
    return entity


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
    new_df["pred_label"] = new_df["pred_label"].apply(num_to_label)
    new_df = new_df.loc[new_df["pred_label"] != new_df["answer"]]
    new_df = new_df[["sentence", "subject", "object", "pred_label", "answer", "probs"]]
    return new_df


def test(args):
    """Perform a test using model of model_dir

    Returns:
        _type_: pd.DataFrame
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    valid_raw_dataset = data_loader(args.valid_file_path)
    valid_label = label_to_num(valid_raw_dataset["label"].values)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    valid_dataset = REDataset(valid_raw_dataset, tokenizer, valid_label)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    pred_answer, output_prob = inference(model, valid_dataset, device)
    outputs = pd.DataFrame(
        {
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    return outputs, metrics
