import argparse
import pickle as pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import REDataset, data_loader
from src.utils import num_to_label


def predict(model, tokenized_sent, device):
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

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def inference(model_args, data_args, training_args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(data_args.best_model_dir_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    new_tokens = pd.read_csv("src/new_tokens.csv").columns.tolist()
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # load test datset
    test_raw_dataset = data_loader(data_args.test_file_path)
    test_label = [100 for _ in range(len(test_raw_dataset))]
    test_dataset = REDataset(test_raw_dataset, tokenizer, test_label)

    # predict answer
    pred_answer, output_prob = predict(model, test_dataset, device)
    pred_answer = num_to_label(pred_answer)

    output = pd.DataFrame(
        {
            "id": test_raw_dataset["id"],
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv(data_args.submission_file_path, index=False)
    print("---- Finish! ----")
