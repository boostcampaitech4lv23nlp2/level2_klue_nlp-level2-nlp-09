import argparse
import pickle as pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, modeling_outputs

from src.data_loader import REDataset, data_loader
from src.utils import num_to_label


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

        concatenated_vectors = torch.cat((logits_1, logits_2, logits_3), dim=-1)

        output = self.classifier(concatenated_vectors)
        outputs = modeling_outputs.SequenceClassifierOutput(logits=output)
        return outputs  # WARNING!!!! supposed to be outputs


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
                labels=data["labels"].to(device),
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
    model = CombineModels()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # new_tokens = pd.read_csv("src/new_tokens.csv").columns.tolist()
    # new_special_tokens = pd.read_csv("src/special_tokens.csv").columns.tolist()
    # special_tokens_dict = {"additional_special_tokens": new_special_tokens}
    # tokenizer.add_tokens(new_tokens)
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

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
