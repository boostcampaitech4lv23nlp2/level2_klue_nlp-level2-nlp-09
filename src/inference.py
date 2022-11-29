import argparse
import pickle as pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import REDataset, data_loader
from src.model import RBert, compute_metrics
from src.utils import get_train_valid_split, label_to_num, num_to_label, save_model_remote, set_mlflow_logger, set_seed
from src.utils.custom_trainer import CustomTrainer


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
                subject_mask=data["subject_mask"].to(device),
                object_mask=data["object_mask"].to(device),
                labels=None,
            )
        logits = outputs["logits"]
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
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=30)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    special_tokens_dict = {"additional_special_tokens": ["<S>", "</S>", "<O>", "</O>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    """ model = RBert.from_pretrained(
        data_args.best_model_dir_path,
    )"""
    model = RBert(model_config, tokenizer, model_args.model_name_or_path)
    state_dict = torch.load("model.pt")
    model.load_state_dict(state_dict)
    model.to(device)

    valid_raw_dataset = data_loader(data_args.validation_file_path)
    valid_label = label_to_num(valid_raw_dataset["label"].values)
    valid_dataset = REDataset(valid_raw_dataset, tokenizer, valid_label)
    trainer = CustomTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(valid_dataset)
    print(metrics)

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
