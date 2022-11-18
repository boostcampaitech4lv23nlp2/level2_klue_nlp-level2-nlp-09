from typing import Optional

import os
import pickle as pickle
from dataclasses import dataclass, field

import mlflow.pytorch
import yaml
from transformers import TrainingArguments


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def set_mlflow_logger(tracking_uri, experiment_name, logging_step):
    """A function sets mlfow logger environments.

    :param `tracking_uri`: A String Data that informs uri of the mlflow site.
                           Usually uses port 5000.
    :param `experiment_name`: A String Data that informs experiment name at mlflow.
                              If it doesn't exist at mlflow, it creates one using this name.
    :param `logging_step`: An Integer Data sets how much steps
    """

    try:
        if type(tracking_uri) != str:
            raise TypeError("##### tracking_uri must be a String Type Data!")
        if type(experiment_name) != str:
            raise TypeError("##### experiment_name must be a String Type Data!")
        if type(logging_step) != int:
            raise TypeError("##### logging_step must be a Integer Type Data!")
    except Exception as e:
        print(e)
        raise TypeError("Plz Check your parameter from train.py. Or Your Train will NOT BE LOGGED ON REMOTE")
    else:
        print("All Parameters from baseline looks Alright")
        print("Connecting to MLflow...")
        with open("./config/mlflow_config.yml") as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            print(config_data)
        if tracking_uri == "":
            print("No input for tracking_uri... import Default")
            tracking_uri = config_data["tracking_uri"]
        if experiment_name == "":
            print("No input for experiment_name... import Default")
            experiment_name = config_data["experiment_name"]
        if logging_step < 1:
            print("logging_step cannot be smaller than 1... import Default")
            logging_step = 100

        print("Set Tracking Uri:", tracking_uri)
        print("Set Experiment Name:", experiment_name)
        print("Set Logging Step:", logging_step)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog(
            log_every_n_step=logging_step,
        )
    finally:
        print("MLflow setup job finished")


def get_training_args(
    output_dir="./results",
    save_total_limit=5,
    save_steps=500,
    num_train_epochs=1,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
):
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=save_total_limit,  # number of total save model.
        save_steps=save_steps,  # model saving step.
        num_train_epochs=num_train_epochs,  # total number of training epochs
        learning_rate=learning_rate,  # learning_rate
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
        logging_steps=logging_steps,  # log saving step.
        evaluation_strategy=evaluation_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=eval_steps,  # evaluation step.
        load_best_model_at_end=load_best_model_at_end,
        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    )
    return training_args


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file_path: Optional[str] = field(
        default="../dataset/train/train.csv", metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file_path: Optional[str] = field(
        default="../dataset/train/train.csv", metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
