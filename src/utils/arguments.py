from typing import Optional

import pickle as pickle
from dataclasses import dataclass, field

from transformers import TrainingArguments


def get_training_args(
    output_dir="./results",
    save_total_limit=5,
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=458,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_micro_f1_score",
    fp16=True,
):
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=save_total_limit,  # number of total save model.
        save_strategy=save_strategy,
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
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        fp16=fp16
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
        default="RE_baseline",
        metadata={"help": "The name of the task to train"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    seed: int = field(default=200)
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Execute Training for model"})
    do_inference: bool = field(default=False, metadata={"help": "Execute Inference for model. Default is False."})
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
        default="dataset/train/train.csv", metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file_path: Optional[str] = field(
        default="dataset/train/valid.csv", metadata={"help": "A csv or a json file containing the validation data."}
    )
    best_model_dir_path: Optional[str] = field(
        default="src/best_model", metadata={"help": "A diretory containing the best model to save"}
    )
    test_file_path: Optional[str] = field(
        default="dataset/test/test_data.csv", metadata={"help": "A csv or a json file containing the test data."}
    )
    submission_file_path: Optional[str] = field(
        default="src/prediction/submission.csv",
        metadata={"help": "A csv or a json file containing the submission data."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
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
