from pathlib import Path

import yaml
from transformers import HfArgumentParser

from src import inference, train
from src.utils import DataTrainingArguments, ModelArguments, get_training_args

if __name__ == "__main__":
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    # model_args, data_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser((ModelArguments))
    training_args = get_training_args()

    args = yaml.safe_load(Path("./src/config/config.yml").read_text())
    unused_keys = set(args.keys())

    print(unused_keys)

    # inputs = {k: v for k, v in args.items() if k in keys}
    # unused_keys.difference_update(inputs.keys())

    configTest = parser.parse_yaml_file("./src/config/config.yml")
    print(configTest)
    # print(parser.parse_yaml_file('./src/config/config.yml'))
    # if data_args.do_train:
    #     train(model_args, data_args, training_args)
    # if data_args.do_inference:
    #     inference(model_args, data_args, training_args)
