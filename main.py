from pathlib import Path

import yaml
from transformers import HfArgumentParser

from config import ModelArgument
from src import inference, train
from src.utils import DataTrainingArguments, ModelArguments, get_training_args

if __name__ == "__main__":
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    # model_args, data_args = parser.parse_args_into_dataclasses()
    modelArgument = ModelArgument()
    modelArgument.load()
    print(modelArgument.model_name_or_path)

    # parser = HfArgumentParser((ModelArguments,DataTrainingArguments))

    # print(parser.parse_yaml_file('./config.yml'))

    # print(parser)
    # model_arg = parser.parse_args_into_dataclasses()[0]
    # print(model_arg)
    # model_arg.model_name_or_path = 'Bert'
    # print(model_arg)
    # training_args = get_training_args()

    # args = yaml.safe_load(Path("./config.yml").read_text())
    # configTest = parser.parse_yaml_file("./config.yml", True)

    # parsers = HfArgumentParser(configTest)

    # print(parsers)
    # print(parsers.parse_args_into_dataclasses())

    # inputs = {k: v for k, v in args.items() if k in keys}
    # unused_keys.difference_update(inputs.keys())

    # print(parser.parse_yaml_file('./src/config/config.yml'))
    # if data_args.do_train:
    #     train(model_args, data_args, training_args)
    # if data_args.do_inference:
    #     inference(model_args, data_args, training_args)
