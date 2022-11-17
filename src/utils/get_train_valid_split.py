from typing import Tuple

import json
import random

import numpy as np
import pandas as pd


def get_train_valid_split(
    train_dataset: pd.DataFrame(), valid_size: float = 0.1, random_seed: int = 404
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        dataset (pd.DataFrame): pandas dataframe
        test_size (float, optional): train dataset, valid dataset split ratio Defaults to 0.1.
        random_seed (int, optional): fix seed Defaults to 404.
    """
    np.random.seed(random_seed)

    # load train label distribution
    with open("./utils/distribution.json", "r") as f:
        distribution = json.load(f)

    # valid data length
    length = len(train_dataset) * valid_size

    # train valid split
    valid_dataset = []
    for key, value in distribution.items():
        cnt = round(value * 0.01 * length)
        index = random.sample(list(train_dataset.loc[train_dataset["label"] == key, "id"]), cnt)
        valid_dataset.append(
            train_dataset.loc[
                index,
            ]
        )
        train_dataset = train_dataset.drop(index=index, axis=0)
    valid_dataset = pd.concat(valid_dataset)

    # sort index
    train_dataset = train_dataset.reset_index(drop=True)
    valid_dataset = valid_dataset.reset_index(drop=True)

    return train_dataset, valid_dataset
