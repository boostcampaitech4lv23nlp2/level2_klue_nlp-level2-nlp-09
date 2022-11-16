from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_valid_split(
    dataset: pd.DataFrame, test_size: float = 0.2, shuffle: bool = True, random_state: int = 404
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        dataset : pandas dataframe
        test_size (float, optional): train, valid dataset split ratio. Defaults to 0.2.
        shuffle (bool, optional): whether mix or not. Defaults to True.
        stratify (bool, optional): label's distribution keep. Defaults to True.
        random_state (int, optional) : fix seed
    """

    train_dataset, valid_dataset = train_test_split(
        dataset, test_size=test_size, shuffle=shuffle, random_state=random_state, stratify=dataset["label"]
    )
    train_dataset = train_dataset.reset_index(drop=True)
    valid_dataset = valid_dataset.reset_index(drop=True)

    return train_dataset, valid_dataset
