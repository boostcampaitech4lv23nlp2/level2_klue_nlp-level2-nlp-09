import os

import mlflow.pytorch

from mlflow_config import experiment_name, tracking_uri


def set_mlflow_logger():
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(
        log_every_n_step=100,
    )
    return
