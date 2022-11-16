import os

import mlflow.pytorch


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
    except TypeError as e:
        print(e)
        print("##### Type Error!! Plz Check your parameter from train.py")
        print("##### Or Your Train will NOT BE LOGGED ON MLFLOW")
    else:
        print("All Parameters from baseline looks Alright")
        print("Connecting to MLflow...")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog(
            log_every_n_step=logging_step,
        )
    finally:
        print("MLflow setup job finished")
