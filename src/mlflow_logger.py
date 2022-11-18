import os

import mlflow.pytorch
import yaml


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
        with open("mlflow_config.yml") as f:
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
