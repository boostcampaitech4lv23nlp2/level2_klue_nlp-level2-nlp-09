import json
import os
import uuid

import mlflow.pytorch
import pysftp
import yaml

run_id = ""


def set_mlflow_logger(tracking_uri, experiment_name, logging_step):
    """A function sets mlfow logger environments.

    Args:
        tracking_uri (String): A String Data that informs uri of the mlflow site.
                               Usually uses port 5000.
        experiment_name (String): A String Data that informs experiment name at mlflow.
                                  If it doesn't exist at mlflow, it creates one using this name.
        logging_step (Integer) : An Integer Data sets how much steps
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


def save_model_remote(experiment_name, special_word):
    """A function saves best model on remote storage.

    Args:
        experiment_name (String): A String Data that informs experiment name at mlflow.
                                  If a folder which name is this doesn't exist at remote, it creates one using this name.
        special_word (String): A String Data that user can customize the name of the model.
                               User can add anything like hyper_parameter setting, user name, etc.
    """
    model_id = uuid.uuid4().hex

    with open("sftp_config.yml") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    host = config_data["host"]
    port = config_data["port"]
    username = config_data["username"]
    password = config_data["password"]

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    mlflow.log_artifact("best_model/config.json")
    with pysftp.Connection(host, port=port, username=username, password=password, cnopts=cnopts) as sftp:
        print("connected!!")
        sftp.chdir("./mlflow_models")
        try:
            sftp.chdir(experiment_name)
        except IOError:
            sftp.mkdir(experiment_name)
            sftp.chdir(experiment_name)

        model_url = (
            "/home/ubuntu-kyc/mlflow_models/" + experiment_name + "/" + special_word + "_" + model_id + "_model.bin"
        )
        model_url_json = {"model_url": model_url}
        with open("best_model/model_url.json", "w") as json_file:
            json.dump(model_url_json, json_file)
        mlflow.log_artifact("best_model/model_url.json")
        sftp.put("best_model/pytorch_model.bin", special_word + "_" + model_id + "_model.bin")

        print(run_id)
        print("Model Saved on", model_url)
    sftp.close()
