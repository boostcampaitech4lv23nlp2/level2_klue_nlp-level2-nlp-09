import json
import os
import pickle as pickle
import uuid

import mlflow
import pysftp
import yaml
from transformers.integrations import MLflowCallback


def end_train():
    mlflow.end_run()
    print("Train Finished... The model will be saved on remote")


def set_mlflow_logger(special_word="", tracking_uri="", experiment_name="", logging_step=0):
    """A function sets mlfow logger environments.

    Args:
        tracking_uri (String): A String Data that informs uri of the mlflow site.
                           Usually uses port 5000.
        experiment_name (String): A String Data that informs experiment name at mlflow.
                              If it doesn't exist at mlflow, it creates one using this name.
        logging_step (String): An Integer Data sets per how much steps
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
        with open("src/config/mlflow_config.yml") as f:
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

        model_id = special_word + "_" + uuid.uuid4().hex

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        os.environ["MLFLOW_TAGS"] = '{"mlflow.runName": "' + model_id + '"}'
        mlflow.doctor()
    finally:
        print("MLflow setup job finished")
        return model_id


def save_model_remote(experiment_name="", model_id=""):
    """A function saves best model on remote storage.

    Args:
        experiment_name (String): A String Data that informs experiment name at mlflow.
                                  If a folder which name is this doesn't exist at remote, it creates one using this name.
        special_word (String): A String Data that user can customize the name of the model.
                               User can add anything like hyper_parameter setting, user name, etc.
    """
    try:
        if "/" in model_id:
            raise TypeError("##### special_word cannot include a character '/'")
    except Exception as e:
        print(e)
        raise TypeError(
            "Plz Check your parameter from train.py. \nThe model you trained may saved locally. \nSo, you can put on remote manually"
        )
    else:
        with open("src/config/mlflow_config.yml") as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            print(config_data)
        if experiment_name == "":
            print("No input for experiment_name... import Default")
            experiment_name = config_data["experiment_name"]

        progressDict = {}
        progressEveryPercent = 10

        for i in range(0, 101):
            if i % progressEveryPercent == 0:
                progressDict[str(i)] = ""

        def printProgressDecimal(x, y):
            """A callback function for show sftp progress log.
            Source: https://stackoverflow.com/questions/24278146/how-do-i-monitor-the-progress-of-a-file-transfer-through-pysftp

            Args:
                x (String): Represent for to-do-data size(e.g. remained file size of pulling of getting)
                y (String): Represent for total file size
            """
            if (
                int(100 * (int(x) / int(y))) % progressEveryPercent == 0
                and progressDict[str(int(100 * (int(x) / int(y))))] == ""
            ):
                print(
                    "{}% ({} Transfered(B)/ {} Total File Size(B))".format(
                        str("%.2f" % (100 * (int(x) / int(y)))), x, y
                    )
                )
                progressDict[str(int(100 * (int(x) / int(y))))] = "1"

        with open("src/config/sftp_config.yml") as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
        host = config_data["host"]
        port = config_data["port"]
        username = config_data["username"]
        password = config_data["password"]

        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None

        with pysftp.Connection(host, port=port, username=username, password=password, cnopts=cnopts) as sftp:
            print("connected!!")
            sftp.chdir("./mlflow_models")
            try:
                sftp.chdir(experiment_name)
            except IOError:
                sftp.mkdir(experiment_name)
                sftp.chdir(experiment_name)
            sftp.mkdir(model_id)
            sftp.chdir(model_id)

            model_url = "/mlflow_models/" + experiment_name + "/" + model_id
            sftp.put(
                localpath="src/best_model/pytorch_model.bin",
                remotepath="pytorch_model.bin",
                callback=lambda x, y: printProgressDecimal(x, y),
            )
            sftp.put(
                localpath="src/best_model/config.json",
                remotepath="config.json",
                callback=lambda x, y: printProgressDecimal(x, y),
            )
            print("Success!!! Model Saved on", model_url)
        sftp.close()
    finally:
        print("Model Saving job Finished")
