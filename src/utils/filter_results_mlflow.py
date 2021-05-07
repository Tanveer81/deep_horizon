"""
NAME:
    filter_results_mlflow

DESCRIPTION:
    Module to get info from Mlflow experiments

"""

import json
import mlflow as mlf
from mlflow.tracking import MlflowClient


TRACKING_URI = r'http://10.195.1.103:5000'
ARTIFACT_URI = r'sftp://ubuntu@10.195.1.103/home/ubuntu/mlflow/mlrun'

mlf.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


def create_output_file(models: list, experiment_name: str):
    """
    SUMMARY
        Writes models to a txt file

    PARAMETERS
        models --- list: List of models

    """
    with open(experiment_name + "_models.txt", 'w') as file:
        for model in models:
            file.write(json.dumps(model) + "\n")


def get_best_models(experiment_name):
    """
    SUMMARY
        Returns the best models of an mlflow experiment

    PARAMETERS
        experiment_name --- str: Experiment name

    RETURNS
        logged_models --- list: Models logged in the experiment
    """
    experiment = client.get_experiment_by_name(experiment_name)
    list_run_infos = client.list_run_infos(experiment.experiment_id)

    logged_models = []
    for run_info in list_run_infos:
        run = mlf.get_run(run_info.run_id).to_dictionary()
        if run["data"]["tags"] != {}:
            logged_models.append({
                "name": run["data"]["tags"]["mlflow.runName"],
                "metrics": run["data"]["metrics"],
                "params": run["data"]["params"]
            })

    # create_output_file(logged_models, experiment_name)
    return logged_models
