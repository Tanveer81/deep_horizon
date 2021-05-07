"""
    This script defines some helper function for ML FLow.

    @author: jhuthmacher, mmuenzer
"""
from torch import nn
import mlflow
from mlflow.tracking import MlflowClient

from config.config import log

def get_experiment_id(mlflow_client: object, name: str = ""):
    """ Helper function to get the experiment id for a given name

        Parameters
        mlflow_client : MlflowClient
            Instance of ML Flow Client to connect to the current ML Flow
            tracking instance.
        by: str
            Criterion which is used to find the experiment ID.
        name: str
            Name of th experiment for which you want to have the ID (if
            criterion by == "name").

        Returns
        int
            ID of the experiment (or None if experiment doesn't exist)
    """
    if not mlflow_client:
        mlflow_client = MlflowClient()
    for experiment in mlflow_client.list_experiments():
        if experiment.name == name:
            return experiment.experiment_id
    return None


# pylint: disable=redefined-outer-name
def create_experiment(mlflow_client: object = None,
                      name: str = None, artifact_uri: str = None):
    """ Helper function to create a ML Flow experiment.

        Default parameters introduced to not change the method signature.

        This function creates a ML Flow experiment if the name for the
        experiment doesn't exist. Otherwise it will return the ID
        from the experiment that already exists with this name.

        Parameters
            name : str
                Name of the experiment.
            artifact_uri: str
                URI for the artifact tracking

        Returns
            int
                ID of the experiment
    """

    experiment_id: int = get_experiment_id(mlflow_client, name)

    if not experiment_id:
        return mlflow.create_experiment(name, artifact_location=artifact_uri)
    return experiment_id


def init_ml_flow(exp_name: str, client: MlflowClient = None,
                 tracking_uri: str = None,
                 artifact_uri: str = None):  # noqa: E501
    """ Initialization of the ML Flow connection.

        IMPORTANT: Deleting the experiment in the web UI doesn't really
                   delete it, but just set the state to delete in the
                   data base. This means we can't use the same name again!

        Parameters:
            exp_name: str
                ML Flow experiment name.
            client: MlflowClient
                ML Flow client for the current session.
            tracking_uri: str
                URI of the tracking server.
            artifact_uri: str
                URI of the artifact.

        Return:
            MLFlowInstance: Instance of the current ML Flow session.
            int: Experiment ID.
    """
    if not client:
        client = MlflowClient()

    log.info("Overwrite ML Flow tracking URI using 'set_tracking_uri(...)'")  # noqa: E501
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # pylint: disable=no-member
        mlflow.end_rund()
    # pylint: disable=bare-except
    except:  # noqa: E722
        pass

    existing_experiments = MlflowClient().list_experiments()
    exp_id = None
    for exp in existing_experiments:
        if exp.name == exp_name:
            exp_id = exp.experiment_id
            break

    if exp_name not in [exp.name for exp in existing_experiments]:
        exp_id = mlflow.create_experiment(exp_name,
                                          artifact_location=artifact_uri)

    # Set experiment to active!
    mlflow.set_experiment(exp_name)

    return mlflow, exp_id
