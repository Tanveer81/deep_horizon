"""
Generic code for HPO.

"""

from typing import Mapping, Any, Type
from ray import tune
from ray.tune.logger import MLFLowLogger, DEFAULT_LOGGERS
from ray.tune.schedulers import FIFOScheduler
from mlflow.tracking import MlflowClient
import numpy
import pickle
from os import path
from models.generic_model import ModelFactory
from models.model_analysis import evaluate_model
from data.data_loading import DataPipeline
from data.data_init import init_data
from data import FEATURES, LABELS, FEATURES_NO_FOOT_TYPE, FEATURES_NO_FOOT_TYPE_WITH_TIME
from utils.ml_flow_utils import create_experiment
from data.data_rolling import append_rolling

client = MlflowClient()
outlier = False
test_run = False
pipe = DataPipeline(test=test_run, version=0.23, load='train+test', trans_type="robust", trans_outlier=outlier, data_outlier=outlier)


def train_model(config: Mapping[str, Any]) -> None:
    """Train a model from a given configuration.

    :param config:
        Dictionary containing all HPO informations.
    :return:
        None
    """

    numpy.random.seed(0)

    # Load data
    channel = config['channel']
    x_train, y_train, x_val, y_val, x_test, y_test, _, _, features = pipe.create_dataset(channel=channel)
    
    # Init model
    model_fact = config.pop('fact')
    model = model_fact.from_config(config=config)

    # Train model
    model.fit(x_train, y_train)

    # Evaluate model
    sp_cor_val = evaluate_model(y_val, model.predict(x_val))['sc']
    sp_cor_test = evaluate_model(y_test, model.predict(x_test))['sc']

    # Report to tune
    tune.report(SC_Val=sp_cor_val, SC_test=sp_cor_test)


def hpo_channel(model_fact: Type[ModelFactory], channel: str,
                num_samples: int, exp_name: str) -> Mapping[str, Any]:
    """Perform HPO for a model type to a given channel.

    :param model_fact:
        Class of model factory to optimize. Must be a child class of Model
    :param channel:
        channel of proton intensities to optimize the model for
    :param num_samples:
        number of samples to run HPO for
    :exp_name:
        Name of experiment for logging in mlflow

    :return:
        Dictionary of best found config
    """

    init_data()

    search_space = dict(model_fact.search_space())
    search_space['mlflow_experiment_id'] = create_experiment(name = exp_name + " // channel: " + channel)  # noqa: E501
    search_space['channel'] = channel
    search_space['fact'] = model_fact

    analysis = tune.run(
        train_model,
        num_samples=num_samples,
        verbose=1,
        scheduler=FIFOScheduler(),
        loggers=DEFAULT_LOGGERS + (MLFLowLogger, ),
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config=search_space
    )
    return analysis.get_best_config(metric="SC_Val")


def hpo_all_channels(model_fact: Type[ModelFactory],
                     num_samples: int, exp_name: str) -> None:
    """Perform HPO for a model type for all channels.

    :param model_fact:
        Class of model factory to optimize. Must be a
        child class of ModelFactory
    :param num_samples:
        number of samples per channel to run HPO for
    :exp_name:
        Name of experiment for logging in mlflow

    :return:
        None
    """

    for channel in LABELS[:5]:
        hpo_channel(model_fact, channel, num_samples, exp_name)
