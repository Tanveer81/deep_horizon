"""
Generic code for HPO.

"""

from typing import Mapping, Any, Type
from ray import tune
from ray.tune.logger import MLFLowLogger, DEFAULT_LOGGERS
from ray.tune.schedulers import FIFOScheduler
import numpy
from models.generic_model import ModelFactory
from models.model_analysis import evaluate_model
from data.data_loading import DataPipeline
from data.data_init import init_data
from data import FEATURES, LABELS, FEATURES_WITH_TIME, FEATURES_NO_FOOT_TYPE_WITH_TIME
from utils.ml_flow_utils import get_experiment_id


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
    outlier = False
    test_run = False #loads only a small amount of data
    x_train, y_train, x_test, y_test, _, _, features = create_dataset(channel=channel)
    
    # Init model
    model_fact = config.pop('fact')
    model = model_fact.from_config(config=config)

    # Train model
    model.fit(x_train, y_train)

    # Evaluate model
    sp_cor = evaluate_model(y_test, model.predict(x_test))['sc']

    # Report to tune
    # TODO: logging? Only SC or basically everything like we used to? Feature importance, permutation importance, coefs, ...?
    tune.track.log(SC=sp_cor)
    
    
def create_dataset(channel: str, features: list = FEATURES_NO_FOOT_TYPE,
                   features_with_time: list = FEATURES_NO_FOOT_TYPE_WITH_TIME, val: bool = False) -> dict:
    """
    SUMMARY
        Creates x/y datasets including the history input features

    PARAMETERS
        channel --- str: Channel
        features --- list[string]: List of features

    RETURNS
        dict
            Dictionaty containing x, y sets
    """
    # Load data from pipeline
    x_train, y_train = pipe.load_df(split='train', features=features,
                                    labels=[channel], dropna=True)
    time_train = pipe.load_df(split='train', features=features_with_time,
                                    labels=[channel], dropna=True)

    if val:
        x_train, x_test, y_train, y_test= train_test_split(x_train, y_train, test_size=0.2,
                                                random_state=42, shuffle=False)
        time_train, time_test = train_test_split(time_train[0][['DateTime']], test_size=0.2,
                                                random_state=42, shuffle=False)
    else:
        x_test, y_test = pipe.load_df(split='test', features=features,
                                      labels=[channel], dropna=True)
        time_test = pipe.load_df(split='test', features=features_with_time,
                                  labels=[channel], dropna=True)
        time_train = time_train[0][['DateTime']]
        time_test = time_test[0][['DateTime']]
    
    
    # Add the new features to the training set, drop the time for the last one.
    # Time is needed for the pipeline to calculate the average.
    list_avgs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for avg in list_avgs:
        if avg == list_avgs[-1]:
            x_train = append_rolling(x_train, str(avg) + 'h', columns=features, droptime=True)
            x_test = append_rolling(x_test, str(avg) + 'h', columns=features, droptime=True)
            #time_train = append_rolling(time_train, str(avg) + 'h', columns=features, droptime=True)
           # time_test = append_rolling(time_test, str(avg) + 'h', columns=features, droptime=True)
        else:
            x_train = append_rolling(x_train, str(avg) + 'h', columns=features, droptime=False)
            x_test = append_rolling(x_test, str(avg) + 'h', columns=features, droptime=False)
            #time_train = append_rolling(time_train, str(avg) + 'h', columns=features, droptime=False)
           # time_test = append_rolling(time_test, str(avg) + 'h', columns=features, droptime=False)
            
    features = x_train.columns.values

    
    return (x_train, y_train, x_test, y_test, time_train, time_test, features)


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
    search_space['mlflow_experiment_id'] = get_experiment_id(exp_name + channel)  # noqa: E501
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
    return analysis.get_best_config(metric="SC")


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

    for channel in LABELS:
        hpo_channel(model_fact, channel, num_samples, exp_name)
