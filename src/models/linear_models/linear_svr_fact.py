"""
Child class of Model for the LinearSVRRegressor.
"""

from typing import Any, Mapping
from ray import tune
import numpy as np
from sklearn.svm import LinearSVR
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class LinearSVRFactory(ModelFactory):
    """LinearSVR Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "LinearSVR",
            "epsilon": tune.uniform(0, 0.5),
            "C": tune.choice(np.arange(1, 100)),
            "loss": tune.choice(["epsilon_insensitive", "squared_epsilon_insensitive"]),
            "max_iter": tune.choice(np.arange(4000, 10000)),
            "fit_intercept": tune.choice([True, False])
        }

    
    @staticmethod
    def from_config(config: Mapping[str, Any]) -> Any:
        """
        Factory method. Returns the model.

        :param config:
            Contains the parameters defined in the search_space of the model.
        :return:
            Model
        """

        return LinearSVR(epsilon=config['epsilon'],
                              C=config['C'],
                              loss=config['loss'],
                              max_iter=config['max_iter'],
                              fit_intercept=config['fit_intercept'])

