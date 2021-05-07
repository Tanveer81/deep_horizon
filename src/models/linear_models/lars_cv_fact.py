"""
Child class of Model for the LarsCVRegressor.
"""

from typing import Any, Mapping
from ray import tune
import numpy as np
from sklearn.linear_model import LarsCV
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class LarsCVFactory(ModelFactory):
    """LarsCV Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "LarsCV",
            "n_jobs": -1,
            "max_n_alphas": tune.choice(np.arange(500, 3000)),
            "max_iter": tune.choice(np.arange(500, 2000)),
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

        return LarsCV(n_jobs=config['n_jobs'],
                      max_n_alphas=config['max_n_alphas'],
                      max_iter=config['max_iter'],
                      fit_intercept=config['fit_intercept'])




