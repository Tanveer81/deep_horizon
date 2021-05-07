"""
Child class of Model for the LightGBMRegressor.
"""

from typing import Any, Mapping
from ray import tune
import numpy as np
from lightgbm import LGBMRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class LightGBMFactory(ModelFactory):
    """LightGBM Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "LightGBM",
            "num_leaves": tune.choice(np.arange(15, 50)),
            "learning_rate": tune.uniform(0.01, 0.5),
            "n_estimators": tune.choice(np.arange(50, 500))
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

        return LGBMRegressor(num_leaves=config['num_leaves'],
                               learning_rate=config['learning_rate'],
                               n_estimators=config['n_estimators'])
