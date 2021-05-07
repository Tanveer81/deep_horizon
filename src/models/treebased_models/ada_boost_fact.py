"""
Child class of Model for the AdaBoostRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.ensemble import AdaBoostRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class AdaBoostFactory(ModelFactory):
    """AdaBoost Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "AdaBoost",
            "n_estimators": loguniform_int(50, 500),
            "learning_rate": tune.uniform(0.3, 1),
            "loss": tune.choice(["linear", "square", "exponential"])
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

        return AdaBoostRegressor(n_estimators=config['n_estimators'],
                                 learning_rate=config['learning_rate'],
                                 loss=config['loss'])
