"""
Child class of Model for the RidgeCVRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.linear_model import RidgeCV
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class RidgeCVFactory(ModelFactory):
    """RidgeCV Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "RidgeCV",
            "alphas": (0.1, 1.0, 10.0),
            "fit_intercept": tune.choice([True, False]),
            "gcv_mode": "auto"
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

        return RidgeCV(alphas=config['alphas'],
                       fit_intercept=config['fit_intercept'],
                        gcv_mode=config['gcv_mode'])


