"""
Child class of Model for the HistGradientBoostingRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.experimental import enable_hist_gradient_boosting  # noqa # pylint: disable=W0611
from sklearn.ensemble import HistGradientBoostingRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class HistGBRFactory(ModelFactory):
    """Histogram based Gradient Boosting Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "HistGradientBoosting",
            "learning_rate": tune.uniform(0.01, 0.1),
            "max_iter": loguniform_int(50, 500),
            "max_leaf_nodes": loguniform_int(10, 500),
            "min_samples_leaf": loguniform_int(1, 40)
        }

    @staticmethod
    def from_config(config: Mapping[str, Any]) -> Any:
        """
        Initializes the model.

        :param config:
            Contains the parameters defined in the search_space of the model.
        :return:
            None

        """

        return HistGradientBoostingRegressor(learning_rate=config['learning_rate'],
                                             max_iter=config['max_iter'],
                                             max_leaf_nodes=config['max_leaf_nodes'],
                                             min_samples_leaf=config['min_samples_leaf'])
