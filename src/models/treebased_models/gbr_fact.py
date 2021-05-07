"""
Child class of Model for the GradienBoostingRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.ensemble import GradientBoostingRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


# Pylint rating 9.96: R0801-Error only in this way possible to disable
# pylint: disable-all
class GBRFactory(ModelFactory):
    """Gradient Boosting Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "GradientBoosting",
            "loss": tune.choice(["ls", "huber"]),
            "max_leaf_nodes": loguniform_int(10, 500),
            "learning_rate": tune.uniform(0.3, 0.1),
            "min_impurity_decrease": tune.uniform(0.0, 0.1),
            "subsample": tune.uniform(0.8, 1),
            "n_estimators": loguniform_int(50, 500),
            "min_samples_split": tune.uniform(0, 1),
            "min_samples_leaf": tune.uniform(0, 0.4)
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

        return GradientBoostingRegressor(loss=config['loss'],
                                         max_leaf_nodes=config['max_leaf_nodes'],
                                         learning_rate=config['learning_rate'],
                                         min_impurity_decrease=config['min_impurity_decrease'], # noqa # pylint: disable=line-too-long, bad-option-value
                                         subsample=config['subsample'],
                                         n_estimators=config['n_estimators'],
                                         min_samples_split=config['min_samples_split'],
                                         min_samples_leaf=config['min_samples_leaf'])
