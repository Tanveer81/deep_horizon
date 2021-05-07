"""
Child class of Model for the RandomForestRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.ensemble import RandomForestRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


# Pylint rating 9.96: R0801-Error only in this way possible to disable
# pylint: disable-all
class RandForFactory(ModelFactory):
    """Random Forest Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "RandomForest",
            "max_leaf_nodes": loguniform_int(10, 500),
            "min_impurity_decrease": tune.uniform(0.0, 0.1),
            "max_samples": tune.uniform(0.05, 0.3),
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
            Model

        """

        return RandomForestRegressor(n_jobs=-1,
                                     max_leaf_nodes=config['max_leaf_nodes'],
                                     min_impurity_decrease=config['min_impurity_decrease'],
                                     max_samples=config['max_samples'],
                                     n_estimators=config['n_estimators'],
                                     min_samples_split=config['min_samples_split'],
                                     min_samples_leaf=config['min_samples_leaf'])
