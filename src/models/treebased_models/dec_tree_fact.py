"""
Child class of Model for the DecisionTreeRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.tree import DecisionTreeRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class DecTreeFactory(ModelFactory):
    """Decision Tree Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "DecisionTree",
            "min_samples_leaf": tune.uniform(0, 0.1),
            "max_leaf_nodes": loguniform_int(10, 5000),
            "min_impurity_decrease": tune.uniform(0.0, 0.1),
            "min_samples_split": tune.uniform(0, 1)
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

        return DecisionTreeRegressor(min_samples_leaf=config['min_samples_leaf'],
                                     max_leaf_nodes=config['max_leaf_nodes'],
                                     min_impurity_decrease=config['min_impurity_decrease'],
                                     min_samples_split=config['min_samples_split'])
