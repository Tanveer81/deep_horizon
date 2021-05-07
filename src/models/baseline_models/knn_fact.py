"""
Child class of Model for the KNeighborsRegressor.
"""

from typing import Any, Mapping
from ray import tune
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class KNeighborsFactory(ModelFactory):
    """KNeighbors Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "kNN",
            "n_jobs": -1,
            "n_neighbors": tune.choice(np.arange(1, 10)),
            "leaf_size": tune.choice(np.arange(20, 50)),
            "weights": tune.choice(["uniform", "distance"])
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

        return KNeighborsRegressor(n_jobs=config['n_jobs'],
                        n_neighbors=config['n_neighbors'],
                        leaf_size=config['leaf_size'],
                        weights=config['weights'])


