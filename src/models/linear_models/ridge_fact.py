"""
Child class of Model for the RidgeRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.linear_model import Ridge
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class RidgeFactory(ModelFactory):
    """Ridge Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "Ridge",
            "alphas": tune.uniform(0.1, 2),
            "fit_intercept": tune.choice([True, False]),
            "solver": tune.choice(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
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

        return Ridge(alpha=config['alphas'],
                    fit_intercept=config['fit_intercept'],
                    solver=config['solver'])
