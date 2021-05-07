"""
Child class of Model for the LarsRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.linear_model import Lars
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class LarsFactory(ModelFactory):
    """Lars Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "Lars",
            "n_jobs": -1,
            "fit_intercept": tune.choice([True, False]),
            "n_nonzero_coefs": loguniform_int(300, 700),
            "eps": tune.uniform(0, 0.3),
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

        return Lars(fit_intercept = config['fit_intercept'],
                    n_nonzero_coefs = config['n_nonzero_coefs'],
                    eps = config['eps'])
