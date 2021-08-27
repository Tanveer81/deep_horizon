"""
Child class of Model for the BayesianRidgeRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.linear_model import BayesianRidge
from models.generic_model import ModelFactory


class BayesianRidgeFactory(ModelFactory):
    """BayesianRidge Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "BayesianRidge",
            "n_iter": tune.choice(np.arange(50, 1000)),
            "fit_intercept": tune.choice([True, False]),
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

        return BayesianRidge(fit_intercept = config['fit_intercept'],
                    n_iter = config['n_iter'],
                    fit_intercept = config['fit_intercept'])
