"""
Child class of Model for the LassoRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.linear_model import Lasso
from models.generic_model import ModelFactory


class LassoFactory(ModelFactory):
    """Lasso Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "Lasso",
        	"fit_intercept": tune.choice([True, False]),
            "max_iter": tune.choice(np.arange(500, 2000)),	
            "alpha": tune.uniform(0.1, 2),
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

        return Lasso(fit_intercept = config['fit_intercept'],
                    max_iter = config['max_iter'],
                    alpha = config['alpha'])
