"""
Child class of Model for the MLPRegressor.
"""

from typing import Any, Mapping
from ray import tune
from sklearn.neural_network import MLPRegressor
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int
from utils.ml_utils import sample_array


class MLPFactory(ModelFactory):
    """MLP Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "MLP",
            "number_of_layers": tune.sample_from([2, 5]),
            "hidden_layer_sizes": (tune.sample_from(
                lambda spec: sample_array(1, 8, spec.config.number_of_layers, False))), 
            "alpha": tune.uniform(0, 0.01),
            "batch_size": loguniform_int(32, 256),
            "learning_rate_init": tune.uniform(0.000001, 0.001),
            "max_iter": 20,
            "random_state": 42
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

        return MLPRegressor(hidden_layer_sizes=config['hidden_layer_sizes'],
                            alpha=config['alpha'],
                            batch_size=config['batch_size'],
                            learning_rate_init=config['learning_rate_init'],
                            max_iter=config['max_iter'],
                            random_state=config['random_state'],
                           )
