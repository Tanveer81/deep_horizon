"""
Module for the historical binning baseline based on kmeans.
"""

from typing import Any, Mapping
import numpy
from sklearn.cluster import KMeans
from models.generic_model import ModelFactory
from utils.tune_utils import loguniform_int


class HistBin():
    """Regression model"""

    def __init__(self, num_bins: int):
        self.cluster_mean = None
        self.model = KMeans(n_clusters=num_bins, n_jobs=-1)

    def fit(self, x_train: numpy.ndarray, y_train: numpy.ndarray) -> None:
        """
        Train the model.

        :param x_train: shape: (num_samples, input_dim)
            The input comprising OMNI data and position.
        :param y_train: shape: (num_samples,)
            The output data.
        """
        # Filter x_train to only contain positional data
        x_train = x_train[:, :4]

        # idx_train the index of the cluster the i-th training point belongs to
        idx_train = self.model.fit_predict(x_train)

        # Compute historical means for all clusters
        cluster_size = numpy.bincount(idx_train)
        cluster_sum = numpy.stack([numpy.bincount(idx_train, weights=y_train[:, i]) for i in range(y_train.shape[1])], axis=-1) # noqa # pylint: disable=line-too-long, bad-option-value
        self.cluster_mean = cluster_sum[:, 0] / cluster_size

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        """
        Make a prediction.

        :param x_test: shape: (num_samples, input_dim)
            The input comprising OMNI data and position.

        :return: shape: (num_samples,)
            The predictions.
        """
        # Filter x_train to only contain positional data
        x_test = x_test[:, :4]
        
        if self.cluster_mean is None:
            print("Error: Must call the fit-method before the model can predict something!")
            raise SystemError

        idx_test = self.model.predict(x_test)
        return self.cluster_mean[idx_test]


class HistBinFactory(ModelFactory):
    """HistBinning Factory"""

    @staticmethod
    def search_space() -> Mapping[str, Any]:
        """The models search space.

        :param:
            None
        :return:
            Search space of the model.
        """

        return {
            "model_type": "HistBin",
            "num_bins": loguniform_int(10, 10000),
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

        return HistBin(num_bins=config["num_bins"])
