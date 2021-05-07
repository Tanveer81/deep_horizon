"""
    This script defines some helper function for Pytorch Training

    @author: tanveer
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from data import FEATURES, LABELS
from data.data_init import init_data
from data.data_loading import DataPipeline
from models.custom_dataset import CustomDataset, TimeSeriesDataset
from utils import ml_flow_utils


def get_data_loaders(covariates: np.ndarray, labels: np.ndarray, batch: int,
                     config):
    """
    Get train and test loaders for x, y.
    :param x: Covariates
    :param y: Labels
    :param batch: batch_size
    :param dataset: Object of a class that inherits
        pyorch's dataset
    :return train_loader: torch.utils.data.DataLoader
    :return val_loader: torch.utils.data.DataLoader
    """
    x_train, x_val, y_train, y_val = train_test_split(covariates,
                                                      labels,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train = torch.from_numpy(x_train). \
        float().to(device)  # pylint: disable=maybe-no-member
    y_train = torch.from_numpy(y_train). \
        float().to(device)  # pylint: disable=maybe-no-member
    x_val = torch.from_numpy(x_val). \
        float().to(device)  # pylint: disable=maybe-no-member
    y_val = torch.from_numpy(y_val). \
        float().to(device)  # pylint: disable=maybe-no-member

    if config["time_series"]:
        train_dataset = TimeSeriesDataset(x_train, y_train, config["window"], config["pred_offset"])
        val_dataset = TimeSeriesDataset(x_val, y_val, config["window"], config["pred_offset"])
    else:
        train_dataset = CustomDataset(x_train, y_train)
        val_dataset = CustomDataset(x_val, y_val)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch,
                            shuffle=True)

    return train_loader, val_loader


def load_data(batch: int,
              test: bool, time_series: bool, window: int = 60, pred_offset: int = 60) -> \
        (torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict):
    """
    Loads data from DataPileLine and creates data loaders.
    :param pred_offset: offset of the label
    :param window: length of the sequence
    :param custom_dataset: type of dataset
    :param time_series: time-series data or not
    :param batch: batch size
    :param test: Test run ot not
    :return train_loader: Data-loader for training set.
    :return val_loader: Data-loader for validation set.
    :return test_data: dictionary with keys "x_test" and "y_test"
    """
    # Load Data from DVC
    init_data()
    data = DataPipeline(test=test)
    x_train, y_train = data.load_data('train',
                                      features=FEATURES,
                                      labels=LABELS,
                                      dropna=True)
    x_test, y_test = data.load_data('test',
                                    features=FEATURES,
                                    labels=LABELS,
                                    dropna=True)
    dataset_config = dict(time_series=time_series, window=window,
                          pred_offset=pred_offset)
    train_loader, val_loader = get_data_loaders(x_train, y_train,
                                                batch, dataset_config)
    test_data = {"x_test": x_test, "y_test": y_test}

    return train_loader, val_loader, test_data


def visualize_loss(path: str, channel: any):
    """
    Makes plot for training and validation loss.
    :param path: path to save the plot
    :param channel: output channel
    """
    loss_file_path = path + f"/loss_p{channel}.txt"
    csv = pd.read_csv(loss_file_path)
    fig = plt.figure()
    plt.plot(csv['training_loss'])
    plt.plot(csv['validation_loss'])
    plt.title('Model loss for Channel_p' + str(channel))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    #    plt.show()
    fig.savefig(path + '/loss_channel_p' + str(channel) + '.png', dpi=fig.dpi)


def visualize_loss_v2(path: str, val_losses: list, channel: int):
    """
    Makes plot for validation loss.
    :param path:
    :param val_losses:
    :param channel:
    :return:
    """
    fig = plt.figure()
    plt.plot(val_losses)
    plt.title('Model loss for Channel_p' + str(channel))
    plt.ylabel('Loss')
    plt.xlabel('10Xbatch')
    plt.legend(['Val'], loc='upper left')
    #    plt.show()
    fig.savefig(path + '/loss_p' + str(channel) + 'every_tenth_batch.png',
                dpi=fig.dpi)


def save_to_mlflow(files: list, experiment_name: str, run: str,
                   variable_dict: dict):
    """
    Saves files and parameters to mlflow.
    :param files: list of all file paths to save
    :param experiment_name: name of the mlflow experiment
    :param run: name of the run
    :param variable_dict: dictionary of parameters to save
    """
    MlflowClient()

    tracking_uri = r'INSERT MLFLOW TRACKING URI'
    artifact_uri = None  # Replace None with artifact url

    mlflow.set_tracking_uri(tracking_uri)

    exp_id = ml_flow_utils.create_experiment(mlflow, MlflowClient(),
                                             experiment_name, artifact_uri)

    mlflow.start_run(experiment_id=exp_id, run_name=run)

    #
    for key, value in variable_dict.items():
        mlflow.log_param(key, value)

    for file in files:
        mlflow.log_artifact(file)

    mlflow.end_run()


def test_mlflow():
    """
    Finction to test mlflow connection.
    """
    files = ['output.txt']

    thisdict = {"brand": "Ford", "model": "Mustang", "year": 1964}

    save_to_mlflow(files, 'check mlflow tanveer', 'run1', thisdict)


# pylint: disable=no-member
def mixed_loss(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               epsilon: torch.Tensor,
               ) -> torch.Tensor:
    """
    Evaluate the loss function

        :param y_true: shape: (batch_size, num_channels)
            The true values.
        :param y_pred: shape: (batch_size, num_channels)
            The predicted values.
        :param epsilon: shape: (num_channels)
            The threshold.

        :return: scalar.
            The loss value.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
        epsilon = epsilon.to(device)

    if epsilon.ndimension() < 2:
        epsilon = epsilon.unsqueeze(dim=0)
    large_mask = y_true > epsilon
    large_loss = (y_pred - y_true).square()
    small_loss = (y_pred - epsilon).relu().square()
    return (large_loss[large_mask].sum() + small_loss[~large_mask].sum()) / y_true.numel()


# pylint: disable=no-member
# pylint: disable=arguments-differ
class MixedLoss(nn.Module):
    """Wrapper around the mixed_loss."""

    def __init__(
            self,
            epsilon: np.ndarray,
    ):
        """
        Initialize the module.

        :param epsilon:
            The thresholds per channel.
        """
        super().__init__()
        # register as buffer: ensures that the tensor gets moved to the device
        self.register_buffer(name='epsilon',
                             tensor=torch.as_tensor(data=epsilon, dtype=torch.float32).view(1, -1))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the loss function.

        :param y_true: shape: (batch_size, num_channels)
            The true values.
        :param y_pred: shape: (batch_size, num_channels)
            The predicted values.

        :return:
            The scalar loss value.
        """
        return mixed_loss(y_true=y_true, y_pred=y_pred, epsilon=self.epsilon)
