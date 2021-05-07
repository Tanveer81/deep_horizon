"""
    Unit test for mixed loss
"""

import numpy
import torch

from utils.pytorch_utilities import MixedLoss


# pylint: disable=no-member
def test_loss():
    """Smoke-test for MixedLoss."""
    batch_size = 7
    num_channels = 5
    epsilon = numpy.random.uniform(size=(num_channels,))
    criterion = MixedLoss(epsilon=epsilon)
    y_true = torch.rand(batch_size, num_channels, requires_grad=False)
    y_pred = torch.rand(batch_size, num_channels, requires_grad=True)
    loss = criterion(y_true=y_true, y_pred=y_pred)
    loss.backward()
