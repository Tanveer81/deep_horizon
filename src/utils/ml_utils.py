"""
    This script defines some helper function for Pytorch Training

    @author: jhuthmacher
    @based-on: hannan
"""

import numpy as np
from numpy.random import randint


def sample_array(lower_bound: int, upper_bound: int, length: int,
                 ascending: bool = True):
    """ Sample function for sampling arrays in tune.sample_from()
    """
    # pylint: disable=no-else-return
    if ascending:
        return np.sort(32 * randint(lower_bound, upper_bound, size=(length,)))
    else:
        return np.sort(32 * randint(lower_bound, upper_bound, size=(length,)))[::-1]  # noqa: E501
