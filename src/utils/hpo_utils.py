""" Helper/Utility functions for HPO.

     @author: tanveer
"""
import numpy as np
from ray import tune


def loguniform(min_bound: int, max_bound: int):
    """
    Returns sample from a log-uniform distribution.

    :param min_bound: minimum bound of expected sample
    :param max_bound: maximum bound of expected sample

    :return: One sample from the distribution
    """

    def apply_log(_):
        return int(np.exp(np.random.uniform(
            np.log(min_bound), np.log(max_bound)))) // 32 * 32

    return tune.sample_from(apply_log)


def uniform(min_bound: int, max_bound: int):
    """
    Returns sample from a uniform distribution.

    :param min_bound: minimum bound of expected sample
    :param max_bound: maximum bound of expected sample

    :return: One sample from the distribution
    """

    def apply_uniform(_):
        """ Inner function for appling random.uniform()
        """
        return int(np.random.uniform(min_bound, max_bound))

    return tune.sample_from(apply_uniform)
