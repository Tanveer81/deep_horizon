"""
Provides functionality to fastly include big arrays for choice into HPO with ray.tune.
"""

import numpy as np
from ray import tune


def loguniform_int(min_bound: int, max_bound: int) -> tune.sample_from:   # noqa
    """Sugar for sampling in different orders of magnitude.

    Args:
        min_bound (float): Lower boundary of the output interval (1e-4)
        max_bound (float): Upper boundary of the output interval (1e-2)
        base (float): Base of the log. Defaults to 10.
    Return:
        int
    """
    base = 10
    logmin = np.log(min_bound) / np.log(base)
    logmax = np.log(max_bound) / np.log(base)

    def apply_log(_):
        return int(base**(np.random.uniform(logmin, logmax)))

    return tune.sample_from(apply_log)
