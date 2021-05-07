"""
    Module for removing the nan value rows out of a x set and y set.
"""

import pandas as pd
import numpy as np


def remove_nans(x_set: np.ndarray, y_set: np.ndarry):
    """
    Removes rows with nan values from both x and y

    Parameters:
        x_set: Feature set as numpy array
        y_set: target set as numpy array

    Returns:
        np_x: x as numpy array without nan rows
        np_x: y as numpy array without nan rows
    """

    # cast x, y to pd.DataFrame
    df_x = pd.DataFrame(x_set)
    df_y = pd.DataFrame(y_set, columns=["p"])

    # concat x and y, so we can remove the whole rows with nans later
    df_xy = pd.concat([df_x, df_y], axis=1)

    # remove rows with nan values in the channel column from the dataset
    df_xy = df_xy.dropna()

    # split x and y again
    df_x = df_xy[df_x.columns]
    df_y = df_xy[df_y.columns]

    # cast x, y to numpy arrays
    np_x = np.array(df_x)
    np_y = np.array(df_y)

    return np_x, np_y
