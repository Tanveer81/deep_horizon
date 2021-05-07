"""
    This script contains the functionality to split the data. Especially for
    the time series data it's important

    @author: jhuthmacher
"""

import pandas as pd


def split_ts(data: pd.DataFrame, train_ratio: float = 0.8):
    """ This method split the data depending on a specified train ratio

        Parameter
        ---------
        df: pd.DataFrame
            DataFrame containing time series data which should be splitted.
        train_ratio: float
            Size of the trainings set in proportion to the complete data set
            (0.8 proposed from expert side).

        Return
        ------
        train_set: pd.DataFrame
            Train set
        test_set: pd.DataFrame
            Test set
    """

    threshold = len(data) * train_ratio

    train_set = data[:int(threshold)]
    test_set = data[int(threshold):]

    return train_set, test_set


def get_num_cycles(data: pd.DataFrame):
    """ This method returns the number of cycles in the data set.

        In our context have data of the position of a satellite which
        circles the earth on its orbit. When there is enought data we
        can check how often the satellite finished its orbit.

        Parameter
        ---------
        data: pd.DataFrame
            DataFrame containing x, y and z coordinates about the satellites
            position.

        Return
        ------
        num_circles: int
            Number of circles the satellite did on its orbit.
    """
    raise NotImplementedError(data)
