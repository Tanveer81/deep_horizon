"""
This is a module for testing rolling statistics. Do not use it in production for lack of efficiency.
"""
from datetime import datetime

import numpy as np
import pandas as pd


class HistoryData:
    """
    Calculate history data for test
    """
    def __init__(self, data, period, time=-1):
        """
        Calculate history data for test
        :param data:
        :param period:
        :param time:
        """
        self.delta = pd.to_timedelta(period).total_seconds()
        self.date_format = '%Y-%m-%d %H:%M:%S'
        size = data.shape[-1]
        self.window = np.array([]).reshape(0, size)
        func = np.frompyfunc(self.calculate_history, size + 1, size - 1)
        split = np.hsplit(data, size)
        self.history = np.array(func(time, *split)).reshape(-1, size - 1)

    def calculate_history(self, time, *args):
        """
        Calculate history data for test
        :param time:
        :param args:
        :return:
        """
        array = list(args)
        array[time] = (array[time] - datetime(1900, 1, 1)).total_seconds()
        array = np.array(array).astype(np.float)
        self.window = np.vstack((self.window, array))
        diff = array[time] - self.window[0, time]
        while diff > self.delta:
            np.delete(self.window, 0, axis=0)
            diff = array[time] - self.window[0, time]
        mean = np.average(self.window[:, :time], axis=0)
        if len(mean) == 1:
            return mean[0]
        return tuple(mean)

    def get_history(self):
        """
        Get history data
        :return: Dataframe of history data
        """
        return self.history
