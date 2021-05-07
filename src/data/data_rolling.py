"""
Provide some functions to get rolling statistics.
"""

import numpy as np
import pandas as pd

from data import COLUMNS


def moving_average(data, window, columns=None, method='mean'):
    """
    Get rolling statistics for selected columns
    :param method: Aggregation method: 'mean', 'fourier' or 'haar'
    :param data: Data set to get rolling statistics from
    :param window: Period in a rolling windows, 'min': minute, 'h': hour, 'd': day
    :param columns: Column selection
    :return: Rolling statistics
    """
    if not columns:
        columns = [columns for columns in data.columns if columns in COLUMNS['OMNI']]
    window = data.set_index(COLUMNS['DATETIME']).loc[:, columns].rolling(window)
    print(window)
    rolling_stat = None
    if method == 'mean':
        rolling_stat = window.mean()
    elif method == 'count':
        rolling_stat = window.count()
    return rolling_stat


def append_rolling(data, window, columns=None, method='mean', droptime=False):
    """
    Append the rolling statistics to the original dataset.
    :param method: Aggregation method: 'mean', 'fourier' or 'haar'
    :param data: Data set to get rolling statistics from
    :param window: Period in a rolling windows, 'min': minute, 'h': hour, 'd': day
    :param columns: Column selection
    :param droptime: If the DateTime column will be removed
    :return: The dataset with rolling statistics appended
    """
    rolling_stat = moving_average(data, window, columns,
                                  method).reset_index(drop=True)
    rolling_stat = rolling_stat.add_suffix('_' + method + '_' + window)
    appended = pd.concat((data.reset_index(), rolling_stat), axis=1).set_index('index')
    if droptime:
        appended = appended.drop(columns=COLUMNS['DATETIME'])
    return appended


def replace_with_rolling(data, window, columns=None, method='mean', droptime=False):
    """
    Replace selected columns with their rolling statistics.
    Append the rolling statistics to the original dataset.
    :param method: Aggregation method: 'mean', 'fourier' or 'haar'
    :param data: Data set to get rolling statistics from
    :param window: Period in a rolling windows, 'min': minute, 'h': hour, 'd': day
    :param columns: Column selection
    :param droptime: If the DateTime column will be removed
    :return: The dataset where selected columns replced with rolling statistics
    """
    rolling_stat = moving_average(data, window, columns, method)
    replaced = data.copy()
    replaced.loc[:, rolling_stat.columns] = np.array(rolling_stat)
    if droptime:
        replaced = replaced.drop(columns=COLUMNS['DATETIME'])
    return replaced
