"""
Interpolate the data to fill the gird.
"""

import numpy as np
from scipy import interpolate
import pandas as pd

from data import COLUMNS
from data.data_loading import DataPipeline


def interp(data: pd.DataFrame):
    """
    Interpolate a dataset in form of pandas data frame.
    :param data: Dataset
    :return: Interpolated data.
    """
    time = data['DateTime'].astype(int)
    unit = 6 * 10 ** 10
    time_new = np.arange(time.iloc[0], time.iloc[-1] + unit, unit)
    x_new = pd.DataFrame(np.empty((len(time_new), data.shape[-1])), columns=data.columns)
    omni = data[COLUMNS['OMNI']]
    for name, column in omni.iteritems():
        func = interpolate.interp1d(time, column, kind='linear')
        data_new = func(time_new)
        x_new.loc[:, name] = data_new
    x_new.loc[:, 'DateTime'] = pd.to_datetime(time_new)
    return x_new


if __name__ == '__main__':
    pipe = DataPipeline(test=False)
    mat_x, _ = pipe.load_df('train', features=COLUMNS['OMNI'] + COLUMNS['DATETIME'])
    mat_x_new = interp(mat_x)
    mat_x_new.to_hdf('../../data/RAPID_OMNI_ML_new_trainOMNIInter.h5', key='df')
