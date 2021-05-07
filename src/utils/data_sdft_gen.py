"""
Tool for generate SDFT additional features
"""
import numpy as np
import pandas as pd
from scipy import signal

from data import COLUMNS, FEATURES_WITH_TIME
from data.data_loading import DataPipeline


def sdft(time_array_old, x_matrix_new, window_size, column):
    """
    Fourier transform
    :param time_array_old: Original timestamps
    :param x_matrix_new: New features with time stamps
    :param window_size: Window size
    :param column: Columns  to be transformed
    :return: Transform
    """
    x_for_stft = np.concatenate([
        np.zeros((window_size - 1,)),
        np.array(x_matrix_new.loc[:, column]).reshape((-1))
    ])
    _, _, sxx = signal.spectrogram(x_for_stft,
                                   nperseg=window_size,
                                   noverlap=window_size - 1,
                                   window='boxcar',
                                   return_onesided=True)
    recover = sxx.T[time_array_old]
    return recover


if __name__ == '__main__':
    pipe = DataPipeline(test=False)
    for hour in [8, 16]:
        for split in ['test', 'train']:
            x_train, y_train = pipe.load_df(split, features=FEATURES_WITH_TIME)
            time_old = x_train['DateTime'].astype(int) / 10 ** 9 / 60
            time_old = (time_old - time_old[0]).astype(int)
            x_new, _ = pipe.load_df(split + '_omni_inter', labels=[])
            print(hour)
            PERIOD = 60 * hour
            NUM_FT = 5
            fts = []
            for col in COLUMNS['OMNI']:
                ft = pd.DataFrame(sdft(time_old, x_new, PERIOD, col)[:, :NUM_FT],
                                  columns=['_'.join([col, str(PERIOD), str(i)])
                                           for i in range(1, NUM_FT + 1)])
                fts.append(ft)
            merged = pd.concat(fts, axis=1)
            merged2 = pd.concat((y_train, x_train, merged), axis=1)
            print('saving')
            merged2.to_hdf('RAPID_OMNI_ML_new_' + split + '_ft_' + str(hour) +
                           'h.d5',
                           key='df')
