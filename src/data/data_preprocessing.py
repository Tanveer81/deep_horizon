"""
    This script defines a class and some helper functions for PreProcessing
    @author: tanveer, zhou
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, RobustScaler, \
    StandardScaler

from data import COLUMNS
from data.data_splitting import split_ts

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


class PreProcessing:
    """
    This is a class for pre-processing data.
    """
    def __init__(self, minima: list, version: str, path: str):
        """
        The constructor for pre-processing class.
        """
        self.version = version
        self.path = path
        self.data_path = f'{WORKING_DIR}/{self.path}/RAPID_OMNI_ML_{self.version}'
        self.minima = minima
        minima_df = pd.DataFrame(minima, index=COLUMNS['CHANNEL'])
        minima_df.to_csv(f'{self.data_path}_minima.csv')

        quan_trans = QuantileTransformer(copy=True)
        power_trans = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
        robust_trans = RobustScaler(copy=True)
        normal_trans = StandardScaler(copy=True)
        self.trans = {'quantile': quan_trans, 'power': power_trans,
                      'robust': robust_trans, 'normal': normal_trans}

    def preprocess(self, data: pd.DataFrame, path: str,
                   drop_min: bool = False) -> [pd.DataFrame, pd.DataFrame]:
        """
        Preprocessing
        :param path:
        :param drop_min:
        :param data: The data set
        :return: The processed dataset
        """
        data.reset_index(drop=True, inplace=True)
        data_copy = data.copy()
        for channel, min_value in enumerate(self.minima, start=1):
            column = f'p{channel}'
            if drop_min:
                data_copy.loc[data_copy[column] < min_value, column] = np.nan
            else:
                data_copy.loc[:, column] = data_copy[column].clip(lower=min_value)
            data_copy.loc[:, column] = np.log10(data_copy[column])
        data_copy.to_hdf(path, key='df')
        return data_copy

    def normalize(self, trans: str, data: pd.DataFrame,
                  suffix: str) -> [pd.DataFrame, pd.DataFrame]:
        """
        Normalize a dataset.
        :param suffix:
        :param trans: transformer
        :param data: dataset
        :return: normalized data
        """
        selection = COLUMNS['CHANNEL'] + COLUMNS['POS'] + COLUMNS['OMNI']
        normalized = data[selection]
        self.trans[trans].fit(normalized)
        joblib.dump(self.trans[trans], f'{self.data_path}_{trans}trans{suffix}.pkl')

    @staticmethod
    def filter_data(data: pd.DataFrame):
        """
        Apply filter to the data.
        :param data: dataset
        :return: filtered data
        """
        filter_rdist = data[data['rdist'] > 6]
        return filter_rdist

    def fit(self, train, test, drop_min: bool):
        """
        Prepare the transformer
        """
        suffix = 'cut' if drop_min else ''
        filtered_train = self.filter_data(data=train)
        train_preprocessed = self.preprocess(data=filtered_train, drop_min=drop_min,
                                             path=f'{self.data_path}_train{suffix}.h5')
        self.normalize(trans='quantile', data=train_preprocessed, suffix=suffix)
        self.normalize(trans='power', data=train_preprocessed, suffix=suffix)
        y_train = self.trans['power'].transform(train_preprocessed)[COLUMNS['CHANNEL']]
        y_train.max().to_csv(f'{self.data_path}_power{suffix}maxima.csv')
        self.normalize(trans='robust', data=train_preprocessed, suffix=suffix)
        self.normalize(trans='normal', data=train_preprocessed, suffix=suffix)
        filtered_test = self.filter_data(data=test)
        self.preprocess(data=filtered_test, drop_min=drop_min,
                        path=f'{self.data_path}_test{suffix}.h5')

    def generate(self):
        """
        generate raw data and transformers
        :return:
        """
        unprocessed = pd.DataFrame(pd.read_hdf(self.data_path + '_raw.h5'))
        train_set, test_set = split_ts(unprocessed, 0.8)
        self.fit(train=train_set, test=test_set, drop_min=False)
        self.fit(train=train_set, test=test_set, drop_min=True)


if __name__ == '__main__':
    min_values = [5, 1, 0.5, 0.1, 0.05, 0.005, 0.001]
    proc = PreProcessing(minima=min_values, version='023', path='../../data/')
    proc.generate()
