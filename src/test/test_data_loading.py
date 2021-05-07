"""
Test Data Loading
"""
import unittest

import numpy as np

from data import FEATURES_NO_FOOT_TYPE
from data.data_init import init_data
from data.data_loading import DataPipeline
from data.data_rolling import moving_average
from data.rolling_utils import HistoryData

SAMPLE = 200


class TestDataLoading(unittest.TestCase):
    """
    Test Data Loading
    """
    @classmethod
    def setUpClass(cls):
        init_data()
        cls.data = DataPipeline(test=SAMPLE, version=0.23, load='train', trans_type='power',
                                trans_outlier=False, data_outlier=False)

    def test_data_pipeline(self):
        """
        Test the data pipeline.
        """
        mat_x_train, mat_y_train = self.data.load_df_by_channel(
            split='train',
            features=FEATURES_NO_FOOT_TYPE,
            label='p1',
            dropna=False)
        self.assertEqual(mat_x_train.shape, (SAMPLE, len(FEATURES_NO_FOOT_TYPE)))
        self.assertEqual(mat_y_train.shape, (SAMPLE, 1))

    def test_nan_dropping(self):
        """
        Test the NaN dropping.
        """
        data = DataPipeline(trans_type='none')
        # for i in [1, 4, 7]:
        #     x_train, y_train = data.load_df_by_channel(split='train', label=f'p{i}', dropna=True)
        #     self.assertEqual(x_train.shape[0], y_train.shape[0])
        #     for arr in [x_train, y_train]:
        #         self.assertFalse(arr.isnull().values.any())
        x_train_all, y_train_all = data.load_df(split='train', dropna=True)
        for arr in [x_train_all, y_train_all]:
            self.assertFalse(arr.isnull().values.any())
        x_test_all, _ = data.load_df(split='test', dropna=True)
        self.assertEqual(x_train_all.shape[0] + x_test_all.shape[0], 5698065)

    def test_history_data(self):
        """
        test history data
        """
        mat_x_train, _ = self.data.load_df('train')
        history = ['VxSW_GSE']
        rolling_mean = moving_average(mat_x_train, '1d', columns=history)
        array = np.array(mat_x_train.loc[:, history + ['DateTime']])
        history_data = HistoryData(array, '1d')
        com1 = np.round(history_data.get_history().astype(np.double), 1)
        com2 = np.array(rolling_mean).round(1)
        comparison = com1 == com2
        comp = np.hstack([com1, com2, comparison])
        print(comp)
        error = comp[comp[:, -1] == 0]
        if array.size > 0:
            print("Error in: ")
            print(error)
        print(comparison.all())
        self.assertTrue(comparison.all())
