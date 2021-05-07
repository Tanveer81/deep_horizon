"""
Data Loading Pipeline
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch

from data import SEGMENTS, COLUMNS, SEGMENTS_ALLOWED, FEATURES, LABELS, VERSIONS, TRANS, SPLIT, \
    SMALL_DATA, FEATURES_NO_FOOT_TYPE, FEATURES_NO_FOOT_TYPE_WITH_TIME
from data.data_rolling import append_rolling
from sklearn.model_selection import train_test_split

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


class DataPipeline:
    """
    Data Loading Pipeline
    """

    def __init__(self, test: bool or int = False, version: float = 0.23, load: str = 'train+test',
                 trans_type: str = 'robust', trans_outlier: bool = True, data_outlier: bool = True):
        """
        Establish a data loading pipeline.
        :param load: to specify a split from {'train', 'test'}
            (multi-selection supported but not recommended)
        :param version: data version (0.23 by default)
        :param trans_type: type of transformer. 'raw' for raw data, 'none' for raw data with
            proton intensities in log space, others to load transformer file ending with
            "_[trans_type]trans(cut).pkl", e.g., "RAPID_OMNI_ML_023_robusttrans.pkl"
        :param trans_outlier: if outliers below the cut-off are included when the transformer was
            fit. If it's set to be true, then the data and transformer files with names ending with
            "cut.[PREFIX]" will be loaded, e.g., "RAPID_OMNI_ML_023_traincut.h5"
        :param data_outlier: if outliers below the cut-off are included in the data to be loaded
        :param test: set to True to get a small sample with default size 3000, set to an integer to
            customize the sample size
        """
        if load == 'all':
            picked = SPLIT
        else:
            picked = load.split('+')
        self.location = '../../data/'
        self.load = load
        self.version = version
        self.trans_type = trans_type
        self.trans_outlier = trans_outlier
        self.labels = []
        self.features = []

        self.data = {}
        if 'train' in picked and trans_outlier != data_outlier:
            raise SystemError('Wrong setting for outliers.')
        data_suffix = '' if data_outlier else 'cut'
        for pick in picked:
            self.data[pick] = pd.DataFrame(pd.read_hdf(self.get_path(self.location,
                                                                     f'{pick}{data_suffix}', 'h5')))
            if test is True:
                self.data[pick] = self.data[pick][: SMALL_DATA]
            elif test:
                self.data[pick] = self.data[pick][: test]

        if trans_type == 'raw':
            self.trans = None
            for pick in picked:
                self.data[pick].loc[:, COLUMNS['CHANNEL']] = \
                    np.power(10, self.data[pick][COLUMNS['CHANNEL']])
        elif trans_type == 'none':
            self.trans = None
        elif trans_type in TRANS:
            trans_suffix = '' if trans_outlier else 'cut'
            self.trans = joblib.load(self.get_path(self.location,
                                                   f'{trans_type}trans{trans_suffix}', 'pkl'))
            for pick in picked:
                self.data[pick] = self._transform(self.data[pick])
        else:
            raise FileNotFoundError('Wrong transformer')

    def _transform(self, data: pd.DataFrame):
        """
        Transform the data.
        :param data: Dataset
        :return: transformed dataset
        """
        selection = COLUMNS['CHANNEL'] + COLUMNS['POS'] + COLUMNS['OMNI']
        transformed = data[selection]
        transformed = self.trans.transform(transformed)
        data.loc[:, selection] = transformed
        return data

    def invert_trans(self, data):
        """
        Invert some transformation of labels in the preprocessing
        :param data: Preprocessed dataset
        :return: Inverse of preprocessing
        """
        all_columns = COLUMNS['CHANNEL'] + COLUMNS['POS'] + COLUMNS['OMNI']
        to_inverse = pd.DataFrame(np.zeros((data.shape[0], len(all_columns))),
                                  columns=all_columns, index=data.index)
        to_inverse.loc[:, data.columns] = data
        output = pd.DataFrame(self.trans.inverse_transform(to_inverse),
                              columns=all_columns, index=data.index)[data.columns]
        return output

    def transform(self, data):
        """
        
        :param data: 
        :return: 
        """
        all_columns = COLUMNS['CHANNEL'] + COLUMNS['POS'] + COLUMNS['OMNI']
        to_inverse = pd.DataFrame(np.zeros((data.shape[0], len(all_columns))),
                                  columns=all_columns, index=data.index)
        to_inverse.loc[:, data.columns] = data
        output = pd.DataFrame(self.trans.transform(to_inverse),
                              columns=all_columns, index=data.index)[data.columns]
        return output

    def invert_x_trans(self, data: pd.DataFrame):
        """
        Invert transformation to features
        :param data: Dataset
        :return: The inverse
        """
        return self.invert_trans(data)

    def invert_y_trans(self, data: pd.DataFrame, log: bool = False):
        """
        Invert transformation to labels
        :param log: If the data is in log-space
        :param data: Dataset
        :return: The inverse
        """
        data_copy = data.copy()
        assert not data.isnull().values.any(), \
            'Data includes NaNs. Please remove them before doing the inverse.'
        if self.trans_type in {'power'}:
            suffix = '' if self.trans_outlier else 'cut'
            maxima_path = self.get_path(self.location, f'{self.trans_type}{suffix}maxima', 'csv')
            maxima = pd.read_csv(maxima_path, squeeze=True, index_col=0)
            upper = maxima[data.columns].tolist()
            data_copy = data_copy.clip(upper=upper)
        output = self.invert_trans(data_copy)
        if not log:
            output = np.power(10, output)
        return output

    def get_path(self, directory, split, suffix, version=True):
        """
        Get the data from storage and return the path to data.
        :param suffix: Format
        :param version: Specify the version of dataset
        :param split: The required split
        :param directory: Relative path to the data directory
        :param split: [Deprecated]
        :return: The absolute path to the data
        """
        path = WORKING_DIR + '/' + directory
        f_list = [
            f for f in os.listdir(path) if f.split('_')[-1] == split + '.' +
            suffix and (not version or f.split('_')[-2] == VERSIONS[self.version])
        ]
        if len(f_list) == 0:
            print("Error: " + split + " data not found!")
            raise FileNotFoundError
        if len(f_list) > 1:
            print("Error: Multiple " + split + " data found!")
            raise SystemError
        print('Valid ' + split + ' data found. Loading...')
        return path + f_list[0]

    @staticmethod
    def validate_columns(columns, segment):
        """
        Validate a column selection.
        :param columns: Column selection to be validated.
        :param segment: The target segment of columns.
        :return: True/False
        """
        return set(columns).issubset(SEGMENTS_ALLOWED[segment])

    def slice(self, data: pd.DataFrame, dropna: bool = False,
              **kwargs) -> (pd.DataFrame, pd.DataFrame):
        """
        The function returns dataset for specific channel.
        :param data: Data set to be sliced
        :param dropna: Remove rows with NaN or not
        :returns: Pandas Dataframe: A dataframe without nans and having only the specific channel.
        """
        selections = {}
        for key, value in kwargs.items():
            key = key.upper()
            default = [col for col in SEGMENTS[key] if col in data.columns]
            selection = value if value else default
            assert self.validate_columns(selection, key), \
                str(selection) + " are invalid selection of " + key + ". "
            selections[key] = selection
        data = data[[col for seg in selections.values() for col in seg]].copy()
        if dropna:
            data = data.dropna()
        sliced = {key: data[value] for key, value in selections.items()}
        return sliced

    def load_df(self,
                split: str,
                labels: list = None,
                features: list = None,
                dropna: bool = False) -> (pd.DataFrame, pd.DataFrame):
        """
        Load the data set as a data frame .
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param labels: Specify columns for the output data in a list with length k,
            e.g., ['p1', 'p2']
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Tuple (X, Y) where
            X: The preprocessed input data in shape of (n, d)
            Y: The preprocessed output data in shape of (n, k)
        """
        dataset = self.data.get(split, None)
        if dataset is None:
            raise FileNotFoundError('Dataset not found!')
        selections = self.slice(data=dataset, dropna=dropna, labels=labels, features=features)
        return selections['FEATURES'], selections['LABELS']

    def load_df_by_channel(
            self,
            split: str,
            label: str,
            features: list = None,
            dropna: bool = False) -> (pd.DataFrame, pd.DataFrame):
        """
        Load the data set as a data frame.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param label: Specify columns for the output data, e.g., 'p1'
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Tuple (X, y) where
            X: The preprocessed input data in shape of (n, d)
            y: The preprocessed output data in shape of (n, y)
        """
        return self.load_df(split=split, labels=[label], features=features, dropna=dropna)

    def load_data(self,
                  split: str,
                  labels: list = None,
                  features: list = None,
                  dropna: bool = False) -> (np.ndarray, np.ndarray):
        """
        Load the data set.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param labels: Specify columns for the output data in a list with length k,
            e.g., ['p1', 'p2']
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Tuple (X, Y) where
            X: The preprocessed input data in shape of (n, d)
            Y: The preprocessed output data in shape of (n, k)
        """
        mat_x, mat_y = self.load_df(split=split, labels=labels, features=features, dropna=dropna)
        return np.array(mat_x), np.array(mat_y)

    def load_data_by_channel(self,
                             split: str,
                             label: str,
                             features: list = None,
                             dropna: bool = False) -> (np.ndarray, np.ndarray):
        """
        Load the data set.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param label: Specify columns for the output data, e.g., 'p1'
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Tuple (X, y) where
            X: The preprocessed input data in shape of (n, d)
            y: The preprocessed output data in shape of (n, y)
        """
        return self.load_data(split=split, labels=[label], features=features, dropna=dropna)

    def load_df_with_t(
            self,
            split: str,
            labels: list = None,
            features: list = None,
            dropna: bool = False,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Load the data set as tensors with timestamp array.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param labels: Specify columns for the output data in a list with length k,
            e.g., ['p1', 'p2']
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Triple (t, X, Y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            Y: The preprocessed output data in shape of (n, k)
        """
        dataset = self.data.get(split, None)
        if dataset is None:
            print('Dataset not found!')
        default_feature = [col for col in FEATURES if col in dataset.columns]
        columns = self.slice(
            split=split,
            data=dataset,
            dropna=dropna,
            datetime=COLUMNS['DATETIME'],
            labels=labels,
            features=features if features else default_feature)
        for key in columns:
            value = columns[key].astype(int) / 10 ** 9 if key == 'DATETIME' else columns[key]
            columns[key] = value
        return columns['DATETIME'], columns['FEATURES'], columns['LABELS']

    def load_data_with_t(
            self,
            split: str,
            labels: list = None,
            features: list = None,
            dropna: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Load the data set as tensors with timestamp array.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param labels: Specify columns for the output data in a list with length k,
            e.g., ['p1', 'p2']
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Triple (t, X, Y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            Y: The preprocessed output data in shape of (n, k)
        """
        vec_t, mat_x, mat_y = self.load_df_with_t(split=split, labels=labels, features=features,
                                                  dropna=dropna)
        return np.array(vec_t), np.array(mat_x), np.array(mat_y)

    def load_tensor_with_t(
            self,
            split: str,
            labels: list = None,
            features: list = None,
            dropna: bool = False
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Load the data set as tensors with timestamp array.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param labels: Specify columns for the output data in a list with length k,
            e.g., ['p1', 'p2']
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Triple (t, X, Y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            Y: The preprocessed output data in shape of (n, k)
        """
        tensors = []
        for tensor in self.load_data_with_t(split=split, labels=labels, features=features,
                                            dropna=dropna):
            tensors.append(torch.from_numpy(tensor))  # pylint: disable=no-member
        return tensors[0], tensors[1], tensors[2]

    def load_df_with_t_by_channel(
            self,
            split: str,
            label: str,
            features: list = None,
            dropna: bool = False
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Load the data set as tensors with timestamp array.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param label: Specify column for the output data, e.g., 'p1'
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Triple (t, X, y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            y: The preprocessed output data in shape of (n, 1)
        """
        return self.load_df_with_t(split=split, labels=[label], features=features, dropna=dropna)

    def load_data_with_t_by_channel(
            self,
            split: str,
            label: str,
            features: list = None,
            dropna: bool = False
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Load the data set as tensors with timestamp array.

        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param label: Specify column for the output data, e.g., 'p1'
        :param split: Specify a dataset from {'train', 'test'}.
        :return: Triple (t, X, y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            y: The preprocessed output data in shape of (n, 1)
        """
        return self.load_data_with_t(split=split, labels=[label], features=features, dropna=dropna)

    def load_tensor_with_t_by_channel(
            self,
            split: str,
            label: str,
            features: list = None,
            dropna: bool = False
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Load the data set as tensors with timestamp array.
        :param split: Specify a dataset from {'train', 'test'}.
        :param dropna: Remove rows with NaN or not
        :param features: Specify columns for the input data in a list with length d,
            e.g., ['NpSW', 'VxSW_GSE']
        :param label: Specify column for the output data, e.g., 'p1'
        :return: Triple (t, X, y) where
            t: Timestamp in shape of (n, 1)
            X: The preprocessed input data in shape of (n, d)
            y: The preprocessed output data in shape of (n, 1)
        """
        return self.load_tensor_with_t(split=split, labels=[label], features=features,
                                       dropna=dropna)

    def get_minima(self, channels: list = None) -> pd.DataFrame:
        """
        Get the minimal value for selected channel.
        :param channels: Selection of channels. Default: all.
        :return: The minimal values.
        """
        minima_path = WORKING_DIR + f'/../../data/RAPID_OMNI_ML_{VERSIONS[self.version]}_minima.csv'
        intensity_minima = pd.read_csv(minima_path, index_col=0, squeeze=True)
        selection = channels if channels else LABELS
        return intensity_minima[selection]
    
        
    def create_dataset(self, channel: str, features: list = FEATURES_NO_FOOT_TYPE,
                   features_with_time: list = FEATURES_NO_FOOT_TYPE_WITH_TIME, val: bool = False) -> dict:
        """
        SUMMARY
            Creates x/y datasets including the history input features

        PARAMETERS
            channel --- str: Channel
            features --- list[string]: List of features

        RETURNS
            dict
                Dictionaty containing x, y sets
        """
        # Load train data from pipeline
        x_train, y_train = self.load_df(split='train', features=features_with_time,
                                        labels=[channel], dropna=True)
        time_train = self.load_df(split='train', features=features_with_time,
                                        labels=[channel], dropna=True)

        # Split train data in train and validation data
        x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2,
                                                    random_state=42, shuffle=False)
        #time_train, time_test = train_test_split(time_train[0][['DateTime']], test_size=0.2,
        #                                            random_state=42, shuffle=False)
        
        # Load test data from datapieline
        x_test, y_test = self.load_df(split='test', features=features_with_time,
                                          labels=[channel], dropna=True)
        time_test = self.load_df(split='test', features=features_with_time,
                                      labels=[channel], dropna=True)
        time_train = time_train[0][['DateTime']]
        time_test = time_test[0][['DateTime']]


        # Add the new features to the training set, drop the time for the last one.
        # Time is needed for the pipeline to calculate the average.
        list_avgs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for avg in list_avgs:
            if avg == list_avgs[-1]:
                x_train = append_rolling(x_train, str(avg) + 'h', columns=features, droptime=True)
                x_val = append_rolling(x_val, str(avg) + 'h', columns=features, droptime=True)
                x_test = append_rolling(x_test, str(avg) + 'h', columns=features, droptime=True)
                #time_train = append_rolling(time_train, str(avg) + 'h', columns=features, droptime=True)
               # time_test = append_rolling(time_test, str(avg) + 'h', columns=features, droptime=True)
            else:
                x_train = append_rolling(x_train, str(avg) + 'h', columns=features, droptime=False)
                x_val = append_rolling(x_val, str(avg) + 'h', columns=features, droptime=False)
                x_test = append_rolling(x_test, str(avg) + 'h', columns=features, droptime=False)
                #time_train = append_rolling(time_train, str(avg) + 'h', columns=features, droptime=False)
               # time_test = append_rolling(time_test, str(avg) + 'h', columns=features, droptime=False)

        features = x_train.columns.values


        # TODO
        return (x_train.to_numpy(), y_train.to_numpy()[:, :], x_val.to_numpy()[:], y_val.to_numpy()[:, 0],\
                x_test.to_numpy(), y_test.to_numpy()[:, 0], time_train, time_test, features)


if __name__ == '__main__':
    pipe_train = DataPipeline(version=0.23, load='test',
                              trans_type='robust', trans_outlier=False, data_outlier=False)
    x_train, y_train = pipe_train.load_df(split='test', labels=['p1'], dropna=True)
    print(x_train)
    print(y_train)
