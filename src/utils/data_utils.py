""" Utility script for data related operations.
"""
from datetime import datetime

import pandas as pd
import numpy as np

from data.data_loading import DataPipeline


def load_data():
    """ Wrapper function to load the data.

        Return:
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: 
                Raw data with protons in log10 space, raw data, test portion of raw data, train portion of raw data
    """

    data = DataPipeline(version=0.23, load='train+test', trans_type='none', trans_outlier=False, data_outlier=False)
    x_test, y_test = data.load_df(split='train')
    x_train, y_train = data.load_df(split='test')

    test_data = pd.concat([x_test, y_test], axis=1)
    train_data = pd.concat([x_train, y_train], axis=1)
    data = pd.concat([test_data, train_data])

    data = data[data["rdist"] >= 6]
    data["-rdist"] = data["rdist"] * -1 

    channels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    data[channels] = 10**data[channels]
    dataLog10 = data.copy()
    dataLog10[channels] = np.log10(dataLog10[channels])

    return dataLog10, data, test_data, train_data


def load_feature_importances(path: str, channels: [int] = [1, 2, 3, 4, 5],
                             one_file: bool = False, mode: str = "test"):
    """ Wrapper function to load the feature importance data frame.

        Parameter:
            path: str
                Location of the feature importance files
            channels: [int]
                List of the channels for which the feture importance is loaded.
            one_file: bool
                Determines if only one file with all channels is returned.
            mode: str
                It is just used to load the test or train feature importance.
                It is used as prefix in the file name.
        Return:
            pd.DataFrame: Data frame containing the feature importances.
    """
    if one_file:
        df = pd.read_csv(path, index_col=None)
        df[df.columns] = df[df.columns].astype(float)
        if "channel" in df.columns:
            df["channel"] = df["channel"].astype(int)
            df = df.set_index("channel")
        return df
    else:
        df_feature_imp = pd.DataFrame([])

        for channel in channels:
            df_feature_imp = df_feature_imp.append([pd.read_csv(f"{path}fi_ch{channel}_{mode}.csv",
                                                   index_col="channel")])

        # df_feature_imp.index = df_feature_imp["Channel"]

        return df_feature_imp


def transform_feature_importance(df: pd.DataFrame):
    """ Function to tranform the feature importance.

        Parameter:
            df: pd.DataFrame
                Containing the data frame with the feature importances
        Return:
            pd.DataFrame: Transformed data frame.
    """
    data = []
    for channel, g in df.groupby(df.index):
        for column in g.columns:
            if column.endswith("h"):
                feature, time = column.rsplit("_", maxsplit=1)
                time = int(time.replace("h", ""))
                data.append((channel, feature, time, g[column].values[0]))

    df = pd.DataFrame(data=data, columns=["channel", "feature", "time", "importance"])

    return df


def load_model_data(path: str, channels: [int] = [1, 2, 3, 4, 5]):
    """ Wrapper function to load the model data.

        Paramters:
            path: str
                Location of the model data.
            channels: [int]
                List of the specified channels for which the data should be loaded.
        Return:
            [pd.DataFrame]: List of data frames where each entry corresponds to the 
                            specific data frame for the channel.
    """
    dfs = []

    pipe_robust = DataPipeline(test=False, version=0.23, load='train', trans_type="robust",
                               trans_outlier=False, data_outlier=False)

    for channel in channels:
        df = pd.read_csv(f"{path}p{channel}_obs_vs_predict.csv")
        # df[["Labels", "Predictions"]] = df.transform({"Labels": np.exp, "Predictions": np.exp})

        if "date_time" in df.columns:
            df = df.transform({'date_time': lambda x: datetime.utcfromtimestamp(float(x))})
            df = df.rename(columns={"date_time": "ts"})

        if "ts" in df.columns:
            df["ts"] = df["ts"].astype("datetime64[ns]")

        # From Tanveer.
        # Transform the data back to the raw format.
        labels_r = pd.DataFrame(df, columns=['Labels'])
        labels_r = labels_r.rename(columns={'Labels': f"p{channel}"})
        predictions_r = pd.DataFrame(df, columns=['Predictions'])
        predictions_r = predictions_r.rename(columns={'Predictions': f"p{channel}"})
        lebels_inv_r = pipe_robust.invert_y_trans(labels_r)
        predictions_inv_r = pipe_robust.invert_y_trans(predictions_r)

        df["Labels"] = lebels_inv_r.to_numpy().ravel()
        df["Predictions"] = predictions_inv_r.to_numpy().ravel()

        dfs.append(df)

    return dfs
