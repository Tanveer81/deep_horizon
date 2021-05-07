#!/usr/bin/env python
# coding: utf-8
"""
NAME:
    model_analysis

DESCRIPTION:
    Machine learning module for Python
    ==================================

    model_analysis is a Python module that provides
    analysis functionality for trained models.
    It offers

"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
import sklearn
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


def evaluate_model_channel_separated(y_true: np.ndarray, y_pred: np.ndarray) \
        -> np.ndarray:
    """
    Evaluates the given prediction for one or all channels.
    It therefore computes the following measurements:
        Mean Squared Error
        Mean Absolute Error
        Pearson Correlation
        Spearman Correlation with and without non zero values
        R2-Score
        Prediction Efficiency
        p_value of sp correlation
        p_value of two sample t test

    Parameters:
        y_true --- numpy.ndarray
            Ground truth (correct) target values.
        y_pred --- numpy.ndarray
            Estimated target values.

    Returns:
        List[float] containing all computed values
        for the measurements mentioned above.
        It can happen that there are nan-values in the list.
    """

    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        print(
            "Error: Input arrays must have the same shape and be 2-dimensional!"
        )
        raise SystemError

    evaluation = []

    # Evaluate each channel separately
    for channel in range(y_true.shape[1]):
        evaluation.append(
            evaluate_model(y_true[:, channel], y_pred[:, channel]))

    return np.asarray(evaluation)


def evaluate_model_multi_channel(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> dict:
    """
    DESCRIPTION
        Evaluates a multi-channel prediction as mean over the single channels

    Parameters:
        y_true --- numpy.ndarray
            Ground truth (correct) target values.
        y_pred --- numpy.ndarray
            Estimated target values.

    Returns:
        List[float]
        evaluate_model_channel_separated
        dict
        Te dictionary is ontaining all computed values
        for the measurements mentioned above.
        It can happen that there are nan-values in the list.

    """

    res_channels = evaluate_model_channel_separated(y_true, y_pred)

    mse = 0
    mae = 0
    pcc = 0
    scc = 0
    scc_nz = 0
    r_2 = 0
    p_values = []
    nz_p_values = []
    two_sample_ttests = []
    two_sample_ttests_nz = []

    for res in res_channels:
        mse += res['mse']
        mae += res['mae']
        pcc += res['pc']
        scc += res['sc']
        scc_nz += res['sc_nz']
        r_2 += res['r_2']
        p_values.append(res['sc_pvalue'])
        nz_p_values.append(res['sc_nz_pvalue'])
        two_sample_ttests.append(res['two_sample_ttest'])
        two_sample_ttests_nz.append(res['two_sample_ttest_nz'])

    dividor = len(res_channels)
    mse /= dividor
    mae /= dividor
    pcc /= dividor
    scc /= dividor
    scc_nz /= dividor
    r_2 /= dividor

    return res_channels, dict(mse=mse, mae=mae, pc=pcc, sc=scc, sc_nz=scc_nz, r_2=r_2,
                              p_values=p_values, nz_p_values=nz_p_values,
                              two_sample_ttests=two_sample_ttests,
                              two_sample_ttests_nz=two_sample_ttests_nz)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates the given prediction of all included channels.
    It therefore computes the following measurements:
        Mean Squared Error
        Mean Absolute Error
        Pearson Correlation
        Spearman Correlation
        R2-Score
        Prediction Efficiency

    Parameters:
        y_true --- numpy.ndarray
            Ground truth (correct) target values.
        y_pred --- numpy.ndarray
            Estimated target values.

    Returns:
        dict
        Te dictionary is ontaining all computed values
        for the measurements mentioned above.
        It can happen that there are nan-values in the list.
    """

    # MSE
    mse = mean_squared_error(y_true, y_pred)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # PC
    prc = pearsonr(y_true.ravel(), y_pred.ravel())
    prc = prc[0]

    # SC
    spc = spearmanr(y_true.ravel(), y_pred.ravel())
    spc_pvalue = spc.pvalue
    spc_corr = spc.correlation

    nz_idx = y_true != y_true.min()
    spnz = spearmanr(y_true[nz_idx].ravel(), y_pred[nz_idx].ravel())
    spnz_corr = spnz.correlation
    spnz_pvalue = spnz.pvalue

    two_sample_ttest = ttest_ind(y_true.ravel(), y_pred.ravel(), equal_var=False)
    two_sample_ttest_nz = ttest_ind(y_true[nz_idx].ravel(), y_pred[nz_idx].ravel(), equal_var=False)

    # R2
    r_2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "pc": prc,
        "sc": spc_corr,
        "sc_pvalue": spc_pvalue,
        "sc_nz": spnz_corr,
        "sc_nz_pvalue": spnz_pvalue,
        "r_2": r_2,
        "two_sample_ttest": two_sample_ttest,
        "two_sample_ttest_nz": two_sample_ttest_nz
    }


def get_perm_importances(model, x_test: np.ndarray, y_test: np.ndarray) \
        -> sklearn.utils.Bunch:
    """
    Evaluates feature importances for a given model
    according to the given dataset.
    It uses the following method:
        Permutation Importances

    Parameters:
        model --- trained model for prediction
        X_test --- numpy.ndarray
        y_test numpy.ndarray

    Returns:
        sklearn.utils.Bunch
        containing attributes:  importances
                                importances_mean
                                importances_std
    """

    perm = permutation_importance(model,
                                  x_test,
                                  y_test,
                                  n_repeats=5,
                                  n_jobs=-1)  # noqa 501

    return perm


def get_embedded_importances(model) \
        -> np.ndarray:
    """
    Evaluates embedded importances for a given model
    It uses the following method:
        Impurity based Feature Importances

    Parameters:
        model --- trained sklearn model for prediction

    Returns:
        numpy.ndarray containing embedded feature importances.
    """

    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_

    print("Error: Model does not include embedded feature importances!")
    raise SystemError
