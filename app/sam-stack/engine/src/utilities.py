# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys
import numbers
import datetime
import boto3
import numpy as np

from warnings import warn
from numpy.lib.stride_tricks import as_strided
from botocore.exceptions import ClientError

from forecast_trend import *
from es_trend import *
from arima_trend import *

np.set_printoptions(threshold=np.inf)

EPS = np.finfo(float).eps

# Expected input data columns
DATA_COLS = ["timestamp", "channel", "item_id", "demand", "family"]

# Only the "demand" column is of type `int`, all others are `str`
INPUT_DTYPE = {c: (float if c == "demand" else str) for c in DATA_COLS}


def multi_horizon_test(out, window_len=8):
    """
    This code is there to take a stack of forecasts generated at every timestep for backtesting.
    EG: Every row represents the forecast created for the next week. We only look at the first eight columns or weeks of the forecast

    Week 1 forecast [forecasts Week 1 +1 , forecasts Week 1 +2, forecasts Week 1 +3 ]
    Week 2 forecast [forecasts Week 2 +1 , forecasts Week 2 +2, forecasts Week 2 +3 ]

    It then gets converted to this format

    [Forecast for today - 8 Weeks ago, Forecast for today - 7 Weeks ago, Forecast for today - 6 Weeks ago]
    [Forecast for today - 8 Weeks ago, Forecast for today - 7 Weeks ago, Forecast for today - 6 Weeks ago]

    This makes it easier when you have an actual value and you can compare the forecasts you made in the past all on the same row.

    """
    window_shape = (window_len, window_len)
    strided = view_as_windows(out, window_shape)
    multiHorizon = []
    for i in range(strided.shape[0]):
        line = np.diag(np.rot90(strided[i, 0, :, :]))
        if line[-1] == 0:
            line = np.full_like(line, 0)
        multiHorizon.append(line)

    return np.vstack(multiHorizon).round(0)


def rolling_window(a, window):
    """
    This isnt used at the moment but it can be used to create a numpy matrix from a rolling window over a vector
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def view_as_windows(arr_in, window_shape, step=1):
    """
    This code has been copied from an older version of scikit-image to save space on lambda. It is needed to generate a 2D rolling window for the multi_horizon_test
    """
    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError(f"`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def get_experiments(debug=False):
    """This is the list of experiments that serves as the grid search over
    different types of models that reside in arima_trend.py, es_trend.py and
    forecast_trend.py

    We search over

    1) Naive (rolling avg)
    2) Linear Trend (momentum)
    3) Single Exponential Smoothing
    4) Double Exponential Smoothing (Holt Method)
    5) ARIMA family AR, MA and ARMA
    6) Fourier extrapolation

    We have the ability for categories to share information by normalising the
    demand with the category_sum, forecasting through one of the base models
    above and de-normailising using projected category_sum This has the ability
    to remove category level trends and focus on the true demand at the SKU
    level.

    We also have the ability to choose only the most recent demand and make it
    more of a local forecasting model.  This has the affect of detrending the
    demand and very useful if the SKU has a clear peak and decline.

    We also have a log transform before forecasting to make the variance more
    stable. I have left this turned off so save a doubling of models

    So far we have more than 70 permutations of models to run.
    """

    config0 = {
        "model": forecast_naive,
        "model_name": "local_naive_model",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }
    config00 = {
        "model": forecast_naive,
        "model_name": "local_naive_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }
    config1 = {
        "model": forecast_trend,
        "model_name": "local_trend_model",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }
    config2 = {
        "model": forecast_trend,
        "model_name": "local_trend_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }
    config3 = {
        "model": forecast_es,
        "model_name": "local_es_model_seasonal_alpha_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "local_model": True,
    }
    config4 = {
        "model": forecast_es,
        "model_name": "local_es_model_alpha_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "local_model": True,
    }
    config5 = {
        "model": forecast_es,
        "model_name": "local_es_model_seasonal_alpha_0.4",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "local_model": True,
    }
    config6 = {
        "model": forecast_es,
        "model_name": "local_es_model_alpha_0.4",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "local_model": True,
    }
    config7 = {
        "model": forecast_es,
        "model_name": "local_es_model_seasonal_alpha_0.6",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "local_model": True,
    }
    config8 = {
        "model": forecast_es,
        "model_name": "local_es_model_alpha_0.6",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "local_model": True,
    }
    config9 = {
        "model": forecast_es,
        "model_name": "local_es_model_seasonal_alpha_0.8",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "local_model": True,
    }
    config10 = {
        "model": forecast_es,
        "model_name": "local_es_model_alpha_0.8",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "local_model": True,
    }
    config11 = {
        "model": forecast_es,
        "model_name": "local_es_model_seasonal_alpha_0.9",
        "log": False,
        "seasonality": True,
        "alpha": 0.9,
        "local_model": True,
    }
    config12 = {
        "model": forecast_es,
        "model_name": "local_es_model_alpha_0.9",
        "log": False,
        "seasonality": True,
        "alpha": 0.9,
        "local_model": True,
    }
    config13 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.2_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.2,
        "local_model": True,
    }
    config14 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.2_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.2,
        "local_model": True,
    }
    config15 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.4_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.2,
        "local_model": True,
    }
    config16 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.4_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.2,
        "local_model": True,
    }
    config17 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.6_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.2,
        "local_model": True,
    }
    config18 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.6_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.2,
        "local_model": True,
    }
    config19 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.8_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.2,
        "local_model": True,
    }
    config20 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.8_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.2,
        "local_model": True,
    }
    config21 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.2_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.5,
        "local_model": True,
    }
    config22 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.2_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.5,
        "local_model": True,
    }
    config23 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.4_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.5,
        "local_model": True,
    }
    config24 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.4_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.5,
        "local_model": True,
    }
    config25 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.6_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.5,
        "local_model": True,
    }
    config26 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.6_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.5,
        "local_model": True,
    }
    config27 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_seasonal_alpha_0.8_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.5,
        "local_model": True,
    }
    config28 = {
        "model": forecast_holt,
        "model_name": "local_holt_model_alpha_0.8_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.5,
        "local_model": True,
    }
    config29 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_seasonal_101",
        "log": False,
        "seasonality": True,
        "q": 1,
        "d": 0,
        "p": 1,
        "local_model": True,
    }
    config30 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_101",
        "log": False,
        "seasonality": True,
        "q": 1,
        "d": 0,
        "p": 1,
        "local_model": True,
    }
    config31 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_seasonal_201",
        "log": False,
        "seasonality": True,
        "q": 2,
        "d": 0,
        "p": 1,
        "local_model": True,
    }
    config32 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_201",
        "log": False,
        "seasonality": True,
        "q": 2,
        "d": 0,
        "p": 1,
        "local_model": True,
    }

    config33 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_seasonal_001",
        "log": False,
        "seasonality": True,
        "q": 0,
        "d": 0,
        "p": 1,
        "local_model": True,
    }
    config34 = {
        "model": forecast_arima,
        "model_name": "local_arima_model_001",
        "log": False,
        "seasonality": True,
        "q": 0,
        "d": 0,
        "p": 1,
        "local_model": True,
    }

    config35 = {
        "model": forecast_fourier,
        "model_name": "local_fourier_model",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }
    config36 = {
        "model": forecast_fourier,
        "model_name": "local_fourier_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": True,
    }

    configA0 = {
        "model": forecast_naive,
        "model_name": "naive_model",
        "log": False,
        "seasonality": True,
        "local_model": False,
    }
    configAA = {
        "model": forecast_naive,
        "model_name": "naive_model",
        "log": False,
        "seasonality": False,
        "local_model": False,
    }
    configA00 = {
        "model": forecast_naive,
        "model_name": "naive_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": False,
    }
    configA1 = {
        "model": forecast_trend,
        "model_name": "trend_model",
        "log": False,
        "seasonality": True,
        "local_model": False,
    }
    configA2 = {
        "model": forecast_trend,
        "model_name": "trend_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": False,
    }
    configA3 = {
        "model": forecast_es,
        "model_name": "es_model_seasonal_alpha_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "local_model": False,
    }
    configA4 = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "local_model": False,
    }
    configA5 = {
        "model": forecast_es,
        "model_name": "es_model_seasonal_alpha_0.4",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "local_model": False,
    }
    configA6 = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.4",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "local_model": False,
    }
    configA7 = {
        "model": forecast_es,
        "model_name": "es_model_seasonal_alpha_0.6",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "local_model": False,
    }
    configA8 = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.6",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "local_model": False,
    }
    configA9 = {
        "model": forecast_es,
        "model_name": "es_model_seasonal_alpha_0.8",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "local_model": False,
    }
    configA10 = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.8",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "local_model": False,
    }
    configA11 = {
        "model": forecast_es,
        "model_name": "es_model_seasonal_alpha_0.9",
        "log": False,
        "seasonality": True,
        "alpha": 0.9,
        "local_model": False,
    }
    configA12 = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.9",
        "log": False,
        "seasonality": True,
        "alpha": 0.9,
        "local_model": False,
    }
    configA13 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.2_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.2,
        "local_model": False,
    }
    configA14 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.2_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.2,
        "local_model": False,
    }
    configA15 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.4_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.2,
        "local_model": False,
    }
    configA16 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.4_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.2,
        "local_model": False,
    }
    configA17 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.6_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.2,
        "local_model": False,
    }
    configA18 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.6_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.2,
        "local_model": False,
    }
    configA19 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.8_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.2,
        "local_model": False,
    }
    configA20 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.8_beta_0.2",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.2,
        "local_model": False,
    }
    configA21 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.2_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.5,
        "local_model": False,
    }
    configA22 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.2_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.2,
        "beta": 0.5,
        "local_model": False,
    }
    configA23 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.4_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.5,
        "local_model": False,
    }
    configA24 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.4_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.4,
        "beta": 0.5,
        "local_model": False,
    }
    configA25 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.6_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.5,
        "local_model": False,
    }
    configA26 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.6_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.6,
        "beta": 0.5,
        "local_model": False,
    }
    configA27 = {
        "model": forecast_holt,
        "model_name": "holt_model_seasonal_alpha_0.8_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.5,
        "local_model": False,
    }
    configA28 = {
        "model": forecast_holt,
        "model_name": "holt_model_alpha_0.8_beta_0.5",
        "log": False,
        "seasonality": True,
        "alpha": 0.8,
        "beta": 0.5,
        "local_model": False,
    }
    configA29 = {
        "model": forecast_arima,
        "model_name": "arima_model_seasonal_101",
        "log": False,
        "seasonality": True,
        "q": 1,
        "d": 0,
        "p": 1,
        "local_model": False,
    }
    configA30 = {
        "model": forecast_arima,
        "model_name": "arima_model_101",
        "log": False,
        "seasonality": True,
        "q": 1,
        "d": 0,
        "p": 1,
        "local_model": False,
    }
    configA31 = {
        "model": forecast_arima,
        "model_name": "arima_model_seasonal_201",
        "log": False,
        "seasonality": True,
        "q": 2,
        "d": 0,
        "p": 1,
        "local_model": False,
    }
    configA32 = {
        "model": forecast_arima,
        "model_name": "arima_model_201",
        "log": False,
        "seasonality": True,
        "q": 2,
        "d": 0,
        "p": 1,
        "local_model": False,
    }

    configA33 = {
        "model": forecast_arima,
        "model_name": "arima_model_seasonal_001",
        "log": False,
        "seasonality": True,
        "q": 0,
        "d": 0,
        "p": 1,
        "local_model": False,
    }
    configA34 = {
        "model": forecast_arima,
        "model_name": "arima_model_001",
        "log": False,
        "seasonality": True,
        "q": 0,
        "d": 0,
        "p": 1,
        "local_model": False,
    }

    configA35 = {
        "model": forecast_fourier,
        "model_name": "fourier_model",
        "log": False,
        "seasonality": False,
        "local_model": False,
    }
    configA36 = {
        "model": forecast_fourier,
        "model_name": "fourier_model_seasonal",
        "log": False,
        "seasonality": True,
        "local_model": False,
    }

    experiments = [
        config0,
        config00,
        config1,
        config2,
        config3,
        config4,
        config5,
        config6,
        config7,
        config8,
        config9,
        config10,
        config11,
        config12,
        config13,
        config14,
        config15,
        config16,
        config17,
        config18,
        config19,
        config20,
        config21,
        config22,
        config23,
        config24,
        config25,
        config26,
        config27,
        config28,
        config29,
        config30,
        config31,
        config32,
        config33,
        config34,
        config35,
        config36,
        configAA,
        configA0,
        configA00,
        configA1,
        configA2,
        configA3,
        configA4,
        configA5,
        configA6,
        configA7,
        configA8,
        configA9,
        configA10,
        configA11,
        configA12,
        configA13,
        configA14,
        configA15,
        configA16,
        configA17,
        configA18,
        configA19,
        configA20,
        configA21,
        configA22,
        configA23,
        configA24,
        configA25,
        configA26,
        configA27,
        configA28,
        configA29,
        configA30,
        configA31,
        configA32,
        configA33,
        configA34,
        configA35,
        configA36,
    ]

    if debug:
        experiments = experiments[:1]

    return experiments


def best_model_forcast(config, use_all_backtests=True, model_name=None,
    ignore_naive=False, debug=False):
    """
    This is the main function that is called from this module

    It takes in a configuration dictionary that has the meta data as well as
    the demand series, runs all the experiments in get_experiments and
    forecasts using the best model.

    Parameters
    ----------
    config : dict
    use_all_backtests : bool, optional
        Set to True to average the cost over all of the sliding window
        backtests; otherwise, set to False to use only the final backtest
        window.
    model_name : str, optional
    ignore_naive : bool, optional
        Ignore naive forecasting approaches.

    Notes
    -----
    - This function is run remotely on Lambda using the custom PyWren runtime.

    """

    # The naive and exponential smoothing algos are use as benchmarks
    naive = {
        "model": forecast_naive,
        "model_name": "local_naive_model",
        "log": False,
        "seasonality": False,
        "local_model": False,
    }

    es = {
        "model": forecast_es,
        "model_name": "es_model_alpha_0.6",
        "log": False,
        "seasonality": False,
        "alpha": 0.6,
        "local_model": False,
    }

    # Get the naive and exponential-smoothing (ES) baseline forecasts
    ys_naive, yp_naive = backtest_slice_forecast(config, naive)
    ys_es, yp_es = backtest_slice_forecast(config, es)

    naive_mape = calc_mape(ys_naive, yp_naive)
    es_mape = calc_mape(ys_es, yp_es)

    if np.count_nonzero(config["demand"]) < config["horizon"] * 2:
        config["local_model"] = True

    experiments = get_experiments(debug=debug)

    # Restrict the possible models to explore.
    if ignore_naive:
        experiments = \
            [c for c in experiments if ("naive" not in c["model_name"])]

    if model_name is not None:
        experiments = [e for e in experiments if model_name in e["model_name"]]

    # generate the backtest "cost" for each "experiment", the final horizon
    # window is excluded from these experiments
    results = []

    for i, exp in enumerate(experiments, start=1):
        ys, yp = backtest_slice_forecast(config, exp)
        cost = calc_cost(ys, yp)
        results.append(cost)

    # find the best model according to the multiple backtesting results
    best_i = np.argsort(results)[0]
    best_model = experiments[best_i]
    best_model["Cost"] = results[best_i]
    best_model["Naive_Error_MAPE"] = naive_mape
    best_model["Naive_Error"] = naive_mape
    best_model["ES_Error"] = es_mape

    output = backtest_slice_forecast(config, best_model)

    yp_max_clip = 2 * config["demand"][-config["horizon"] :].max()

    # extract the (final horizon window) backtest values
    ys_engine, yp_engine = backtest_slice_forecast(config, best_model)
    yp_engine = yp_engine.round(0)

    # calc. the MAPE for the "best" engine model using the entire backtest
    engine_mape = calc_mape(ys_engine, yp_engine)

    # generate the final backtest forecast
    yp_engine_backtest = np.clip(
        generate_forecast_backtest_horizon(config, best_model),
        0,
        2 * config["demand"][-config["horizon"] :].max(),
    )

    # generate the full forecast using the best engine model
    yp_engine_fcast = generate_forecast(config, best_model)
    yp_engine_fcast = yp_engine_fcast.clip(0, yp_max_clip)

    ys_hist, forecasts = output

    best_model["Voyager_Historic_Demand"] = ys_hist.tolist()

    best_model["Voyager_Forecast_W-1"] = forecasts[:, 7].tolist()
    best_model["Voyager_Forecast_W-8"] = forecasts[:, 0].tolist()

    best_model["Voyager_Error"] = engine_mape
    best_model["Voyager_Error_MAPE"] = engine_mape

    best_model["forecast_horizon_backtest"] = yp_engine_backtest.round(0)
    best_model["forecast"] = yp_engine_fcast.round(0)

    best_model["forecast_p90"] = np.clip(
        generate_forecast_p90(config, best_model),
        0,
        2 * config["demand"][-config["horizon"] :].max(),
    )
    best_model["forecast_p10"] = np.clip(
        generate_forecast_p10(config, best_model),
        0,
        2 * config["demand"][-config["horizon"] :].max(),
    )

    best_model.pop("model", None)

    dict_results = {**config, **best_model}
    dict_results.pop("category_sum", None)

    return dict_results


def generate_forecast(config, best_model):
    """
    This generates the final forecast once the best model is found in the best_model_forecast function
    """

    horizon = config["horizon"]
    model = best_model["model"]
    model_config = best_model
    model_config["horizon"] = horizon

    if model_config["seasonality"]:
        demand = config["demand"] + 1
        seasonal_factor = config["category_sum"]

        deseasonal_demand = np.divide(demand, seasonal_factor)

        if model_config["local_model"]:
            deseasonal_forecast = model(deseasonal_demand[-horizon:], model_config)
        else:
            deseasonal_forecast = model(deseasonal_demand, model_config)

        forecast_seasonal_factor = forecast_fourier(seasonal_factor, model_config)

        results = np.multiply(deseasonal_forecast, forecast_seasonal_factor)
    else:
        demand = config["demand"] + 1
        if model_config["local_model"]:
            forecast = model(
                demand[
                    -horizon:,
                ],
                model_config,
            )
        else:
            forecast = model(demand, model_config)
        results = np.clip(forecast, 0, np.inf)

    return results - 1


def generate_forecast_backtest_horizon(config, best_model):

    """
    This generates the final forecast once the best model is found in the best_model_forecast function
    """

    horizon = config["horizon"]
    model = best_model["model"]
    model_config = best_model
    model_config["horizon"] = horizon

    if model_config["seasonality"]:
        demand = config["demand"][:-horizon] + 1
        seasonal_factor = config["category_sum"][:-horizon]

        deseasonal_demand = np.divide(demand, seasonal_factor)

        if model_config["local_model"]:
            deseasonal_forecast = model(deseasonal_demand[-horizon:], model_config)
        else:
            deseasonal_forecast = model(deseasonal_demand, model_config)

        forecast_seasonal_factor = forecast_fourier(seasonal_factor, model_config)

        results = np.multiply(deseasonal_forecast, forecast_seasonal_factor)

    else:
        demand = config["demand"] + 1
        if model_config["local_model"]:
            forecast = model(
                demand[
                    -horizon:,
                ],
                model_config,
            )
        else:
            forecast = model(demand, model_config)
        results = np.clip(forecast, 0, np.inf)

    return results


def generate_forecast_p90(config, best_model):

    """
    This generates the final forecast once the best model is found in the best_model_forecast function
    """

    horizon = config["horizon"]
    model = best_model["model"]
    model_config = best_model
    model_config["horizon"] = horizon

    if model_config["seasonality"]:
        # print("Seasonalworkflow")

        demand = config["demand_p90"]
        seasonal_factor = config["category_sum"]

        deseasonal_demand = np.divide(demand, seasonal_factor)

        if model_config["local_model"]:
            deseasonal_forecast = model(deseasonal_demand[-horizon:], model_config)
        else:
            deseasonal_forecast = model(deseasonal_demand, model_config)

        forecast_seasonal_factor = forecast_fourier(seasonal_factor, model_config)

        results = np.multiply(deseasonal_forecast, forecast_seasonal_factor)

    else:
        # print("Normalworkflow")
        demand = config["demand_p90"]
        if model_config["local_model"]:
            forecast = model(
                demand[
                    -horizon:,
                ],
                model_config,
            )
        else:
            forecast = model(demand, model_config)
        results = np.clip(forecast, 0, np.inf)

    return results


def generate_forecast_p10(config, best_model):

    """
    This generates the final forecast once the best model is found in the best_model_forecast function
    """

    horizon = config["horizon"]
    model = best_model["model"]
    model_config = best_model
    model_config["horizon"] = horizon

    if model_config["seasonality"]:
        # print("Seasonalworkflow")

        demand = config["demand_p10"]
        seasonal_factor = config["category_sum"]

        deseasonal_demand = np.divide(demand, seasonal_factor)

        if model_config["local_model"]:
            deseasonal_forecast = model(deseasonal_demand[-horizon:], model_config)
        else:
            deseasonal_forecast = model(deseasonal_demand, model_config)

        forecast_seasonal_factor = forecast_fourier(seasonal_factor, model_config)

        results = np.multiply(deseasonal_forecast, forecast_seasonal_factor)

    else:
        # print("Normalworkflow")
        demand = config["demand_p10"]
        if model_config["local_model"]:
            forecast = model(
                demand[
                    -horizon:,
                ],
                model_config,
            )
        else:
            forecast = model(demand, model_config)
        results = np.clip(forecast, 0, np.inf)

    return results


'''
def backtest_slice_forecast_OLD(config, experiment):
    """
    This generates the backtesting results from within the best_model_forecast function
    """
    horizon = config["horizon"]

    if np.count_nonzero(config["demand"]) < horizon*2:
        experiment["local_model"] = True

    model = experiment["model"]
    model_config = experiment
    historic_demand = config["demand"][horizon+8-1:-horizon-1]

    experiment["horizon"] = horizon
    if experiment["seasonality"]:

        demand = config["demand"] + 1
        seasonal_factor = config["category_sum"]

        deseasonal_demand = np.divide(demand,seasonal_factor)

        if experiment["local_model"]:
            deseasonal_forecasts = [model(deseasonal_demand[i-horizon:i],model_config) for i in range(horizon,demand.shape[0]- horizon -1)]
        else:
            deseasonal_forecasts = [model(deseasonal_demand[:i],model_config) for i in range(horizon,demand.shape[0]- horizon -1)]

        deseasonal_results = np.vstack(deseasonal_forecasts)

        forecast_seasonal_factor = [forecast_fourier(seasonal_factor[:i],model_config) for i in range(horizon,demand.shape[0]- horizon -1)]
        forecast_seasonal_factor = np.vstack(forecast_seasonal_factor)

        results = np.multiply(deseasonal_results,forecast_seasonal_factor)
    else:
        demand = config["demand"] + 1
        if experiment["local_model"]:
            forecasts = [model(demand[i-horizon:i],model_config) for i in range(horizon,demand.shape[0]- horizon -1)]
        else:
            forecasts = [ model(demand[:i], model_config) for i in range(horizon,demand.shape[0] - horizon - 1)]
        results = np.vstack(forecasts)

    results = multi_horizon_test(results[:,:8]) - 1
    historic_windows = rolling_window(historic_demand, 8)

    return (historic_demand, results.round(0))
'''


def backtest_slice_forecast(config, experiment):
    """Generate the backtesting results across multiple historic horizon
    windows.

    """
    horizon = config["horizon"]

    # Enforce the local model for intermittent time-series
    if np.count_nonzero(config["demand"]) < horizon * 2:
        experiment["local_model"] = True

    model = experiment["model"]
    model_config = experiment
    historic_demand = config["demand"][horizon + 8 - 1 : -horizon - 1]
    experiment["horizon"] = horizon

    if experiment["seasonality"]:
        demand = config["demand"] + 1
        seasonal_factor = config["category_sum"]

        deseasonal_demand = np.divide(demand, seasonal_factor)

        if experiment["local_model"]:
            deseasonal_forecasts = [
                model(deseasonal_demand[i - horizon : i], model_config)
                for i in range(horizon, demand.shape[0] - horizon - 1)
            ]
        else:
            deseasonal_forecasts = [
                model(deseasonal_demand[:i], model_config)
                for i in range(horizon, demand.shape[0] - horizon - 1)
            ]

        deseasonal_results = np.vstack(deseasonal_forecasts)

        forecast_seasonal_factor = [
            forecast_fourier(seasonal_factor[:i], model_config)
            for i in range(horizon, demand.shape[0] - horizon - 1)
        ]

        forecast_seasonal_factor = np.vstack(forecast_seasonal_factor)

        results = np.multiply(deseasonal_results, forecast_seasonal_factor)
    else:
        demand = config["demand"] + 1
        if experiment["local_model"]:
            forecasts = [
                model(demand[i - horizon : i], model_config)
                for i in range(horizon, demand.shape[0] - horizon - 1)
            ]
        else:
            forecasts = [
                model(demand[:i], model_config)
                for i in range(horizon, demand.shape[0] - horizon - 1)
            ]
        results = np.vstack(forecasts)

    results = multi_horizon_test(results[:, :8]) - 1
    historic_windows = rolling_window(historic_demand, 8)
    results = results[: historic_windows.shape[0]]

    return (historic_windows, results.round(0))


def error_backtest_OLD(inp):
    """"""

    historic_demand, results = inp

    mask = np.argwhere(historic_demand > 0.01)

    historic_demand = historic_demand[:, np.newaxis]

    residual = historic_demand - results

    residual = residual[mask, :]
    quantile = 0.6

    return np.mean(
        np.maximum(quantile * residual.ravel(), (quantile - 1) * residual.ravel())
    )


def error_backtest(inp):
    """"""

    historic_demand, results = inp

    mask = np.argwhere(historic_demand > 0.01)

    # historic_demand = historic_demand[:, np.newaxis]
    residual = historic_demand - results
    residual = residual[mask, :]
    quantile = 0.6

    return np.mean(
        np.maximum(quantile * residual.ravel(), (quantile - 1) * residual.ravel())
    )


def error_backtest_MAE_OLD(inp):

    """
    This generates the mean absolute error from the backtesting results
    """

    historic_demand, results = inp

    mask = np.argwhere(historic_demand > 0.01)

    historic_demand = historic_demand[:, np.newaxis]

    residual = historic_demand - results

    residual = residual[mask, :]
    quantile = 0.5

    return np.mean(
        np.maximum(quantile * residual.ravel(), (quantile - 1) * residual.ravel())
    )


'''
def error_backtest_MAE(inp):
    """
    """
    historic_demand, results = inp
    historic_demand = historic_demand[:, np.newaxis]
    historic_demand[historic_demand<1] = -1
    numerator = abs(results- historic_demand)
    error_score1 = np.divide(numerator,historic_demand)
    score1 = error_score1.ravel()
    score1 = score1[score1>0]
    score1 = np.clip(score1,0,1)
    error =  np.nanmean(score1)
    return error
'''


def error_backtest_MAE(inp):
    """"""
    historic_demand, results = inp
    historic_demand = historic_demand[:, np.newaxis]
    historic_demand[historic_demand < 1] = -1

    numerator = abs(results - historic_demand)
    error_score1 = np.divide(numerator, historic_demand)
    score1 = error_score1.ravel()
    score1 = score1[score1 > 0]
    score1 = np.clip(score1, 0, 1)
    error = np.nanmean(score1)
    return error


def calc_aape(ys, yp):
    """"""
    ys, yp = np.array(ys), np.array(yp)
    aape = np.arctan(np.abs(np.divide((ys - yp), np.clip(ys, EPS, None))))
    return aape


def calc_maape(ys, yp):
    """Calculate mean arctangent absolute percentage error (MAAPE)."""
    return calc_aape(ys, yp).mean()


def calc_ape(ys, yp):
    """Calculate the (vectorized) absolute percentage error."""
    ys = np.array(ys)
    yp = np.clip(np.array(yp), 0, None)

    with np.errstate(divide="ignore"):
        ape = np.divide(np.abs(yp - ys), ys)
        ape = np.clip(ape, 0, 1)

    return ape


def calc_mape(ys, yp):
    """Calculate mean absolute percentage error (MAPE). Note, MAPE is clipped
    to the range [0.0, 1.0], to allow for accuracy (%) to be calculated as
    100 * (1. - MAPE).

    """

    return np.nanmean(calc_ape(ys, yp))


def calc_cost(ys, yp):
    """"""

    historic_demand, results = ys, yp

    mask = np.argwhere(historic_demand > 0.01)

    # historic_demand = historic_demand[:, np.newaxis]
    residual = historic_demand - results

    try:
        residual = residual[mask, :]
    except IndexError:
        pass

    quantile = 0.6

    return np.mean(
        np.maximum(quantile * residual.ravel(), (quantile - 1) * residual.ravel())
    )


def error_backtest_MAPE(inp):
    """"""

    historic_demand, results = inp
    historic_demand = historic_demand[:, np.newaxis]
    numerator = abs(results - historic_demand)
    denominator = np.clip(historic_demand, 1, np.inf)
    error_score = np.divide(numerator, denominator)
    error = np.mean(error_score).round(3)

    return error


def remap_freq(freq):
    """Remap the forecasting frequency to an explicit offset alias and also to
    conform to the Amazon Forecast standard of using "Monday" as the start of
    the week.

    """

    if freq in ("W", "W-SUN", "W-MON"):
        freq = "W-MON"
    elif freq in (
        "M",
        "MS",
    ):
        freq = "MS"
    else:
        raise NotImplementedError

    return freq
