# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys

PWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PWD, "..", "src")))

import numpy as np
import pandas as pd
import pytest
import voyager as vr

from forecast_trend import forecast_naive
from voyager import Engine, grouper
from utilities import INPUT_DTYPE, get_experiments

DATA_DIR = os.path.join(PWD, "data")


@pytest.fixture
def setup_simple_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "sample-demand-01-small.csv"),
                     dtype=INPUT_DTYPE)
    return df


def test_grouper():
    """
    """

    exp = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    assert(list(grouper(3, "ABCDEFG")) == exp)

    return


def test_jsonify_wren_rows():
    """
    """

    wren_rows = [
        {
            "demand": np.array([10, 20, 30, 40, 50]),
            "demand_p10": np.array([0, 2, 3, 4, 5]),
            "demand_p90": np.array([100, 200, 300, 400, 500]),
            "forecast_horizon_backtest": np.array([0, 2, 3, 4, 5]),
            "forecast": np.array([0, 2, 3, 4, 5]),
            "forecast_p10": np.array([0, 2, 3, 4, 5]),
            "forecast_p90": np.array([100, 200, 300, 400, 500])
        }
    ]

    wren_rows = vr.jsonify_wren_rows(wren_rows)

    return


def test_forecast_naive(setup_simple_data):
    """
    """

    df = setup_simple_data

    cfg = {
        "horizon": 6,
        "local_model": False
    }

    xs = df["demand"].values
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([20]))
    assert(yp.shape[0] == cfg["horizon"])

    yp = forecast_naive([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cfg)

    assert(np.unique(yp) == np.array([5]))
    assert(yp.shape[0] == cfg["horizon"])

    yp = forecast_naive([1, 2, 3], cfg)

    assert(np.unique(yp) == np.array([2]))
    assert(yp.shape[0] == cfg["horizon"])

    yp = forecast_naive([1, 2, -3], cfg)

    assert(np.unique(yp) == np.array([1]))
    assert(yp.shape[0] == cfg["horizon"])

    yp = forecast_naive([1, 2, -3, None], cfg)

    assert(np.unique(yp) == np.array([1]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = [0, 0, 0, 0, 0, 10, 10, 10, 10]
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([10]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = [0, 0, 0, 0, 10, 10, 10, 10]
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([5]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = [1, 1, 1, 0, 0, 0, 0]
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([0]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = [1, 2, 3, 0, 0, 0, 0, 0]
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([1]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = np.array([1, 2, -3, 0, 0, 0, 0, 0])
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([0]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = np.array([1, 2, np.nan, 0, 0, 0, 0, 0])
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([0]))
    assert(yp.shape[0] == cfg["horizon"])

    #
    # Test the local naive forecast
    #
    cfg = {
        "horizon": 6,
        "local_model": True
    }

    xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([6]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = [1,2,3]
    yp = forecast_naive([1, 2, 3], cfg)

    assert(np.unique(yp) == np.array([2]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = np.array([10000, 20000, 16, 0, 0, 0, 0, 0, 0, 0])
    yp = forecast_naive(xs, cfg)

    assert(np.unique(yp) == np.array([2]))
    assert(yp.shape[0] == cfg["horizon"])

    xs = np.array([10000, 20000, 16, 0, 0, 0, 0, 0, 0, 0])
    yp = forecast_naive(xs, cfg)

    return


def test_forecast_trend(setup_simple_data):
    """
    """

    from forecast_trend import forecast_trend

    df = setup_simple_data

    cfg = {
        "horizon": 12,
        "local_model": False, 
    }

    xs = df["demand"].values
    yp = forecast_trend(xs, cfg)

    assert(np.unique(yp) == np.array([24]))
    assert(np.size(yp) == cfg["horizon"])

    return


def test_forecast_fourier(setup_simple_data):
    """
    """

    from forecast_trend import forecast_fourier

    df = setup_simple_data
    cfg = {
        "horizon": 12,
        "local_model": False, 
    }

    xs = df["demand"].values
    yp = forecast_fourier(xs, cfg)

    assert(np.unique(yp) == np.array([17]))
    assert(yp.size == cfg["horizon"])

    xs = np.zeros(df["demand"].shape[0])
    yp = forecast_fourier(xs, cfg)

    assert(np.unique(yp) == np.array([0]))
    assert(yp.size == cfg["horizon"])

    return


def test_engine():
    fn = os.path.join(DATA_DIR, "sample-demand-01-small.csv")
    fc_freq = "W"
    horizon = 6

    engine = Engine(fn=fn, fc_freq=fc_freq, horizon=horizon, debug=True)
    engine.forecast()

    return


def test_get_experiments():
    """
    """

    experiments = get_experiments()

    assert(len(experiments) > 0)

    return
