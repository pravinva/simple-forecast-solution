# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys

PWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PWD, "..", "src")))

import pandas as pd
import numpy as np

from utilities import calc_mape, calc_maape


def test_mape():
    ys = [1, 1, 1, 1, 1]
    yp = [1, 1, 1, 1, 1]

    assert(not np.any(calc_mape(ys, yp)))

    ys = [1, 1, 1, 1, 1]
    yp = [0, 0, 0, 0, 0]

    assert(np.isclose(calc_mape(ys, yp), 1.0))

    ys = [1, 1, 1, 1, 1]
    yp = [100, 100, 100, 100, 100]

    assert(np.isclose(calc_mape(ys, yp), 1.0))

    ys = [1, 0, 0, 1, 1]
    yp = [1, 1, 1, 1, 1]

    assert(np.isclose(calc_mape(ys, yp), 0.4))

    ys = [1, 0, 0, 1, 1]
    yp = [1, 100, 100, 1, 1]

    assert(np.isclose(calc_mape(ys, yp), 0.4))

    ys = [1, 1, 1, 1, 1]
    yp = [1, 100, 100, 1, 1]

    assert(np.isclose(calc_mape(ys, yp), 0.4))

    ys = [1, 1, 1, 1, 1]
    yp = [1, 2, 2, 1, 1]

    assert(np.isclose(calc_mape(ys, yp), 0.4))

    ys = [10, 10, 1, 1, 1]
    yp = [1, 2, 2, 0, 0]

    assert(np.isclose(calc_mape(ys, yp), 0.94))

    ys = [10, 10, 1, 1, 1]
    yp = [9, 9, 1, 0, 0]

    assert(np.isclose(calc_mape(ys, yp), 0.44))

    ys = [10, 10, 1, 0, 1]
    yp = [99, 99, 1, 1, 0]

    assert(np.isclose(calc_mape(ys, yp), 0.80))

    return
