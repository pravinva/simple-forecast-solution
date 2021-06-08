# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import traceback
import inspect
import numpy as np

from numpy import fft


def preprocess_xs(xs, cfg, min_len=8, trim_zeros=True):
    """Preprocess an array, trimming and subsetting as required.

    """

    # trim if len > 8
    if len(xs) > 8 and trim_zeros:
        xs = np.trim_zeros(xs, trim="f")

    # reduce to only a local window
    if cfg["local_model"]:
        xs = xs[-8:]

    return xs


def forecast_fourier(xs, cfg):
    """Generate a forecast using Fourier extrapolation.

    Parameters
    ----------
    xs : array_like
    cfg : dict

    Results
    -------
    yp : array_like

    """

    horiz = cfg["horizon"]

    try:
        n = xs.size
        n_harm = 20                     # number of harmonics in model
        t = np.arange(n)
        p = np.polyfit(t, xs, 1)         # find linear trend in x
        f = fft.fftfreq(n)              # frequencies

        xs_detrended = xs - p[0] * t        # detrended x
        xs_freqdom = fft.fft(xs_detrended)  # detrended x in frequency domain
        indices = list(range(n))

        # sort indices by frequency, lower -> higher
        indices.sort(key=lambda i: np.absolute(f[i]))
    
        t = np.arange(n + horiz)
        restored_sig = np.zeros(t.size)

        for i in indices[:2 + n_harm * 2]:
            amp = np.absolute(xs_freqdom[i]) / n # amplitude
            phase = np.angle(xs_freqdom[i]) # phase
            restored_sig += amp * np.cos(2 * np.pi * f[i] * t + phase)

        yp = (restored_sig + p[0] * t)[-horiz:]
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return yp.clip(0).round(0)


def forecast_naive(xs, cfg):
    """Generate a "naive" forecast using the historic mean of the demand.

    Parameters
    ----------
    xs : array_like
    cfg : dict

    Returns
    ------
    yp : array_like

    """

    xs = np.array(xs, dtype=float)
    xs = np.nan_to_num(xs.clip(0))
    xs = preprocess_xs(xs, cfg)

    yp = xs.mean() * np.ones(cfg["horizon"])
    yp = yp.clip(0)

    return yp.round(0)


def forecast_trend(xs, cfg):
    """Generate a forecast using a linear model.

    Parameters
    ----------
    xs : array_like
    cfg : dict

    Returns
    -------
    yp : array_like

    """

    def yp_linear(xs, horiz):
        """Fit a simple linear model (AKA trend) and generate the corresponding
        forecast.

        """
        x = np.arange(xs.size)
        mu = np.mean(xs)
        y = xs / mu
        theta = np.polyfit(x, y, 1)
        f = np.poly1d(theta)
        x1 = np.arange(xs.size, xs.size + horiz)
        yp = f(x1) * mu

        return yp

    horiz = cfg["horizon"]
    use_log = cfg.get("log", False)
    xs = preprocess_xs(xs, cfg)

    try:
        if use_log and np.sum(xs) > 20:
            if np.sum(xs) > 20:
                # log-transform the historical demand
                yp = np.exp(yp_linear(np.log1p(xs), horiz))
            else:
                yp = np.nan_to_num(xs).mean() * np.ones(horiz)
        elif np.sum(xs) > 0:
            yp = yp_linear(xs, horiz)
        else:
            yp = np.nan_to_num(xs).mean() * np.ones(horiz)
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return yp.round(0)


#
# REFACTOR
#
def preprocess_xs2(xs, local_model=False, trim_zeros=True):
    """
    """
    if len(xs) > 8 and trim_zeros:
        xs = np.trim_zeros(xs, trim="f")

    if local_model:
        xs = xs[-8:]

    return xs


def forecast_fourier2(xs, horiz, n_harm=20):
    """Generate a forecast using Fourier extrapolation.

    Parameters
    ----------
    n_harm : int
        Number of harmonics in the model

    Results
    -------
    yp : array_like

    """

    # pad singleton timeseries with a single zero
    if len(xs) == 1:
        xs = np.append(xs, 0)

    try:
        n = xs.shape[0]
        t = np.arange(n)
        p = np.polyfit(t, xs, 1) # find linear trend in x
        f = fft.fftfreq(n) # frequencies

        xs_detrended = xs - p[0] * t # detrended x
        xs_freqdom = fft.fft(xs_detrended) # detrended x in frequency domain
        indices = list(range(n))

        # sort indices by frequency, lower -> higher
        indices.sort(key=lambda i: np.absolute(f[i]))
    
        t = np.arange(n + horiz)
        restored_sig = np.zeros(t.size)

        for i in indices[:2 + n_harm * 2]:
            amp = np.absolute(xs_freqdom[i]) / n # amplitude
            phase = np.angle(xs_freqdom[i]) # phase
            restored_sig += amp * np.cos(2 * np.pi * f[i] * t + phase)

        yp = (restored_sig + p[0] * t)[-horiz:]
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return yp.clip(0).round(0)


def forecast_naive2(xs, horiz, local_model=False, trim_zeros=True):
    """
    """
    xs = np.nan_to_num(xs.clip(0))
    xs = preprocess_xs2(xs, local_model, trim_zeros)
    yp = np.clip(xs.mean() * np.ones(horiz), 0, None).round(0)

    return yp


def forecast_trend2(xs, horiz, use_log=False):
    """Generate a forecast using a linear model.

    Parameters
    ----------
    xs : array_like
    cfg : dict

    Returns
    -------
    yp : array_like

    """

    def yp_linear(xs, horiz):
        """Fit a simple linear model (AKA trend) and generate the
        corresponding forecast.

        """
        x = np.arange(xs.size)
        mu = np.mean(xs)
        y = xs / mu
        theta = np.polyfit(x, y, 1)
        f = np.poly1d(theta)
        x1 = np.arange(xs.size, xs.size + horiz)
        yp = f(x1) * mu

        return yp

    xs = preprocess_xs2(xs)

    # pad singleton timeseries with a single zero
    if len(xs) == 1:
        xs = np.append(xs, 0)

    try:
        if use_log and np.sum(xs) > 20:
            if np.sum(xs) > 20:
                # log-transform the historical demand
                yp = np.exp(yp_linear(np.log1p(xs), horiz))
            else:
                yp = np.nan_to_num(xs).mean() * np.ones(horiz)
        elif np.sum(xs) > 0:
            yp = yp_linear(xs, horiz)
        else:
            yp = np.nan_to_num(xs).mean() * np.ones(horiz)
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return np.clip(yp, 0, None).round(0)

