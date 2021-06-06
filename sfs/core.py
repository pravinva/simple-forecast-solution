import traceback
import pandas as pd
import numpy as np

from functools import partial
from tqdm.auto import tqdm
from numpy import fft
from numpy.lib.stride_tricks import sliding_window_view

OBJ_METRICS = ["smape_mean"]


#
# Forecast functions
#
def exp_smooth(y, horiz, **kwargs):
    """
    """

    alpha = kwargs.get("alpha", 0.2)
    local_mode = kwargs.get("local_mode", False)
    use_log = kwargs.get("use_log", False)

    if len(y) > 8:
        y = np.trim_zeros(y, trim ='f')

    if local_mode:
        y = y[-8:]

    if use_log:
        y = np.log1p(y)

    extra_periods = horiz-1

    f = [y[0]]

    # Create all the m+1 forecast
    for t in range(1,len(y)-1):
        f.append((1-alpha)*f[-1]+alpha*y[t])

    # Forecast for all extra months
    for t in range(extra_periods):
        # Update the forecast as the last forecast
        f.append(f[-1])    

    yp = np.array(f[-horiz:]).clip(0)

    if use_log:
        yp = np.exp(yp)

    return yp.round(0)


def holt(y, horiz, **kwargs):
    """
    """

    alpha = kwargs.get("alpha", 0.2)
    beta = kwargs.get("beta", 0.2)
    local_mode = kwargs.get("local_mode", False)
    use_log = kwargs.get("use_log", False)

    if len(y) == 1:
        y = np.append(y, 0)

    if len(y) > 8:
        y = np.trim_zeros(y, trim ='f')

    if local_mode:
        y = y[-8:]

    if use_log:
        y = np.log1p(y)

    extra_periods = horiz-1
    
    y = np.log1p(y)

    # Initialization
    f = [np.nan] # First forecast is set to null value
    a = [y[0]] # First level defined as the first demand point
    b = [y[1]-y[0]] # First trend is computed as the difference between the two first demand poiny

    # Create all the m+1 forecast
    for t in range(1,len(y)):
        # Update forecast based on last level (a) and trend (b)
        f.append(a[t-1]+b[t-1])

        # Update the level based on the new data point
        a.append(alpha*y[t]+(1-alpha)*(a[t-1]+b[t-1]))

        # Update the trend based on the new data point
        b.append(beta*(a[t]-a[t-1])+(1-beta)*b[t-1])

    # Forecast for all extra months
    for t in range(extra_periods):
        # Update the forecast as the most up-to-date level + trend
        f.append(a[-1]+b[-1])
        # the level equals the forecast
        a.append(f[-1])
        # Update the trend as the previous trend
        b.append(b[-1])

    yp = np.clip(np.exp(f[-horiz:]), 0, None).round(0)

    return yp


def fourier(y, horiz, n_harm=20, **kwargs):
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
    if len(y) == 1:
        y = np.append(y, 0)

    try:
        n = y.shape[0]
        t = np.arange(n)
        p = np.polyfit(t, y, 1) # find linear trend in x
        f = fft.fftfreq(n) # frequencies

        y_detrended = y - p[0] * t # detrended x
        y_freqdom = fft.fft(y_detrended) # detrended x in frequency domain
        indices = list(range(n))

        # sort indices by frequency, lower -> higher
        indices.sort(key=lambda i: np.absolute(f[i]))
    
        t = np.arange(n + horiz)
        restored_sig = np.zeros(t.size)

        for i in indices[:2 + n_harm * 2]:
            amp = np.absolute(y_freqdom[i]) / n # amplitude
            phase = np.angle(y_freqdom[i]) # phase
            restored_sig += amp * np.cos(2 * np.pi * f[i] * t + phase)

        yp = (restored_sig + p[0] * t)[-horiz:]
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return yp.clip(0).round(0)


def naive(y, horiz, **kwargs):
    """
    """
    y = np.nan_to_num(y.clip(0))
    y = preprocess(y, **kwargs)
    yp = np.clip(y.mean() * np.ones(horiz), 0, None).round(0)

    return yp


def linear_trend(y, horiz, **kwargs):
    """Generate a forecast using a linear model.

    Parameters
    ----------
    y : array_like
    cfg : dict

    Returns
    -------
    yp : array_like

    """

    def yp_linear(y, horiz):
        """Fit a simple linear model (AKA trend) and generate the
        corresponding forecast.

        """
        x = np.arange(y.size)
        mu = np.mean(y)
        y = y / mu
        theta = np.polyfit(x, y, 1)
        f = np.poly1d(theta)
        x1 = np.arange(y.size, y.size + horiz)
        yp = f(x1) * mu

        return yp

    use_log = kwargs.get("use_log", False)

    # pad singleton timeseries with a single zero
    if len(y) == 1:
        y = np.append(y, 0)

    try:
        if use_log and np.sum(y) > 20:
            if np.sum(y) > 20:
                # log-transform the historical demand
                yp = np.exp(yp_linear(np.log1p(y), horiz))
            else:
                yp = np.nan_to_num(y).mean() * np.ones(horiz)
        elif np.sum(y) > 0:
            yp = yp_linear(y, horiz)
        else:
            yp = np.nan_to_num(y).mean() * np.ones(horiz)
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    return np.clip(yp, 0, None).round(0)


def forecast_arima(y, horiz, q=1, d=0, p=1, trim_zeros=True,
    local_model=False, use_log=False):
    """
    """
    import traceback 

    def preprocess_y2(y, local_model=False, trim_zeros=True):
        """
        """
        if len(y) > 8 and trim_zeros:
            y = np.trim_zeros(y, trim="f")

        if local_model:
            y = y[-8:]

        return y
    
    y = preprocess_y2(y, local_model, trim_zeros) 

    try:
        extra_periods=horiz-1
        
        if use_log:
            y = np.log(y+1)
            m = ARIMA(q, d, p)
            pred = m.fit_predict(y)
            pred = m.forecast(y, horiz)
            yp = np.exp(pred[-horiz:]).round(0)
        else:
            m = ARIMA(q, d, p)
            pred = m.fit_predict(y)
            pred = m.forecast(y, horiz)
            yp = np.round(pred[-horiz:],0)
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    yp = np.clip(yp, 0, None).round(0)

    assert(len(yp) == horiz)

    return yp


#
# Experiments
#
def run_cv(func, y, horiz, step=1, **kwargs):
    """Run a sliding-window temporal cross-validation (aka backtest) using a 
    given forecasting function (`func`).
    
    """
    
    # allow only 1D time-series arrays
    assert(y.ndim == 1)
    
    #
    # |  y     |  horiz  |..............|
    # |  y      |  horiz  |.............|
    # |  y       |  horiz  |............|
    #   ::
    #   ::
    # |  y                    | horiz   |
    # 
    ixs = range(1, len(y)-horiz+1, step)
    yp = np.vstack([func(y=y[:i], horiz=horiz) for i in ixs])
    
    return yp


def run_cv_select(y, horiz, obj_metric="smape_mean", show_progress=False):
    """Run the timeseries cross-val model selection across the forecasting
    functions for a single timeseries (`y`) and horizon length (`horiz`).

    """

    assert(y.ndim == 1)
    assert(obj_metric in OBJ_METRICS)

    # these are the model configurations to run
    func_delayed = [
        ("naive", partial(naive)),
        ("naive|local", partial(naive, local_mode=True)),
        ("exp_smooth", partial(exp_smooth)),

        # fourier forecasts
        ("fourier|n_harm=25", partial(fourier, n_harm=25))]

    if show_progress:
        func_iter = tqdm(func_delayed)
    else:
        func_iter = iter(func_delayed)

    # sliding window horizon actuals
    Y = sliding_window_view(y[1:], horiz)

    Yps = []

    for model_name, func in func_iter:
        Yp = run_cv(func, y, horiz)
        assert(Yp.shape == Y.shape)

        # generate the final forecast
        yhat = func(y, horiz)
        assert(len(yhat) == horiz)

        Yps.append((model_name, Yp, yhat))

    # calculate the errors for each model configuration
    err_delayed = [("smape", calc_smape)]
    results_rows = []

    for model_name, Yp, yhat in Yps:
        results = { name : func(Y, Yp) for name, func in err_delayed }

        # keep the final forecast for the model
        results["yhat"] = yhat
        results["model"] = model_name
        results_rows.append(results)

    df_results = pd.DataFrame(results_rows)

    # calc. metric stats
    for metric, _ in err_delayed:
        df_results[f"{metric}_mean"] = \
            df_results[metric].apply(np.nanmean).round(4)
        df_results[f"{metric}_median"] = \
            df_results[metric].apply(np.nanmedian).round(4)
        df_results[f"{metric}_std"] = \
            df_results[metric].apply(np.nanstd).round(4)

    assert(obj_metric in df_results)

    # rank results according to the `obj_metric`
    df_results.sort_values(by=obj_metric, ascending=True, inplace=True)

    return Yps, df_results


#
# Error functions
#
def calc_smape(y, yp, axis=0):
    """Calculate the symmetric mean absolute percentage error (sMAPE).
    
    sMAPE is calulated as follows:
    
        sMAPE = Σ|y - yp| / Σ(y + yp)
    
    where, 0.0 <= sMAPE <= 1.0 (lower is better)
    
    """
    
    try:
        smape = np.nansum(np.abs(y - yp), axis=axis) / \
                    np.nansum(y + yp, axis=axis)
    except ZeroDivisionError:
        smape = 0.0
        
    return smape.round(4).astype("float16")


#
# Utilities
#
def preprocess(y, local_mode=False, use_log=False , trim_zeros=False):
    """
    """

    assert(len(y) > 0)

    if len(y) > 8 and trim_zeros:
        y = np.trim_zeros(y, trim ='f')

    if len(y) == 1:
        y = np.append(y, 0)

    if use_log:
        y = np.log1p(y)

    return y


def data_health_check(df):
    """Perform a health check of a timeseries dataset (provided as dataframe).

    """

    # TODO:

    return


def postprocess(df, df_results):
    """Post-process raw results to a user-friendly dataframe format, including
    timestamps etc. for easy file exporting.

    """

    # TODO:

    return


#
# ARIMA classes
#
class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None
        self.intercept_ = None
        self.coef_ = None
    
    def _prepare_features(self, x):
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def _least_squares(self, x, y):
        return np.linalg.pinv((x.T @ x)) @ (x.T @ y)
    
    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = self._least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta
        
    def predict(self, x):
        x = self._prepare_features(x)
        return x @ self.beta
    
    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA model.
        :param q: (int) Order of the MA model.
        :param p: (int) Order of the AR model.
        :param d: (int) Number of times the data needs to be differenced.
        """
        super().__init__(True)
        self.p = p
        self.d = d
        self.q = q
        self.ar = None
        self.resid = None

    def _lag_view(self, x, order):
        """
        For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]
        """
        y = x.copy()
        # Create features by shifting the window of `order` size by one step.
        # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
        x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])
        
        # Reverse the array as we started at the end and remove duplicates.
        # Note that we truncate the features [order -1:] and the labels [order]
        # This is the shifting of the features with one time step compared to the labels
        x = np.stack(x)[::-1][order - 1: -1]
        y = y[order:]

        return x, y

    def _difference(self, x, d=1):
        if d == 0:
            return x
        else:
            x = np.r_[x[0], np.diff(x)]
            return self._difference(x, d - 1)
        

    def _undo_difference(x, d=1):
        if d == 1:
            return np.cumsum(x)
        else:
            x = np.cumsum(x)
            return self._undo_difference(x, d - 1)
        
    def prepare_features(self, x):
        if self.d > 0:
            x = difference(x, self.d)
                    
        ar_features = None
        ma_features = None
        
        # Determine the features and the epsilon terms for the MA process
        if self.q > 0:
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p)
                self.ar.fit_predict(x)
            eps = self.ar.resid
            eps[0] = 0
            
            # prepend with zeros as there are no residuals_t-k in the first X_t
            ma_features, _ = self._lag_view(np.r_[np.zeros(self.q), eps], self.q)
            
        # Determine the features for the AR process
        if self.p > 0:
            # prepend with zeros as there are no X_t-k in the first X_t
            ar_features = self._lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
                                
        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features)) 
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None: 
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]
        
        return features, x[:n]
    
    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x)
        return features
            
    def fit_predict(self, x): 
        """
        Fit and transform input
        :param x: (array) with time series.
        """
        features = self.fit(x)
        return self.predict(x, prepared=(features))
    
    def predict(self, x, **kwargs):
        """
        :param x: (array)
        :kwargs:
            prepared: (tpl) containing the features, eps and x
        """
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)
        
        y = super().predict(features)
        self.resid = x - y

        return self.return_output(y)
    
    def return_output(self, x):
        if self.d > 0:
            x = undo_difference(x, self.d) 
        return x
    
    def forecast(self, x, n):
        """
        Forecast the time series.
        
        :param x: (array) Current time steps.
        :param n: (int) Number of time steps in the future.
        """
        features, x = self.prepare_features(x)
        y = super().predict(features)
        
        # Append n time steps as zeros. Because the epsilon terms are unknown
        y = np.r_[y, np.zeros(n)]
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)
