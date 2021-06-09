import traceback
import pandas as pd
import numpy as np

from collections import OrderedDict
from functools import partial
from tqdm.auto import tqdm

from scipy import signal, stats
from numpy import fft
from numpy.lib.stride_tricks import sliding_window_view

EXP_COLS = ["timestamp", "channel", "family", "item_id", "demand"]
GROUP_COLS = ["channel", "family", "item_id"]
OBJ_METRICS = ["smape_mean"]

# these define how long a tail period is for each freq.
TAIL_LEN = {"D": 56, "W": 8, "M": 2}


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


#
# Analysis
#
def analyze_dataset(df, freq):
    """Run analyses on each timeseries, e.g. to determine
    retired/intermittent/continuous timeseries. This needs to be run on a
    non-normalized dataframe (i.e. prior to pre-processing in the pipeline).

    """

    def _retired(y):
        return y[-tail_len:].sum() < 1

    def _life_periods(y):
        y = np.trim_zeros(y)
        y = y[np.logical_not(np.isnan(y)) & (y > 0)]
        return len(y)

    def _category(r):
        if r["retired"]:
            if r["life_periods"] < r["len"] / 4.0:
                return "short"
            return "med"
        else:
            return "cont"

    def _spectral_entropy(y):
        y = y[np.logical_not(np.isnan(y))]
        f, Pxx_den = signal.periodogram(y)
        psd_norm = np.divide(Pxx_den, Pxx_den.sum())
        psd_norm += 1e-6
        return -np.multiply(psd_norm, np.log2(psd_norm)).sum().round(2)

    def _lambda_boxcox(y):
        return stats.boxcox(y.clip(lower=0.1))[1].round(2)

    assert "demand" in df

    tail_len = TAIL_LEN[freq]

    df_analysis = \
        ts_groups(df).agg({"demand": [_retired, _life_periods, len,
                                      _spectral_entropy, _lambda_boxcox]})

    # flatten column names
    df_analysis.columns = ["|".join([c.lstrip("_") for c in col])
                              .strip(" |")
                              .replace("demand|", "")
                           for col in df_analysis.columns.values]

    df_analysis["intermittent"] = df_analysis["spectral_entropy"] > 5.0

    # classify series as short, medium ("med"), or continuous ("cont")
    df_analysis["category"] = df_analysis.apply(_category, axis=1)

    df_analysis = df_analysis.astype({"life_periods": int, "len": int})

    return df_analysis


def summarize(df, freq):
    """Analyze the timeseries of a dataframe, providing summaries of missing
    dates, min-max values, counts etc.

    """

    # no. of missing dates b/w the first and last dates of the timeseries
    def null_count(xs):
        return xs.isnull().sum().astype(int)

    # no. days of non-zero demand
    def nonzero_count(xs):
        return (xs > 0).sum().astype(int)

    def date_count(xs):
        return xs.index.nunique()

    df_grps = df.groupby(GROUP_COLS)

    # get per-sku summary
    df_sku_summary = \
        df_grps.agg({"demand": [null_count, nonzero_count, np.nanmean, np.nanmedian],
                     "timestamp": [min, max, date_count]}) \
               .rename({"null_count": "missing_dates"}, axis=1)

    # get overall summary
    # - most comm

    return df_sku_summary


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

    assert len(y) > 0
    assert y.ndim == 1
    assert obj_metric in OBJ_METRICS

    # pad a length sequence with a single `1`
    if len(y) < 2:
        y = np.pad(y, [1,0], constant_values=1)

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

    # shrink the horizon if it is >= the timeseries
    if horiz >= len(y):
        horiz = len(y) - 1

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

    assert obj_metric in df_results

    # rank results according to the `obj_metric`
    df_results.sort_values(by=obj_metric, ascending=True, inplace=True)
    df_results["rank"] = np.arange(len(df_results)) + 1

    return df_results


def run_pipeline(df, horiz, freq, obj_metric="smape_mean"):
    """Run he CV model selection over *all* timeseries in a dataframe.
    Note that this is a generator and will yield one results dataframe per
    timeseries at a single iteration.

    """

    df = load_data(df)

    # resample the input dataset to the desired frequency.
    df = resample(df, freq)

    for key, df_grp in df.groupby(GROUP_COLS, as_index=False, sort=False):
        y = df_grp["demand"].values

        # generate forecast results for a single timeseries
        df_results = run_cv_select(y, horiz, obj_metric)
        df_results.insert(0, "channel", key[0])
        df_results.insert(1, "family", key[1])
        df_results.insert(2, "item_id", key[2])

        # make the historical dataframe
        df_hist = df_grp[EXP_COLS]
        df_hist["type"] = "actual"

        # get the forecast from the best model
        yhat = df_results.iloc[0]["yhat"]
        yhat_ts = pd.date_range(df_hist["timestamp"].max(),
                                periods=len(yhat)+1, freq=freq, closed="right")

        assert len(yhat_ts) > 0
        assert len(yhat_ts) == len(yhat)

        # make the forecast dataframe
        df_pred = pd.DataFrame({"demand": yhat, "timestamp": yhat_ts})
        df_pred.insert(0, "channel", key[0])
        df_pred.insert(1, "family", key[1])
        df_pred.insert(2, "item_id", key[2])
        df_pred["type"] = "fcast"
        df_pred.set_index("timestamp", drop=False, inplace=True)

        # combine historical and predictions dataframes, re-ordering columns
        df_pred = df_hist.append(df_pred)

        yield df_results, df_pred

    return


#
# Error metrics
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
        
    return smape.round(4)


#
# Data wrangling
#
def load_data(data, impute_freq=None):
    """Read a raw timeseries dataset from disk, S3, or a dataframe. Note that
    the "timestamp" column will be removed as a bonafide column and set as 
    the dataframe (timeseries) index.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to the data file if `str`, otherwise a "raw" timeseries dataframe.
    impute_freq : str or None; optional
        If `str`, impute the missing dates between the first and last
        dates, according to the sampling frequency `str` value.

    Returns
    -------
    pd.DataFrame

    """

    if isinstance(data, str):
        if data.endswith(".csv.gz"):
            _read_func = partial(pd.read_csv, compression="gzip")
        elif data.endswith(".csv"):
            _read_func = partial(pd.read_csv)
        else:
            raise NotImplementedError

        # cast exp. columns to correct types
        df = _read_func(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise NotImplementedError

    # enforce column datatypes
    df = df.astype({"channel": str, "family": str, "item_id": str,
                    "demand": float})

    # set timeseries dataframe index
    df.set_index(pd.DatetimeIndex(df.pop("timestamp")), inplace=True)
    df.index.name = None

    if impute_freq is not None:
        df = impute_dates(df, impute_freq)

    return df


def resample(df, freq):
    """Resample a dataframe to a new frequency. Note that if a period in the
    new frequency contains only nulls, then the resulting resampled sum is NaN.

    Parameters
    ----------
    df : pd.DataFrame
    freq : str

    """
    def _sum(y):
        if np.all(pd.isnull(y)):
            return np.nan
        return np.nansum(y)

    df = df.groupby(GROUP_COLS, sort=False) \
           .resample(freq) \
           .agg({"demand": _sum}) \
           .reset_index(level=[0,1,2])

    return df


def impute_dates(df, freq, dt_stop=None):
    """Fill missing dates in the timeseries dataframe.

    Parameters
    ----------
    dd : pd.DataFrame
    freq : str
    dt_stop : str or datetime, optional
        The final date in the timeseries, it will be inferred if not specified.
        Otherwise, dates will be imputed between the end of the the timeseries
        and this date.

    Returns
    -------
    pd.DataFrame

    """

    def _impute_dates(dd, freq, dt_stop=None):
        """
        """

        if dt_stop is None:
            dt_stop = dd.index.max()

        # don't shrink timeseries
        assert(dt_stop >= dd.index.max())

        # assign the new timeseries index
        dd = dd.reindex(pd.date_range(dd.index.min(), dt_stop, freq=freq))

        # fwd fill only the essential columns
        dd.loc[:,GROUP_COLS] = dd[GROUP_COLS].ffill()

        return dd

    assert isinstance(df.index, pd.DatetimeIndex)

    if "timestamp" in df:
        df.drop("timestamp", axis=1, inplace=True)

    df = ts_groups(df).apply(partial(_impute_dates, freq=freq, dt_stop=dt_stop))
    df.index = df.index.droplevel(0)

    return df

#
# Utilities
#

def validate(df):
    """Validate the timeseries dataframe

    """

    err_msgs = []
    warn_msgs = []

    # check column names
    for col in EXP_COLS:
        if col not in df:
            err_msgs.append(f"**{col}** column missing")

    msgs = {
        "errors": err_msgs,
        "warnings": warn_msgs
    }

    is_valid_file = len(err_msgs) == 0

    return msgs, is_valid_file


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


def postprocess(df, df_results):
    """Post-process raw results to a user-friendly dataframe format, including
    timestamps etc. for easy file exporting.

    """

    # TODO:

    return


def ts_groups(self, **groupby_kwds):
    """Helper function to get the groups for a timeseries dataframe.

    """
    groupby_kwds.setdefault("as_index", False)
    groupby_kwds.setdefault("sort", False)

    return self.groupby(GROUP_COLS, **groupby_kwds)
