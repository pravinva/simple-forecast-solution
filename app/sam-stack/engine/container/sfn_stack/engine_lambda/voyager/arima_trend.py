# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import numpy as np





def forecast_arima(tst, config):
    tmp = np.trim_zeros(tst, trim ='f')

    if len(tmp) > 8:
        tst = tmp

    if config["local_model"]:
        tst = tst[-8:]   
    
    horizon = config["horizon"]

    
    try:
        ts = tst.copy()
        
        extra_periods=horizon-1
        
        q = config["q"]
        d = config["d"]
        p = config["p"]
        
        if config["log"]:
            
            ts = np.log(ts+1)
            m = ARIMA(q, d, p)
            pred = m.fit_predict(ts)
            pred = m.forecast(ts, horizon)

            return (np.exp(pred[-horizon:])).round(0)
                        
                
        else:

            m = ARIMA(q, d, p)
            pred = m.fit_predict(ts)
            pred = m.forecast(ts, horizon)

            return np.round(pred[-horizon:],0)
    except:

        return np.zeros(horizon)


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
# REFACTOR
#
def forecast_arima2(xs, horiz, q=1, d=0, p=1, trim_zeros=True,
    local_model=False, use_log=False):
    """
    """
    import traceback 

    def preprocess_xs2(xs, local_model=False, trim_zeros=True):
        """
        """
        if len(xs) > 8 and trim_zeros:
            xs = np.trim_zeros(xs, trim="f")

        if local_model:
            xs = xs[-8:]

        return xs
    
    xs = preprocess_xs2(xs, local_model, trim_zeros) 

    try:
        extra_periods=horiz-1
        
        if use_log:
            xs = np.log(xs+1)
            m = ARIMA(q, d, p)
            pred = m.fit_predict(xs)
            pred = m.forecast(xs, horiz)
            yp = np.exp(pred[-horiz:]).round(0)
        else:
            m = ARIMA(q, d, p)
            pred = m.fit_predict(xs)
            pred = m.forecast(xs, horiz)
            yp = np.round(pred[-horiz:],0)
    except:
        traceback.print_exc()
        yp = np.zeros(horiz)

    yp = np.clip(yp, 0, None).round(0)

    assert(len(yp) == horiz)

    return yp
