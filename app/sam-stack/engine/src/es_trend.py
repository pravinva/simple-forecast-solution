# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import numpy as np


def forecast_es(tst, config):
            tmp = np.trim_zeros(tst, trim ='f')

            if len(tmp) > 8:
                tst = tmp

            if config["local_model"]:
                tst = tst[-8:]

            horizon = config["horizon"]

            try:    

                ts = tst.copy()
                
                extra_periods=horizon-1
                
                alpha = config["alpha"]
                
                if config["log"]:
                    
                    ts = np.log(ts+1)
                    f = []

                    f.append(ts[0])

                    # Create all the m+1 forecast
                    for t in range(1,len(ts)-1):
                        f.append((1-alpha)*f[-1]+alpha*ts[t])

                    # Forecast for all extra months
                    for t in range(extra_periods):
                        # Update the forecast as the last forecast
                        f.append(f[-1])    

                    #
                    return (np.exp(f[-horizon:])).round(0)
                                
                        
                else:

                    f = []

                    f.append(ts[0])

                    # Create all the m+1 forecast
                    for t in range(1,len(ts)-1):
                        f.append((1-alpha)*f[-1]+alpha*ts[t])

                    # Forecast for all extra months
                    for t in range(extra_periods):
                        # Update the forecast as the last forecast
                        f.append(f[-1])    

                    #
                    return np.round(f[-horizon:],0)

            except:
                return np.zeros(horizon)


def forecast_holt(tst, config):
            tmp = np.trim_zeros(tst, trim ='f')

            if len(tmp) > 8:
                tst = tmp

            if config["local_model"]:
                tst = tst[-8:]
    

            
            horizon = config["horizon"]

            try:
                extra_periods=horizon-1
                
                alpha = config["alpha"]
                beta = config["beta"]
                
                if config["log"]:
                    
                    ts = np.log(ts+1)
                    # Initialization
                    f = [np.nan] # First forecast is set to null value
                    a = [ts[0]] # First level defined as the first demand point
                    b = [ts[1]-ts[0]] # First trend is computed as the difference between the two first demand points

                    # Create all the m+1 forecast
                    for t in range(1,len(ts)):
                        # Update forecast based on last level (a) and trend (b)
                        f.append(a[t-1]+b[t-1])

                        # Update the level based on the new data point
                        a.append(alpha*ts[t]+(1-alpha)*(a[t-1]+b[t-1]))

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
                    return (np.exp(f[-horizon:])).round(0)
                                
                        
                else:

                    f = [np.nan] # First forecast is set to null value
                    a = [ts[0]] # First level defined as the first demand point
                    b = [ts[1]-ts[0]] # First trend is computed as the difference between the two first demand points

                    # Create all the m+1 forecast
                    for t in range(1,len(ts)):
                        # Update forecast based on last level (a) and trend (b)
                        f.append(a[t-1]+b[t-1])

                        # Update the level based on the new data point
                        a.append(alpha*ts[t]+(1-alpha)*(a[t-1]+b[t-1]))

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
                    #
                    return np.round(f[-horizon:],0)
            except:
                return np.zeros(horizon)
