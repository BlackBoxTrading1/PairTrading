### <summary>
### Statistical Library Class
###
### Library containing testing and transformation functions.
### Used in Pair Trading Algorithm.
### </summary>

import numpy as np
import statsmodels.tsa.stattools as sm
from scipy.stats import shapiro, pearsonr, linregress
import scipy.stats as ss
from pykalman import KalmanFilter
from pandas import DataFrame as df
import math
from params import *

class StatsLibrary:
    
    def __init__(self, hedge_lookback, min_weight, downtick):
        self.hedge_lookback = hedge_lookback
        self.min_weight = min_weight
        self.downtick = downtick

    def get_func_by_name(self, name):
        return getattr(self, name.lower())

    def correlation(self, series1, series2):
        r, p = pearsonr(series1, series2)
        if p <= PVALUE:
            return r
        else:
            return float('NaN')
        return r
    
    def cointegration(self, series1, series2):
        return sm.coint(series1, series2, autolag='BIC', trend = 'ct')[1]
        
    def adfuller(self, series):
        return sm.adfuller(series,autolag='BIC')[1]
    
    def hurst(self,series):
        max_window = len(series)-1
        min_window = 10
        window_sizes = list(map(lambda x: int(10**x),np.arange(math.log10(min_window), 
                            math.log10(max_window), 0.25)))
        window_sizes.append(len(series))
        RS = []
        for w in window_sizes:
            rs = []
            for start in range(0, len(series), w):
                if (start+w)>len(series):
                    break

                incs = series[start:start+w][1:] - series[start:start+w][:-1]

                mean_inc = (series[start:start+w][-1] - series[start:start+w][0]) / len(incs)
                deviations = incs - mean_inc
                Z = np.cumsum(deviations)
                R = max(Z) - min(Z)
                S = np.std(incs, ddof=1)

                if R != 0 and S != 0:
                    rs.append(R/S)
            RS.append(np.mean(rs))
        A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
        H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
        return H
    
    def halflife(self, series): 
        lag = np.roll(series, 1)
        ret = series - lag
        slope, intercept = self.linreg(lag,ret)
        halflife = (-np.log(2) / slope)
        return halflife
    
    def shapirowilke(self, series):
        w, p = shapiro(series)
        return p
    
    def adfprices(self, series1, series2):
        p1 = sm.adfuller(series1, autolag='BIC')[1]
        p2 = sm.adfuller(series2, autolag='BIC')[1]
        return min(p1,p2)
    
    def ewa(self, series):
        current_residual = series[-1]
        std = np.std(series)
        spreads_df = df(series)
        spreads_ewm_df = df.ewm(spreads_df, span=HEDGE_LOOKBACK).mean()
        avg = list(spreads_ewm_df[0])[-1]
        zscore = (current_residual-avg)/std
        return zscore
    
    def calc_rsi(self, array, deltas, avg_gain, avg_loss, n ):
        up   = lambda x:  x if x > 0 else 0
        down = lambda x: -x if x < 0 else 0
        i = n+1
        for d in deltas[n+1:]:
            avg_gain = ((avg_gain * (n-1)) + up(d)) / n
            avg_loss = ((avg_loss * (n-1)) + down(d)) / n
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                array[i] = 100 - (100 / (1 + rs))
            else:
                array[i] = 100
            i += 1
        return array
    
    def get_rsi(self, array, n):   
        deltas = np.append([0],np.diff(array))
        avg_gain =  np.sum(deltas[1:n+1].clip(min=0)) / n
        avg_loss = -np.sum(deltas[1:n+1].clip(max=0)) / n
        array = np.empty(deltas.shape[0])
        array.fill(np.nan)
        array = self.calc_rsi( array, deltas, avg_gain, avg_loss, n )
        latest_rsi = array[-1]-50
        return latest_rsi
    
    def zscore(self, series):
        latest_residuals = series[-HEDGE_LOOKBACK:]
        if EWA:
            zscore = self.ewa(latest_residuals)
        else:
            zscore = ss.zscore(latest_residuals, nan_policy='omit')[-1]
        return abs(zscore)
    
    def alpha(self, series1, series2):
        slope, intercept = self.linreg(series2, series1)
        y_target_shares = 1
        X_target_shares = -slope
        notionalDol =  abs(y_target_shares * series1[-1]) + abs(X_target_shares * series2[-1])
        (y_target_pct, x_target_pct) = (y_target_shares * series1[-1] / notionalDol, X_target_shares * series2[-1] / notionalDol)
        if (min (abs(x_target_pct), abs(y_target_pct)) > MIN_WEIGHT):
            return slope
        return float('NaN')
    
    def run_kalman(self, series):
        # kf_stock = KalmanFilter(transition_matrices = [1], observation_matrices = [1],
        #                         initial_state_mean = series[0], 
        #                         observation_covariance=0.001,
        #                         transition_covariance=0.0001)
        # filtered_series = kf_stock.filter(series)[0].flatten()
        # return filtered_series
        return series
    
    def get_spreads(self, series1, series2, length):
        if SIMPLE_SPREADS:
            spreads = np.array(series1)/np.array(series2)
            return spreads

        residuals = []
        for i in range(length):
            start_index = len(series1) - length + i
            X = sm.add_constant(series2[(start_index-HEDGE_LOOKBACK):start_index])
            model = sm.OLS(series1[(start_index-HEDGE_LOOKBACK):start_index], X)
            results = model.fit()
            resid = results.resid[-1]
            residuals = np.append(residuals, resid)
        return residuals
        
    def linreg(self, series1, series2):
        slope, intercept, rvalue, pvalue, stderr = linregress(series1,series2)
        return slope, intercept