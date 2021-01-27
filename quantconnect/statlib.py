### <summary>
### Statistical Library Class
###
### Library containing testing and transformation functions.
### Used in Pair Trading Algorithm.
### </summary>

import numpy as np
import statsmodels.tsa.stattools as sm
import statsmodels.stats.diagnostic as sd
from scipy.stats import shapiro, pearsonr, linregress
from pykalman import KalmanFilter
import math

class StatsLibrary:
    
    def __init__(self, hedge_lookback, min_weight):
        self.hedge_lookback = hedge_lookback
        self.min_weight = min_weight

    def get_func_by_name(self, name):
        return getattr(self, name.lower())

    def correlation(self, series1, series2):
        r, p = pearsonr(series1, series2)
        if p < 0.01:
            return r
        else:
            return float('NaN')
        return r
    
    def cointegration(self, series1, series2):
        score, pvalue, _ = sm.coint(series1, series2)
        return pvalue
    
    def adfuller(self, series):
        return sm.adfuller(series,autolag='t-stat')[1]
    
    # def hurst(self,series):
    #     # Ernie
    #     tau, lagvec = [], []
    #     for lag in range(2,20):  
    #         pp = np.subtract(series[lag:],series[:-lag])
    #         lagvec.append(lag)
    #         tau.append(np.sqrt(np.std(pp)))
    #     slope, intercept = linreg(np.log10(lagvec),np.log10(tau))
    #     hurst = slope*2
    #     return hurst
    
    # def hurst(self, series):
    #     # Simplified
    #     max_window = len(series)-1
    #     min_window = 10
    #     window_sizes = list(map(lambda x: int(10**x),np.arange(math.log10(min_window), 
    #                         math.log10(max_window), 0.25)))
    #     window_sizes.append(len(series))
    #     RS = []
    #     for w in window_sizes:
    #         rs = []
    #         for start in range(0, len(series), w):
    #             if (start+w)>len(series):
    #                 break

    #             incs = series[start:start+w][1:] - series[start:start+w][:-1]
    #             R = max(series[start:start+w]) - min(series[start:start+w])
    #             S = np.std(incs, ddof=1)
                
    #             if R != 0 and S != 0:
    #                 rs.append(R/S)
    #         RS.append(np.mean(rs))
    #     A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    #     H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    #     return H
    
    def hurst(self,series):
        # Full
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
        return(-np.log(2) / slope)
    
    def shapirowilke(self, series):
        w, p = shapiro(series)
        return p
    
    def adfprices(self, series1, series2):
        p1 = sm.adfuller(series1, autolag='t-stat')[1]
        p2 = sm.adfuller(series2, autolag='t-stat')[1]
        return min(p1,p2)
    
    def zscore(self, series):
        return abs(series[-1])
    
    def alpha(self, series1, series2):
        slope, intercept = self.linreg(series2, series1)
        y_target_shares = 1
        X_target_shares = -slope
        notionalDol =  abs(y_target_shares * series1[-1]) + abs(X_target_shares * series2[-1])
        (y_target_pct, x_target_pct) = (y_target_shares * series1[-1] / notionalDol, X_target_shares * series2[-1] / notionalDol)
        if (min (abs(x_target_pct), abs(y_target_pct)) > 0.25):
            return slope
        return float('NaN')
    
    def run_kalman(self, series):
        kf_stock = KalmanFilter(transition_matrices = [1], observation_matrices = [1],
                                initial_state_mean = series[0], 
                                initial_state_covariance = 1, observation_covariance=1,
                                transition_covariance=.05)
        filtered_series = kf_stock.filter(series)[0].flatten()
        return filtered_series
    
    def get_spreads(self, series1, series2, length):
        residuals = []
        zscores = []
        for i in range(1, self.hedge_lookback):
            start_index = len(series1) - length - self.hedge_lookback + i
            hedge, intercept = self.linreg(series2[start_index-self.hedge_lookback:start_index], 
                                      series1[start_index-self.hedge_lookback:start_index])
            residuals = np.append(residuals, series1[i] - hedge*series2[i] + intercept)
            
        for i in range(length):
            start_index = len(series1) - length + i
            hedge, intercept = self.linreg(series2[start_index-self.hedge_lookback:start_index], 
                                      series1[start_index-self.hedge_lookback:start_index])
            current_residual = series1[i] - hedge*series2[i] + intercept
            residuals = np.append(residuals, current_residual)
            std = np.std(residuals[-self.hedge_lookback:])
            zscores = np.append(zscores, current_residual/std)
        return zscores, residuals[-self.hedge_lookback:]
    
    def linreg(self, series1, series2):
        slope, intercept, rvalue, pvalue, stderr = linregress(series1,series2)
        return slope, intercept