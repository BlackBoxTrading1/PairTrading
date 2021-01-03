import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
import statsmodels.stats.diagnostic as sd
from scipy.stats import shapiro, pearsonr, linregress
import math
from pykalman import KalmanFilter

import copy
from itertools import groupby
from math import ceil
        
###################
# UNIVERSE PARAMS #
####################################################################################################
RUN_TEST_STOCKS     = False
TEST_STOCKS         = ['AAPL', 'MSFT', 'NFLX', 'AMZN', 'TSLA', 'FB', 'YELP', 'AMD', 'NVDA', 'EA', 'INTC', 'XRX']
COARSE_LIMIT        = 1000
FINE_LIMIT          = 200

##################
# TRADING PARAMS #
####################################################################################################
LEVERAGE               = 1.0
INTERVAL               = 1
DESIRED_PAIRS          = 10
HEDGE_LOOKBACK         = 21  #usually 15-300
ENTRY                  = 1.5 #usually 1.5
EXIT                   = 0.1 #usually 0.0
Z_STOP                 = 4.0 #usually >4.0
STOPLOSS               = 0.15
MIN_SHARE              = 1.00
MIN_WEIGHT             = 0.25
MAX_PAIR_WEIGHT        = 0.2
EQUAL_WEIGHTS          = False

##################
# TESTING PARAMS #
####################################################################################################
RANK_BY                   = 'Hurst' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = False
DESIRED_PVALUE            = 0.01
LOOKBACK                  = 253
LOOSE_PVALUE              = 0.05
BONFERRONI_TESTS              = ['Cointegration', 'ADFuller', 'Shapiro-Wilke']

TEST_PARAMS               = {
    'Correlation':  {'min': 0.8,   'max': 1.00,            'spreads': 0,  'run': 1 },
    'Cointegration':{'min': 0.0,   'max': DESIRED_PVALUE,  'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.0,   'max': 0.49,            'spreads': 1,  'run': 1 },
    'ADFuller':     {'min': 0.0,   'max': DESIRED_PVALUE,  'spreads': 1,  'run': 1 },
    'Half-life':    {'min': 1.0,   'max': HEDGE_LOOKBACK*2,'spreads': 1,  'run': 1 },
    'Shapiro-Wilke':{'min': 0.0,   'max': DESIRED_PVALUE,  'spreads': 1,  'run': 1 },
    'Zscore':       {'min': ENTRY, 'max': Z_STOP,          'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 0.0,   'max': np.inf,          'spreads': 0,  'run': 1 },
    'ADF-Prices':   {'min': LOOSE_PVALUE, 'max': 1.00,     'spreads': 0,  'run': 1 }
}
    
LOOSE_PARAMS              = {
    'Correlation':  {'min': 0.8,     'max': 1.00,            'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.0,     'max': LOOSE_PVALUE,    'spreads': 0,  'run': 0 },
    'ADFuller':     {'min': 0.0,     'max': LOOSE_PVALUE,    'spreads': 1,  'run': 0 },
    'Hurst':        {'min': 0.0,     'max': 0.49,            'spreads': 1,  'run': 0 },
    'Half-life':    {'min': 1.0,     'max': HEDGE_LOOKBACK*2,'spreads': 1,  'run': 0 },
    'Shapiro-Wilke':{'min': 0.0,     'max': LOOSE_PVALUE,    'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,     'max': Z_STOP,          'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 0.0,     'max': np.inf,          'spreads': 0,  'run': 1 },
    'ADF-Prices':   {'min': LOOSE_PVALUE, 'max': 1.00,       'spreads': 0,  'run': 0 }
}

class PairsTrader(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 11, 25)
        self.SetCash(10000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.initial_portfolio_value = self.Portfolio.TotalPortfolioValue
        self.curr_month = -1
        self.lastMonth = -1
        self.target_weights = {}
        self.industry_map = {}
        self.industries = []
        self.pairs = []
        self.weight_change = False
        self.dollarVolumeBySymbol = {}
        
        self.strict_tester = StatisticalTester(config=TEST_PARAMS)
        self.loose_tester = StatisticalTester(config=LOOSE_PARAMS)
        self.AddUniverse(self.SelectCoarse, self.SelectFine)
        self.Schedule.On(self.DateRules.MonthStart(self.spy), self.TimeRules.AfterMarketOpen(self.spy,5), Action(self.choose_pairs))
    
    def OnData(self, data):
        pass
    
    def choose_pairs(self):
        if not self.reset_vars():
            return
    
        self.industries = self.get_industries()
        if self.industries == []:
            self.Log("No substantial universe found. Waiting until next cycle")
            return

        # Store Price Histories #
        comps = 0
        count = 0
        report = "Calculating Price Histories..."
        for i in range(len(self.industries)):
            for stock in self.industries[i].stocks:
                price_history = self.History(self.Symbol(stock.ticker), TimeSpan.FromDays(365+2*30), Resolution.Daily).close
                stock.update_price_history(run_kalman(price_history.values.tolist()))
            self.industries[i].set_stocks([s for s in self.industries[i].stocks if (len(s.price_history) >= self.true_lookback)])
            report += ("\n\t\t\t" if i%3 == 0 else "\t") + str(self.industries[i])
            comps += self.industries[i].size * (self.industries[i].size - 1)
            count += self.industries[i].size
        self.Log(report + "\n\n\t\t\tTotal Tickers = {0}\n\t\t\tTotal Pairs = {1}".format(count, comps))
              
        # Price Test Pairs #
        self.Log("Price Testing Pairs...")
        all_pairs = {}
        failure_counts = {}
        counter = 0
        for c in range(len(self.industries)):
            industry = self.industries[c]
            all_pairs[industry.code] = []
            for i in range (industry.size):
                for j in range (i+1, industry.size):
                    pair_forward = Pair(industry.stocks[i], industry.stocks[j], industry)
                    pair_reverse = Pair(industry.stocks[j], industry.stocks[i], industry)
                    for pair in [pair_forward, pair_reverse]:
                        if self.strict_tester.test_pair_prices(pair):
                            counter += 1
                            spreads, _ = get_spreads(pair.left.price_history, pair.right.price_history, self.true_lookback-42)
                            pair.update_spreads(spreads)
                            all_pairs[industry.code].append(pair)
                        else:
                            failure_counts[pair.failed_test] = failure_counts.get(pair.failed_test, 0) + 1
        self.Log("Price test passed: {0}\n\t\t\t\tFailure Report: {1}".format(counter, failure_counts))
        
        # Spread Test Pairs #
        report = "CHOOSING {0} PAIRS".format(self.desired_pairs)
        final_pairs = []
        failure_counts = {}
        for code in all_pairs:
            industry = self.get_industry_by_code(code)
            for pair in all_pairs[code]:
                if self.strict_tester.test_pair_spreads(pair):
                    industry.add_good_pair(pair)
                else:
                    failure_counts[pair.failed_test] = failure_counts.get(pair.failed_test, 0) + 1
            final_pairs += industry.good_pairs
    
        final_pairs = sorted(final_pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
        num_pairs = min(len(final_pairs), self.desired_pairs)
        self.desired_pairs = self.desired_pairs - num_pairs
        report += " --> FOUND {0}\nFailure Report: {1}".format(num_pairs, failure_counts)
        for i in range(num_pairs):
            report += "\n\t\t\t{0}) {1}\n\t\t\t\tIndustry Code:\t {2}".format(len(self.pairs)+1, final_pairs[i], final_pairs[i].industry.code)
            for test in final_pairs[i].latest_test_results:
                report += ("\n\t\t\t\t" + str(test) + ": \t" + "\t"*(len(test) <= 5 ) 
                       + str(final_pairs[i].latest_test_results[test]))
            self.pairs.append(final_pairs[i])
        self.Log(report)
    
    def SelectCoarse(self, coarse):
        self.industry_map = {}
        if self.Time.month == self.lastMonth:
            return Universe.Unchanged

        sortedByDollarVolume = sorted([x for x in coarse if x.HasFundamentalData and x.Volume > 0 and x.Price > 0], key = lambda x: x.DollarVolume, reverse=True)[:COARSE_LIMIT]
        self.dollarVolumeBySymbol = {x.Symbol:x.DollarVolume for x in sortedByDollarVolume}
        if len(self.dollarVolumeBySymbol) == 0:
            return Universe.Unchanged

        return list(self.dollarVolumeBySymbol.keys())
        
    def SelectFine(self, fine):
        sortedBySector = sorted([x for x in fine if x.CompanyReference.CountryId == "USA"
                                        and x.CompanyReference.PrimaryExchangeID in ["NYS","NAS"]
                                        and (self.Time - x.SecurityReference.IPODate).days > 540
                                        and x.MarketCap > 5e8],
                               key = lambda x: x.CompanyReference.IndustryTemplateCode)

        count = len(sortedBySector)
        if count == 0:
            return Universe.Unchanged

        self.lastMonth = self.Time.month
        percent = 500 / count
        sortedByDollarVolume = []

        for code, g in groupby(sortedBySector, lambda x: x.CompanyReference.IndustryTemplateCode):
            y = sorted(g, key = lambda x: self.dollarVolumeBySymbol[x.Symbol], reverse = True)
            c = ceil(len(y) * percent)
            sortedByDollarVolume.extend(y[:c])

        sortedByDollarVolume = sorted(sortedByDollarVolume, key = lambda x: self.dollarVolumeBySymbol[x.Symbol], reverse=True)
        final_securities = sortedByDollarVolume[:FINE_LIMIT]
        for x in final_securities:
            if not (x.AssetClassification.MorningstarIndustryCode in self.industry_map):
                self.industry_map[x.AssetClassification.MorningstarIndustryCode] = []
            self.industry_map[x.AssetClassification.MorningstarIndustryCode].append(x.Symbol.Value)
        return [x.Symbol for x in final_securities]
        
    def reset_vars(self):
        this_month = self.Time.month 
        if self.curr_month < 0:
            self.curr_month = this_month
        self.next_month = self.curr_month + INTERVAL - 12*(self.curr_month + INTERVAL > 12)
        if (this_month != self.curr_month):
            return False
        self.curr_month = self.next_month
        self.Log("New interval start. Portfolio Value: {0}".format(self.Portfolio.TotalPortfolioValue))
        self.pairs = []
        self.industries = []
        self.delisted = []
        self.target_weights = {}
        self.Liquidate()
        
        spy_history = self.History(self.Symbol("SPY"), TimeSpan.FromDays(365 + 2*30), Resolution.Daily).close
        self.true_lookback = len(spy_history.values.tolist())
        return True
        
    def get_industries(self):
        industry_pool = []
        industries = []
        if RUN_TEST_STOCKS:
            industry = Industry(123456789, stocks=TEST_STOCKS)
            industries.append(industry)
        else:
            for code in self.industry_map:
                industry = Industry(code=code, stocks=self.industry_map[code])
                industries.append(industry)
            
        self.desired_pairs = int(round(DESIRED_PAIRS * (self.Portfolio.TotalPortfolioValue / self.initial_portfolio_value)))
    
        self.Log("Setting Universe...")
        for i in range(len(industries)):
            stock_list = []
            for ticker in industries[i].stocks:
                try:
                    equity = self.AddEquity(ticker)
                    stock = Stock(ticker, equity.Symbol.ID.ToString())
                    stock_list.append(stock)
                except:
                    pass
            industry_pool = industry_pool+stock_list
            industries[i].set_stocks(stock_list)

        temp = copy.deepcopy(industries)
        for i in range(len(temp)):
            if temp[i].size < 1:
                del industries[i]
                
        industries = sorted(industries, key=lambda x: x.size, reverse=False)
        for stock in industry_pool:
            self.target_weights[stock.id] = 0.0
    
        return industries
    
    def get_industry_by_code(self, code):
        for i in range(len(self.industries)):
            if self.industries[i].code == code:
                return self.industries[i]
        return None
        
#####################
# CLASS DEFINITIONS #
####################################################################################################
class Stock:
    def __init__(self, ticker, id):
        self.ticker = ticker
        self.id = id
        self.price_history = []
        self.purchase_price = {'price': 0, 'long': False}
        
    def __str__(self):
        return "{1}".format(self.id)
        
    def update_price_history(self, price_history):
        self.price_history = price_history

    def update_purchase_price(self, price, is_long):
        self.purchase_price['price'] = price
        self.purchase_price['long'] = is_long

    def test_stoploss(self, current_price):
        is_long = self.purchase_price['long']
        initial_price = self.purchase_price['price']
        if initial_price == 0:
            return True
        return not ((is_long and current_price < (1-STOPLOSS)*initial_price) or (not is_long and current_price > (1+STOPLOSS)*initial_price))

class Pair:
    def __init__(self, s1, s2, industry):
        self.left = s1
        self.right= s2
        self.industry = industry
        self.spreads = []
        self.unfiltered_spreads = []
        self.latest_test_results = {}
        self.failed_test = ""
        self.currently_long = False
        self.currently_short = False
    
    def __str__(self):
        return "([{0}] & [{1}])".format(self.left.ticker, self.right.ticker)
        
    def contains(self, id):
        return (self.left.id == id) or (self.right.id == id)
    
    def update_spreads(self, spreads):
        self.spreads = spreads

class Industry:
    def __init__(self, code, stocks=[]):
        self.code = code
        self.stocks = stocks
        self.good_pairs = []
        self.size = len(stocks)
    
    def __str__(self):
        return "Industry {0}: {1} tickers".format(self.code, self.size)
    
    def add_stock(self, stock):
        self.stocks.append(stock)
        self.size += 1
        
    def set_stocks(self, stocks):
        self.stocks = stocks
        self.size = len(stocks)
    
    def add_good_pair(self, pair):
        match = False
        for i in range(len(self.good_pairs)):
            if self.good_pairs[i].contains(pair.left.id) or self.good_pairs[i].contains(pair.right.id):
                match = True
                old_result = self.good_pairs[i].latest_test_results[RANK_BY]
                new_result = pair.latest_test_results[RANK_BY]
                if (RANK_DESCENDING and new_result > old_result) or ((not RANK_DESCENDING) and new_result < old_result):
                    self.good_pairs[i] = pair
        if (not match):
            self.good_pairs.append(pair)
        
class StatisticalTester:
    def __init__(self, config):
        self.tests = {
            "correlation": self.correlation, 
            "cointegration": self.cointegration,
            "adfuller": self.adf_pvalue,
            "hurst": self.hurst_hvalue,
            "half-life": self.half_life,
            "shapiro-wilke": self.shapiro_pvalue,
            "zscore": self.zscore,
            "alpha": self.alpha,
            "adf-prices": self.adf_prices
        }
        
        self.price_tests = [name for name in config if ((not config[name]['spreads']) and config[name]['run'])]
        self.spread_tests = [name for name in config if (config[name]['spreads'] and config[name]['run'])]
        self.config = config
        
    def test_pair_prices(self, pair):
        for test in self.price_tests:
            result = None
            test_function = self.tests[test.lower()]
            try:
                if test == "Alpha":
                    result = 1
                else:
                    result = test_function(pair.left.price_history[-HEDGE_LOOKBACK:], pair.right.price_history[-HEDGE_LOOKBACK:])
            except:
                pass
            
            pair.latest_test_results[test] = result
            if (not result) or (not self.test_value_bounds(test, result)):
                pair.failed_test = test
                return False
        return True
        
    def test_pair_spreads(self, pair):
        for test in self.spread_tests:
            result = None
            test_function = self.tests[test.lower()]
            if pair.spreads == []:
                pair.failed_test = "empty_spreads"
                return False
            try:
                if test == "Half-life":
                    result = test_function(pair.spreads[-HEDGE_LOOKBACK:])
                else:
                    result = test_function(pair.spreads)
            except:
                pass
            
            pair.latest_test_results[test] = result
            if (not result) or (not self.test_value_bounds(test, result)):
                pair.failed_test = test
                return False
        return True
    
    def test_value_bounds(self, test, result):
        upper_bound = self.config[test]['max']
        lower_bound = self.config[test]['min']
        if test in BONFERRONI_TESTS:
            upper_bound /= len(BONFERRONI_TESTS)
        return (result >= lower_bound and result <= upper_bound)

    def correlation(self, a,b):
        r, p = pearsonr(a,b)
        if p<DESIRED_PVALUE:
            return r
        else:
            return float('NaN')
        return r
    
    def cointegration(self, s1_price, s2_price):
        score, pvalue, _ = sm.coint(s1_price, s2_price)
        return pvalue
    
    def adf_pvalue(self, spreads):
        return sm.adfuller(spreads,autolag='t-stat')[1]
    
    def hurst_hvalue(self, series):
        max_window = len(series)-1
        min_window = 10
        window_sizes = list(map(lambda x: int(10**x),np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
        window_sizes.append(len(series))
        RS = []
        for w in window_sizes:
            rs = []
            for start in range(0, len(series), w):
                if (start+w)>len(series):
                    break

                incs = series[start:start+w][1:] - series[start:start+w][:-1]
                R = max(series[start:start+w]) - min(series[start:start+w])
                S = np.std(incs, ddof=1)

                if R != 0 and S != 0:
                    rs.append(R/S)
            RS.append(np.mean(rs))
        A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
        H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
        return H
    
    def half_life(self, spreads): 
        lag = np.roll(spreads, 1)
        ret = spreads - lag
        slope, intercept = linreg(lag,ret)
        return(-np.log(2) / slope)
    
    def shapiro_pvalue(self, spreads):
        w, p = shapiro(spreads)
        return p
    
    def adf_prices(self, s1_price, s2_price):
        p1 = sm.adfuller(s1_price, autolag='t-stat')[1]
        p2 = sm.adfuller(s2_price, autolag='t-stat')[1]
        return min(p1,p2)
    
    def zscore(self, spreads):
        return abs(spreads[-1])
    
    def alpha(self, price1, price2, stock1_price, stock2_price):
        slope, intercept = linreg(price2, price1)
        y_target_shares = 1
        X_target_shares = -slope
        (y_target_pct, x_target_pct) = calculate_target_pcts(y_target_shares, X_target_shares,stock1_price, stock2_price)
        if (min (abs(x_target_pct),abs(y_target_pct)) > MIN_WEIGHT):
            return slope
        return float('NaN')
    
######################
# AUXILARY FUNCTIONS #
####################################################################################################    
def run_kalman(price_history):
    kf_stock = KalmanFilter(transition_matrices = [1], observation_matrices = [1],
                            initial_state_mean = price_history[0], 
                            initial_state_covariance = 1, observation_covariance=1,
                            transition_covariance=.05)
    filtered_prices = kf_stock.filter(price_history)[0].flatten()
    return filtered_prices
    
def get_spreads(s1_price, s2_price, length):
    residuals = []
    zscores = []
    for i in range(1, HEDGE_LOOKBACK):
        start_index = len(s1_price) - length - HEDGE_LOOKBACK + i
        hedge, intercept = linreg(s2_price[start_index-HEDGE_LOOKBACK:start_index], 
                                  s1_price[start_index-HEDGE_LOOKBACK:start_index])
        residuals = np.append(residuals, s1_price[i] - hedge*s2_price[i] + intercept)
        
    for i in range(length):
        start_index = len(s1_price) - length + i
        hedge, intercept = linreg(s2_price[start_index-HEDGE_LOOKBACK:start_index], 
                                  s1_price[start_index-HEDGE_LOOKBACK:start_index])
        current_residual = s1_price[i] - hedge*s2_price[i] + intercept
        residuals = np.append(residuals, current_residual)
        std = np.std(residuals[-HEDGE_LOOKBACK:])
        zscores = np.append(zscores, current_residual/std)
    return zscores, residuals[-HEDGE_LOOKBACK:]

def linreg(s1,s2):
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(s1,s2)
    except:
        try:
            reg = np.polynomial.polynomial.polyfit(s1,s2)
            slope = reg[1]
            intercept = reg[0]
        except:
            try:
                s1 = sm.add_constant(s1)
                model = sm.OLS(s2, s1).fit()
                intercept = model.params[0]
                slope = model.params[1]
            except:
                slope = float('NaN')
                intercept = float('NaN')
    return slope, intercept