#Pair Trading Algorithm

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline,CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data import Fundamentals
import quantopian.pipeline.classifiers.morningstar
import quantopian.pipeline.data.morningstar as ms

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
import statsmodels.stats.diagnostic as sd
from scipy.stats import shapiro
import math
from pykalman import KalmanFilter

COMMISSION         = 0.005
LEVERAGE           = 1.0
MAX_GROSS_EXPOSURE = LEVERAGE
INTERVAL           = 5
DESIRED_PAIRS      = 2
HEDGE_LOOKBACK     = 20 # used for regression
Z_WINDOW           = 20 # used for zscore calculation, must be <= HEDGE_LOOKBACK
ENTRY              = 1.5
EXIT               = 0.2
RECORD_LEVERAGE    = True

SAMPLE_UNIVERSE    = [(symbol('STX'), symbol('WDC')),
                      (symbol('CBI'), symbol('JEC')),
                      (symbol('MAS'), symbol('VMC')),
                      (symbol('XOM'), symbol('CVX')),
                      (symbol('JPM'), symbol('C')),
                      (symbol('AON'), symbol('MMC')),
                      (symbol('COP'), symbol('CVX'))]

# REAL_UNIVERSE = [30947102, 31169147]
REAL_UNIVERSE = [10428070, 10320051, 10428069, 20744096, 31165131]
# REAL_UNIVERSE = [10320051]

# REAL_UNIVERSE = [10209016, 10209017, 10209018, 10209019, 10209020, 30946101, 30948103, 
#                  30949104, 30950105, 30951106, 10428064, 10428065, 10428066, 10428067, 10428068, 
#                  31167136, 31167137, 31167138, 31167139, 31167140, 31167141, 31167142, 31167143]

RUN_SAMPLE_PAIRS   = False
TEST_SAMPLE_PAIRS  = False

#Choose tests
RUN_KALMAN_FILTER         = False
RUN_CORRELATION_TEST      = False
RUN_COINTEGRATION_TEST    = True
RUN_ADFULLER_TEST         = True
RUN_HURST_TEST            = True
RUN_HALF_LIFE_TEST        = True
RUN_SHAPIROWILKE_TEST     = True
RUN_LJUNGBOX_TEST         = False

#Ranking metric: select key from TEST_PARAMS
RANK_BY         = 'hurst h-value'
DESIRED_PVALUE  = 0.05
TEST_PARAMS     = { #Used when choosing pairs
            'Correlation':      {'lookback': 730, 'min': 0.95, 'max': 1.00,           'key': 'correlation'  },
            'Cointegration':    {'lookback': 730, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'coint p-value'},
            'ADFuller':         {'lookback': 63,  'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'adf p-value'  },
            'Hurst':            {'lookback': 126, 'min': 0.00, 'max': 0.50,           'key': 'hurst h-value'},
            'Half-life':        {'lookback': 126, 'min': 15,   'max': 35,             'key': 'half-life'    },
            'Shapiro-Wilke':    {'lookback': 730, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'sw p-value'   },
            'Ljung-Box':        {'lookback': 730, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'lb-pvalue'    }

                  }
LOOSE_PVALUE    = 0.05
LOOSE_PARAMS    = { #Used when checking pair quality
            'Correlation':      {'min': 0.95, 'max': 1.00,         'run': False},
            'Cointegration':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': True },
            'ADFuller':         {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
            'Hurst':            {'min': 0.00, 'max': 0.50,         'run': False},
            'Half-life':        {'min': 0,    'max': 100,          'run': False},
            'Shapiro-Wilke':    {'min': 0.00, 'max': 1.00,         'run': False},
            'Ljung-Box':        {'min': 0.00, 'max': 1.00,         'run': False}
                  }

def initialize(context):

    set_slippage(slippage.FixedBasisPointsSlippage())
    set_commission(commission.PerShare(cost=COMMISSION, min_trade_cost=1))
    set_benchmark(symbol('SPY'))
    context.industry_code = ms.asset_classification.morningstar_industry_code.latest
    context.codes = REAL_UNIVERSE
    context.num_universes = len(context.codes)
    context.universes = {}

    context.initial_portfolio_value = context.portfolio.portfolio_value
    
    if not RUN_SAMPLE_PAIRS:
        for code in context.codes:
            context.universes[code] = {}
            context.universes[code]['pipe'] = Pipeline()
            context.universes[code]['pipe'] = algo.attach_pipeline(context.universes[code]['pipe'],
                                                          name = str(code))
            context.universes[code]['pipe'].set_screen(QTradableStocksUS() &
                                    context.industry_code.eq(code))

    context.num_pairs = DESIRED_PAIRS
    context.top_yield_pairs = []
    context.universe_set = False

    context.test_data = {}
    context.passing_pairs = {}
    context.pair_status = {}
    context.total_stock_list = []
    context.universe_pool = []

    context.target_weights = {}

    context.curr_month = -1
    
    context.curr_price_history = ()
    context.spreads = {}
    context.spread_lookbacks = []
    
    if (RANK_BY == 'coint' or RANK_BY == 'adf p-value' or RANK_BY == 'sw p-value'):
        log.warn("Ranking by p-value is undefined. Rank by different metric")
    
    if ((not RUN_ADFULLER_TEST and RANK_BY == 'adf p-value') 
        or (not RUN_HURST_TEST and RANK_BY == 'hurst h-value')
        or (not RUN_HALF_LIFE_TEST and RANK_BY == 'half-life') 
        or (not RUN_SHAPIROWILKE_TEST and RANK_BY == 'sw p-value')
        or (not RUN_CORRELATION_TEST and RANK_BY == 'correlation')
        or (not RUN_COINTEGRATION_TEST and RANK_BY == 'cointegration')):
        log.error("Ranking by untested metric... Cannot proceed")
        log.debug("1. Change value of RANK_BY to a tested metric")
        log.debug("2. Set the test of RANK_BY value to True")
        return

    if RUN_SAMPLE_PAIRS:
        schedule_function(sample_comparison_test, date_rules.month_start(), time_rules.market_open(hours=0,
                                                                                                   minutes=1))
    else:
        schedule_function(choose_pairs, date_rules.month_start(), time_rules.market_open(hours=0, minutes=1))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))

def empty_data(context):
    context.test_data = {}
    context.passing_pairs = {}
    context.top_yield_pairs = []
    context.total_stock_list = []

def empty_target_weights(context):
    for s in context.target_weights.keys():
        context.target_weights.loc[s] = 0.0
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)

def get_stock_partner(context, stock):
    partner = 0
    for pair in context.passing_pairs.keys():
        if stock == pair[0]:
            partner = pair[1]
        elif stock == pair[1]:
            partner = pair[0]
    return partner     

#calculate total commission cost of a stock given betsize
def get_commission(data, stock, bet_size):
    price = data.current(stock, 'price')
    num_shares = bet_size/price
    return (COMMISSION*num_shares)

def get_price_history(data, stock, length):
    return data.history(stock, "price", length, '1d')

def get_stored_prices(context, data, s1, s2, lookback):
    s1_price = context.curr_price_history[0][-lookback:]
    s2_price = context.curr_price_history[1][-lookback:]
    return s1_price, s2_price

def get_stored_spreads(context, data, s1_price, s2_price, lookback):
    spreads = 0
    if lookback in context.spread_lookbacks:
        spreads = context.spreads[lookback]
    else:
        spreads = get_spreads(data, s1_price, s2_price, lookback)
        context.spreads[lookback] = spreads
        context.spread_lookbacks.append(lookback)
    return spreads

def hedge_ratio(Y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit_regularized()
        return model.params[1]
    model = sm.OLS(Y, X).fit_regularized()
    return model.params.values 

def get_current_portfolio_weights(context, data):  
    positions = context.portfolio.positions  
    positions_index = pd.Index(positions)  
    share_counts = pd.Series(  
        index=positions_index,  
        data=[positions[asset].amount for asset in positions]  
    )

    current_prices = data.current(positions_index, 'price')
    current_weights = share_counts * current_prices / context.portfolio.portfolio_value
    #return current_weights.reindex(positions_index.union(context.universe), fill_value=0.0)
    return current_weights.reindex(positions_index.union(context.universe_pool), fill_value=0.0)

def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)  

def get_spreads(data, s1_price, s2_price, length):
    try:
        hedge = hedge_ratio(s1_price, s2_price, add_const=True)
    except ValueError as e:
        log.debug(e)
        return
    spreads = []
    for i in range(length):
        spreads = np.append(spreads, s1_price[i] - hedge*s2_price[i])
    return spreads

def get_cointegration(s1_price, s2_price):
    score, pvalue, _ = sm.coint(s1_price, s2_price)
    return pvalue

def get_adf_pvalue(spreads):
    return sm.adfuller(spreads,1)[1]

def get_half_life(spreads): 
    lag = np.roll(spreads, 1)
    lag[0] = 0
    ret = spreads - lag
    ret[0] = 0
    lag2 = sm.add_constant(lag)
    model = sm.OLS(ret, lag2)
    res = model.fit_regularized()
    return (-np.log(2) / res.params[1])

def get_hurst_hvalue(ts):
    ts = np.asarray(ts)
    lagvec = []
    tau = []
    lags = range(2, 100)
    for lag in lags:
        pdiff = np.subtract(ts[lag:],ts[:-lag])
        lagvec.append(lag)
        tau.append(np.sqrt(np.std(pdiff)))
    m = np.polynomial.polynomial.polyfit(np.log10(np.asarray(lagvec)), np.log10(np.asarray(tau).clip(min=0.0000000001)), 1)
    return m[1]*2.0
    
def get_shapiro_pvalue(spreads):
    w, p = shapiro(spreads)
    return p

def get_ljung_pvalue(spreads):
    count = 0
    for p in sd.acorr_ljungbox(spreads)[1]:
        if p > 0.05:
            count += 1
    return (count / 40.0)

def run_test(test, value, loose_screens):
    if loose_screens:  
        return (value != 'N/A' and value >= LOOSE_PARAMS[test]['min'] and value <= LOOSE_PARAMS[test]['max'])
    return (value != 'N/A' and value >= TEST_PARAMS[test]['min'] and value <= TEST_PARAMS[test]['max'])

def passed_all_tests(context, data, s1, s2, loose_screens=False):
    context.spreads = {}
    context.spread_lookbacks = []
    context.test_data[(s1,s2)] = {}
    
    if RUN_CORRELATION_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Correlation']['run']):
            lookback = TEST_PARAMS['Correlation']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            corr = 'N/A'
            try:
                corr = s1_price.corr(s2_price)
            except:
                corr = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['Correlation']['key']] = corr
            if not run_test('Correlation', corr, loose_screens) and (not RUN_SAMPLE_PAIRS
                                                  or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    if RUN_COINTEGRATION_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Cointegration']['run']):
            lookback = TEST_PARAMS['Cointegration']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            coint = 'N/A'
            try:
                coint = get_cointegration(s1_price,s2_price)
            except:
                coint = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['Cointegration']['key']] = coint
            if not run_test('Cointegration', coint, loose_screens) and (not RUN_SAMPLE_PAIRS 
                                                         or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False

    if RUN_ADFULLER_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['ADFuller']['run']):
            lookback = TEST_PARAMS['ADFuller']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            spreads = get_stored_spreads(context, data, s1_price, s2_price, lookback)
            adf = 'N/A'
            try:
                adf = get_adf_pvalue(spreads)
            except:
                adf = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['ADFuller']['key']] = adf
            if not run_test('ADFuller', adf, loose_screens) and (not RUN_SAMPLE_PAIRS 
                                                  or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    if RUN_HURST_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Hurst']['run']):
            lookback = TEST_PARAMS['Hurst']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            spreads = get_stored_spreads(context, data, s1_price, s2_price, lookback)
            hurst = 'N/A'
            try:
                hurst = get_hurst_hvalue(spreads)
            except:
                hurst = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['Hurst']['key']] = hurst
            if not run_test('Hurst', hurst, loose_screens) and (not RUN_SAMPLE_PAIRS
                                                 or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    if RUN_HALF_LIFE_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Half-life']['run']):
            lookback = TEST_PARAMS['Half-life']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            spreads = get_stored_spreads(context, data, s1_price, s2_price, lookback)
            hl = 'N/A'
            try:
                hl = get_half_life(spreads)
            except:
                hl = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['Half-life']['key']] = hl
            if not run_test('Half-life', hl, loose_screens) and (not RUN_SAMPLE_PAIRS 
                                                  or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    if RUN_SHAPIROWILKE_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Shapiro-Wilke']['run']):
            lookback = TEST_PARAMS['Shapiro-Wilke']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            spreads = get_stored_spreads(context, data, s1_price, s2_price, lookback)
            sw = 'N/A'
            try:
                sw = get_shapiro_pvalue(spreads)
            except:
                sw = 'N/A'
            context.test_data[(s1,s2)][TEST_PARAMS['Shapiro-Wilke']['key']] = sw
            if not run_test('Shapiro-Wilke', sw, loose_screens) and (not RUN_SAMPLE_PAIRS 
                                                      or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    if RUN_LJUNGBOX_TEST:
        if not loose_screens or (loose_screens and LOOSE_PARAMS['Ljung-Box']['run']):
            lookback = TEST_PARAMS['Ljung-Box']['lookback']
            s1_price, s2_price = get_stored_prices(context, data, s1, s2, lookback)
            spreads = get_stored_spreads(context, data, s1_price, s2_price, lookback)
            lb = 'N/A'
            try:
                lb = get_ljung_pvalue(spreads)
            except:
                lb = 'N'
            context.test_data[(s1,s2)][TEST_PARAMS['Ljung-Box']['key']] = lb
            if not run_test('Ljung-Box', lb, loose_screens) and (not RUN_SAMPLE_PAIRS
                                                      or (RUN_SAMPLE_PAIRS and TEST_SAMPLE_PAIRS)):
                return False
    return True

#*****************************************************************************************
def sample_comparison_test(context, data):
    this_month = get_datetime('US/Eastern').month
    if context.curr_month < 0:
        context.curr_month = this_month
    context.next_month = context.curr_month + INTERVAL - 12*(context.curr_month + INTERVAL > 12)
    if (this_month != context.curr_month):
        return
    context.curr_month = context.next_month

    context.num_pairs = DESIRED_PAIRS
    if (DESIRED_PAIRS > len(SAMPLE_UNIVERSE)):
        context.num_pairs = len(SAMPLE_UNIVERSE)

    empty_data(context)
    empty_target_weights(context)

    print ("RUNNING SAMPLE PAIRS...")
    context.universe_pool = pd.Index([])
    for pair in SAMPLE_UNIVERSE:
        context.universe_pool = context.universe_pool.append(pd.Index([pair[0], pair[1]]))
        if passed_all_tests(context, data, pair[0], pair[1]):
            context.passing_pairs[(pair[0],pair[1])] = context.test_data[(pair[0],pair[1])]
        if passed_all_tests(context, data, pair[1], pair[0]):
            context.passing_pairs[(pair[1],pair[0])] = context.test_data[(pair[1],pair[0])]

    context.target_weights = get_current_portfolio_weights(context, data)
    rev = (RANK_BY == 'corr')
    passing_pair_keys = sorted(context.passing_pairs, key=lambda kv: context.passing_pairs[kv][RANK_BY],
                                     reverse=rev)
    temp_real_yield_keys = passing_pair_keys
    if TEST_SAMPLE_PAIRS:
        for pair in temp_real_yield_keys:
            if (pair[0] in context.total_stock_list) or (pair[1] in context.total_stock_list):
                passing_pair_keys.remove(pair)
                del context.passing_pairs[pair]
            else:
                context.total_stock_list.append(pair[0])
                context.total_stock_list.append(pair[1])


    #select top num_pairs pairs
    if (context.num_pairs > len(passing_pair_keys)):
        context.num_pairs = len(passing_pair_keys)
    if not TEST_SAMPLE_PAIRS:
        context.num_pairs = len(SAMPLE_UNIVERSE)
        passing_pair_keys = SAMPLE_UNIVERSE
    for i in range(context.num_pairs):
        context.top_yield_pairs.append(passing_pair_keys[i])
        report = "TOP PAIR "+str(i+1)+": "+str(passing_pair_keys[i])
        for test in context.passing_pairs[passing_pair_keys[i]]:
            report += "\n\t\t\t" + str(test) + ": \t" + str(context.passing_pairs[passing_pair_keys[i]][test])
        print(report)

    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False
    context.universe_set = True
    context.spread = np.ndarray((context.num_pairs, 0))
#*************************************************************************************************************

def choose_pairs(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.curr_month < 0:
        context.curr_month = this_month
    context.next_month = context.curr_month + INTERVAL - 12*(context.curr_month + INTERVAL > 12)
    if (this_month != context.curr_month):
        return
    context.curr_month = context.next_month
    
    #context.num_pairs = DESIRED_PAIRS
    context.num_pairs = DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)
    context.num_pairs = int(round(context.num_pairs))
    
    empty_data(context)
    empty_target_weights(context)
    context.target_weights = get_current_portfolio_weights(context, data)
    
    size_str = ""
    usizes = []
    for code in context.codes:
        context.universes[code]['universe'] = algo.pipeline_output(str(code))
        context.universes[code]['universe'] = context.universes[code]['universe'].index
        context.universes[code]['size'] = len(context.universes[code]['universe'])
        if context.universes[code]['size'] > 1:
            context.universe_set = True
        size_str = size_str + " " + str(context.universes[code]['size'])
        usizes.append(context.universes[code]['size'])
    #usizes = [100]
    comps = 0
    total = 0
    for size in usizes:
      total += size
      for i in range (size+1):
        comps+=i
    comps = comps*2
    print ("CHOOSING " + str(context.num_pairs) +" PAIRS...\nUniverse sizes:" + size_str + "\nTotal stocks: " + str(total)
           + "\nProcessed pairs: " + str(comps))
    context.universe_pool = context.universes[context.codes[0]]['universe']
    for code in context.codes:
        context.universe_pool = context.universe_pool | context.universes[code]['universe']
    
    #context.spread = np.ndarray((context.num_pairs, 0))
    
    max_lookback = 0
    for test in TEST_PARAMS:
        if TEST_PARAMS[test]['lookback'] > max_lookback:
            max_lookback = TEST_PARAMS[test]['lookback']

    #SCREENING
    for code in context.codes:
        for i in range (context.universes[code]['size']):
            for j in range (i+1, context.universes[code]['size']):
                s1 = context.universes[code]['universe'][i]
                s2 = context.universes[code]['universe'][j]

                s1_price = get_price_history(data, s1, max_lookback)
                if RUN_KALMAN_FILTER:
                    kf_s1 = KalmanFilter(transition_matrices = [1],
                                      observation_matrices = [1],
                                      initial_state_mean = s1_price.values[0],
                                      initial_state_covariance = 1,
                                      observation_covariance=1,
                                      transition_covariance=.01)

                    s1_price,_ = kf_s1.filter(s1_price.values)
                    s1_price = s1_price.flatten()
                
                s2_price = get_price_history(data, s2, max_lookback)
                if RUN_KALMAN_FILTER:
                    kf_s2 = KalmanFilter(transition_matrices = [1],
                                      observation_matrices = [1],
                                      initial_state_mean = s2_price.values[0],
                                      initial_state_covariance = 1,
                                      observation_covariance=1,
                                      transition_covariance=.01)
                    s2_price,_ = kf_s2.filter(s2_price.values)
                    s2_price = s2_price.flatten()

                context.curr_price_history = (s1_price, s2_price)
                if passed_all_tests(context, data, s1, s2):
                    context.passing_pairs[(s1,s2)] = context.test_data[(s1,s2)]
                context.curr_price_history = (s2_price, s1_price)
                if passed_all_tests(context, data, s2, s1):
                    context.passing_pairs[(s2,s1)] = context.test_data[(s2,s1)]
    #sort pairs from highest to lowest cointegrations
    rev = (RANK_BY == 'correlation')
    passing_pair_keys = sorted(context.passing_pairs, key=lambda kv: context.passing_pairs[kv][RANK_BY],
                                     reverse=rev)
    temp_real_yield_keys = []
    for k in passing_pair_keys:
        temp_real_yield_keys.append(k)
    #print (temp_real_yield_keys)
    for pair in temp_real_yield_keys:
        if (pair[0] in context.total_stock_list) or (pair[1] in context.total_stock_list):
            passing_pair_keys.remove(pair)
            del context.passing_pairs[pair]
        else:
            context.total_stock_list.append(pair[0])
            context.total_stock_list.append(pair[1])

    #select top num_pairs pairs
    if (context.num_pairs > len(passing_pair_keys)):
        context.num_pairs = len(passing_pair_keys)
    print ("Pairs found: " + str(context.num_pairs))
    for i in range(context.num_pairs):
        context.top_yield_pairs.append(passing_pair_keys[i])
        u_code = 0
        for code in context.codes:
            if passing_pair_keys[i][0] in context.universes[code]['universe']:
                u_code = code
        report = "TOP PAIR "+str(i+1)+": "+str(passing_pair_keys[i])+"\n\t\t\tsector: \t"+str(u_code)
        for test in context.passing_pairs[passing_pair_keys[i]]:
            report += "\n\t\t\t" + str(test) + ": \t" + str(context.passing_pairs[passing_pair_keys[i]][test])
        print(report)

    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False

    context.num_remaining_pairs = context.num_pairs
    context.spread = np.ndarray((context.num_pairs, 0))

def check_pair_status(context, data):
    if (not context.universe_set):
        return
    
    new_spreads = np.ndarray((context.num_pairs, 1))
    temp_top_pairs = []
    for pair in context.top_yield_pairs:
        temp_top_pairs.append(pair)
    
    for i in range(context.num_pairs):
        if (len(temp_top_pairs) == 0):
            month = get_datetime('US/Eastern').month
            context.curr_month = month + 1 - 12*(month == 12)
            context.universe_set = False
            break
        elif (i == len(temp_top_pairs)):
            break
        pair = temp_top_pairs[i]
        # print pair
        s1 = pair[0]
        s2 = pair[1]
        
        max_lookback = 0
        for test in TEST_PARAMS:
            if TEST_PARAMS[test]['lookback'] > max_lookback:
                max_lookback = TEST_PARAMS[test]['lookback']
        
        s1_price_test = get_price_history(data, s1, max_lookback)
        s2_price_test = get_price_history(data, s2, max_lookback)
        context.curr_price_history = (s1_price_test, s2_price_test)
        if not passed_all_tests(context, data, s1, s2, loose_screens=True):
            summary = "Closing " + str((s1,s2)) + "\n\t\t\tSummary below:"
            for val in context.test_data[(s1,s2)].keys():
                end = ""
                for t in TEST_PARAMS:
                    if (TEST_PARAMS[t]['key'] == val):
                        if context.test_data[(s1,s2)][val] > LOOSE_PARAMS[t]['max']:
                            end = "> " + str(LOOSE_PARAMS[t]['max']) + " --> FAILED"
                        elif context.test_data[(s1,s2)][val] < LOOSE_PARAMS[t]['min']:
                            end = "< " + str(LOOSE_PARAMS[t]['min']) + " --> FAILED"
                summary += "\n\t\t\t" + str(val) + ": \t" + str(context.test_data[(s1,s2)][val]) + " " + end
            print (summary)
            context.top_yield_pairs.remove(pair)
            context.num_remaining_pairs = context.num_remaining_pairs - 1
            order_target_percent(s1, 0)
            order_target_percent(s2, 0)
            continue

        s1_price = data.history(s1, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::]
        s2_price = data.history(s2, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::]

        try:
            hedge = hedge_ratio(s1_price, s2_price, add_const=True)      
        except ValueError as e:
            log.debug(e)
            return

        context.target_weights = get_current_portfolio_weights(context, data)
        new_spreads[i, :] = s1_price[-1] - hedge * s2_price[-1]
        if context.spread.shape[1] > Z_WINDOW:

            spreads = context.spread[i, -Z_WINDOW:]
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()

            if context.pair_status[pair]['currently_short'] and zscore < EXIT:
                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                #set_pair_status(context, data, s1,s2,s1_price,s2_price, 0, 0, False, False)
                if not RECORD_LEVERAGE:
                    record(Y_pct=0, X_pct=0)
                allocate(context, data)
                return

            if context.pair_status[pair]['currently_long'] and zscore > -EXIT:
                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                #set_pair_status(context, data, s1,s2,s1_price,s2_price, 0, 0, False, False)
                if not RECORD_LEVERAGE:
                    record(Y_pct=0, X_pct=0)
                allocate(context, data)
                return

            if zscore < -ENTRY and (not context.pair_status[pair]['currently_long']):
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = True
                y_target_shares = 1
                X_target_shares = -hedge
                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )

                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_remaining_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_remaining_pairs)

                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)
                #set_pair_status(context,s1,s2,s1_price,s2_price, 1, -hedge, True, False)
                allocate(context, data)
                return

            if zscore > ENTRY and (not context.pair_status[pair]['currently_short']):
                context.pair_status[pair]['currently_short'] = True
                context.pair_status[pair]['currently_long'] = False
                y_target_shares = -1
                X_target_shares = hedge
                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )

                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_remaining_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_remaining_pairs)

                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)
                #set_pair_status(context,s1,s2,s1_price,s2_price, -1, hedge, False, True)
                allocate(context, data)
                return

    context.spread = np.hstack([context.spread, new_spreads])

def allocate(context, data):
    if RECORD_LEVERAGE:
        record(market_exposure=context.account.net_leverage, leverage=context.account.leverage)
    print ("ALLOCATING...")
    for s in context.target_weights.keys():
        error = ""
        if not (s in context.target_weights):
            continue
        elif not (data.can_trade(s)):
            error = "Cannot trade " + str(s)
        elif(np.isnan(context.target_weights.loc[s])):
            error = "Invalid target weight " + str(s)
        if error:
            print(error)
            partner = get_stock_partner(context, s)
            context.target_weights = context.target_weights.drop([s])
            context.universe_pool = context.universe_pool.drop([s])
            if partner in context.target_weights:
                context.target_weights = context.target_weights.drop([partner])
                context.universe_pool = context.universe_pool.drop([partner])
                print("--> Removed partner " + str(partner))

    print ("Target weights:")
    for s in context.target_weights.keys():
        if context.target_weights.loc[s] != 0:
            print ("\t" + str(s) + ":\t" + str(round(context.target_weights.loc[s],3)))
    # print(context.target_weights.keys())
    objective = opt.TargetWeights(context.target_weights)


    # Define constraints
    constraints = []
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    #print(context.target_weights)
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )


def handle_data(context, data):
    pass
    #check_pair_status(context, data)
    # if context.account.leverage>LEVERAGE or context.account.leverage < 0:
    #     warn_leverage(context, data)