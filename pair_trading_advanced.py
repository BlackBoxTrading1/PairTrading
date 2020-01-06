#Pair Trading Algorithm

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline,CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage
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

COMMISSION             = 0
LEVERAGE               = 1.0
MAX_GROSS_EXPOSURE     = LEVERAGE
MARKET_CAP             = 0
INTERVAL               = 3
DESIRED_PAIRS          = 2
HEDGE_LOOKBACK         = 20 # used for regression
Z_WINDOW               = 20 # used for zscore calculation, must be <= HEDGE_LOOKBACK
ENTRY                  = 1.5
EXIT                   = 0.2
RECORD_LEVERAGE        = True

# Quantopian constraints
SET_PAIR_LIMIT         = True
SET_KALMAN_LIMIT       = True
MAX_PROCESSABLE_PAIRS  = 19000
MAX_KALMAN_STOCKS      = 275

REAL_UNIVERSE             = [
                               30947102, 31169147, 10428070, 10325059, 10321053, 10428068, 30951106,
                                31165133, 31052107, 10320050, 31061119, 31054109, 31165131, 20744096,
                                31166135, 31168144, 20635084, 10323057, 20636086, 20637087, 10320051,
                                20532078, 10322056, 10103004, 10217033, 10212027, 10104005, 10218039,
                                10211024, 10212026, 10106011, 10210023, 10216032, 10428069, 10209018,
                                10217037, 10212028, 10106010, 20744097, 20641092, 31167140, 10102002,
                                30845100, 20642093, 31058114, 31062125, 31062126, 30950105, 10428065
                            ]

# REAL_UNIVERSE             = [   
#                                 30910020, 31130020, 10420060, 10340020, 10330010, 10420040, 30910060,
#                                 31110020, 31010010, 10320020, 20645030, 20720020, 31120010, 30830010,
#                                 20610010, 10340060, 20620020, 20630010, 20540010, 10360010, 10130020,
#                                 10280010, 10240030, 30920010, 10290020, 10230010, 10240020, 10150030,
#                                 10270010, 10420050, 10200030, 10280060, 10220010, 10150040, 20720030,
#                                 20670010, 10120010, 30810010, 20650020, 31040010, 31080030, 31080040,
#                                 30910050, 10420010
#                             ]

# REAL_UNIVERSE             = [ 30910020, 31130020]

#Choose tests
RUN_CORRELATION_TEST      = False
RUN_COINTEGRATION_TEST    = True
RUN_ADFULLER_TEST         = True
RUN_HURST_TEST            = True
RUN_HALF_LIFE_TEST        = True
RUN_SHAPIROWILKE_TEST     = False

RUN_BONFERRONI_CORRECTION = True
RUN_KALMAN_FILTER         = True

#Ranking metric: select key from TEST_PARAMS
RANK_BY                   = 'hurst h-value'
DESIRED_PVALUE            = 0.01
PVALUE_TESTS              = ['Cointegration','ADFuller','Shapiro-Wilke']
TEST_PARAMS               = { #Used when choosing pairs
            'Correlation':      {'lookback': 365, 'min': 0.95, 'max': 1.00,           'key': 'correlation'  },
            'Cointegration':    {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'coint p-value'},
            'ADFuller':         {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'adf p-value'  },
            'Hurst':            {'lookback': 365, 'min': 0.00, 'max': 0.30,           'key': 'hurst h-value'},
            'Half-life':        {'lookback': 365, 'min': 1,    'max': 50,             'key': 'half-life'    },
            'Shapiro-Wilke':    {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'sw p-value'   }

                             }
LOOSE_PVALUE              = 0.10
LOOSE_PARAMS              = { #Used when checking pair quality
            'Correlation':      {'min': 0.95, 'max': 1.00,         'run': False},
            'Cointegration':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
            'ADFuller':         {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
            'Hurst':            {'min': 0.00, 'max': 0.49,         'run': True },
            'Half-life':        {'min': 0,    'max': 100,          'run': True },
            'Shapiro-Wilke':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': True }
                             }

def initialize(context):

    set_slippage(slippage.FixedBasisPointsSlippage())
    set_commission(commission.PerShare(cost=COMMISSION, min_trade_cost=0))
    set_benchmark(symbol('SPY'))

    context.num_universes = len(REAL_UNIVERSE)
    context.num_pvalue_tests = len(PVALUE_TESTS)
    context.initial_universes = {}

    context.initial_portfolio_value = context.portfolio.portfolio_value

    industry_code = ms.asset_classification.morningstar_industry_code.latest
    sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
    for code in REAL_UNIVERSE:
        pipe = Pipeline()
        pipe = algo.attach_pipeline(pipe, name = str(code))
        pipe.set_screen(QTradableStocksUS() 
                        & industry_code.eq(code) 
                        & (ms.valuation.market_cap.latest > MARKET_CAP) 
                        & (sma_short > 1.0))

    context.num_pairs = DESIRED_PAIRS
    context.universe_set = False
    context.pairs_chosen = False

    context.pair_status = {}
    context.universe_pool = []

    context.target_weights = {}

    context.curr_month = -1
    
    context.price_histories = {}
    context.curr_price_history = ()
    context.spreads = {}
    context.spread_lookbacks = []
    
    context.max_lookback = 0
    for test in TEST_PARAMS:
        if TEST_PARAMS[test]['lookback'] > context.max_lookback:
            context.max_lookback = TEST_PARAMS[test]['lookback']
    
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

    day = get_datetime().day
    print(("DAY # " + str(day) + " OF MONTH"))
    day = day - 2*day/7 - 3
    day = 0 if (day < 0 or day > 19) else day

    schedule_function(choose_pairs, date_rules.month_start(day), time_rules.market_open(hours=0, minutes=30))
    schedule_function(set_universe, date_rules.month_start(day), time_rules.market_open(hours=0, minutes=1))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))

def empty_target_weights(context):
    for s in list(context.target_weights.keys()):
        context.target_weights.loc[s] = 0.0
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)

def get_stock_partner(context, stock):
    partner = 0
    for pair in list(context.passing_pairs.keys()):
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

def get_allocated_stocks(context, target_weights):
    current_weights = []
    for k in list(target_weights.keys()):
        if target_weights.loc[k] != 0:
            partner = get_stock_partner(context, k)
            if not partner in target_weights:
                continue
            current_weights.append(k)
            if not partner in current_weights:
                current_weights.append(partner)
    return current_weights

def scale_stock_to_leverage(context, stock, pair_weight):
    partner = get_stock_partner(context, stock)
    stock_weight = context.target_weights[stock]
    partner_weight = context.target_weights[partner]
    total = abs(stock_weight) + abs(partner_weight)
    if total != LEVERAGE*pair_weight:
        context.target_weights[stock] = LEVERAGE * pair_weight * stock_weight / total
        context.target_weights[partner] = LEVERAGE * pair_weight * partner_weight / total

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
    lags = list(range(2, 100))
    for lag in lags:
        pdiff = np.subtract(ts[lag:],ts[:-lag])
        lagvec.append(lag)
        tau.append(np.sqrt(np.std(pdiff)))
    m = np.polynomial.polynomial.polyfit(np.log10(np.asarray(lagvec)), np.log10(np.asarray(tau).clip(min=0.0000000001)), 1)
    return m[1]*2.0
    
def get_shapiro_pvalue(spreads):
    w, p = shapiro(spreads)
    return p

def run_test(context, test, value, loose_screens):
    upper_bound = TEST_PARAMS[test]['max']
    if RUN_BONFERRONI_CORRECTION and test in PVALUE_TESTS:
        upper_bound /= context.num_pvalue_tests
    lower_bound = TEST_PARAMS[test]['min']
    if loose_screens:
        upper_bound = LOOSE_PARAMS[test]['max']
        lower_bound = LOOSE_PARAMS[test]['min']
    return (value != 'N/A' and value >= lower_bound and value <= upper_bound)

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
            if not run_test(context, 'Correlation', corr, loose_screens):
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
            if not run_test(context, 'Cointegration', coint, loose_screens):
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
            if not run_test(context, 'ADFuller', adf, loose_screens):
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
            if not run_test(context, 'Hurst', hurst, loose_screens):
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
            if not run_test(context, 'Half-life', hl, loose_screens):
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
            if not run_test(context, 'Shapiro-Wilke', sw, loose_screens):
                return False       
            
    return True
    
def set_universe(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.curr_month < 0:
        context.curr_month = this_month
    context.next_month = context.curr_month + INTERVAL - 12*(context.curr_month + INTERVAL > 12)
    if (this_month != context.curr_month):
        return
    context.curr_month = context.next_month
    
    context.num_pairs = DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)
    context.num_pairs = int(round(context.num_pairs))
    
    empty_target_weights(context)
    context.target_weights = get_current_portfolio_weights(context, data)
    
    context.universes = {}
    context.price_histories = {}
    size_str = ""
    total = 0
    for code in REAL_UNIVERSE:
        context.universes[code] = {}
        context.universes[code]['universe'] = algo.pipeline_output(str(code))
        context.universes[code]['universe'] = context.universes[code]['universe'].index
        context.universes[code]['size'] = len(context.universes[code]['universe'])
        if context.universes[code]['size'] > 1:
            context.universe_set = True
        size_str = size_str + " " + str(context.universes[code]['size'])
        total += context.universes[code]['size']
    
    diff = total-MAX_KALMAN_STOCKS
    kalman_overflow = (SET_KALMAN_LIMIT and diff > 0)
    context.remaining_codes = sorted(context.universes, key=lambda kv: context.universes[kv]['size'], reverse=False)
    
    sorted_codes = context.remaining_codes
    context.remaining_codes = []
    total = 0
    for code in sorted_codes:
        total += context.universes[code]['size']
    diff = total-MAX_KALMAN_STOCKS
    while (SET_KALMAN_LIMIT and diff > 0):
        diff = diff - context.universes[sorted_codes[0]]['size']
        del context.universes[sorted_codes[0]]
        context.remaining_codes.append(sorted_codes[0])
        sorted_codes.pop(0)
    context.codes = sorted_codes
    
    while (context.universes and context.universes[context.codes[0]]['size'] < 2):
        diff = diff - context.universes[context.codes[0]]['size']
        del context.universes[context.codes[0]]
        context.codes.pop(0)
        
    if (not context.universes):
        print("No substantial universe found. Waiting until next cycle")
        context.universe_set = False
        return
        
    context.codes = sorted(context.universes, key=lambda kv: context.universes[kv]['size'], reverse=True)
    
    updated_sizes_str = ""
    new_sizes = []
    for code in context.codes:
        if kalman_overflow:
            updated_sizes_str = updated_sizes_str + str(code) + " (" + str(context.universes[code]['size']) + ")  "
        new_sizes.append(context.universes[code]['size'])
    
    comps = 0
    for size in new_sizes:
        for i in range(size+1):
            comps += i
    comps = comps * 2
    valid_num_comps = (comps <= MAX_PROCESSABLE_PAIRS or (not SET_PAIR_LIMIT))
    
    print(("SETTING UNIVERSE " + " (running Kalman Filters)"*RUN_KALMAN_FILTER + 
           "...\nUniverse sizes:" + size_str + "\nTotal stocks: " + str(total)
           + (" > " + str(MAX_KALMAN_STOCKS) + " --> removing smallest universes" 
           + "\nUniverse sizes: " + str(updated_sizes_str))*(kalman_overflow)
           + "\nProcessed pairs: " + str(comps) + (" > " + str(MAX_PROCESSABLE_PAIRS)
           + " --> processing first " + str(MAX_PROCESSABLE_PAIRS) + " pairs") * (not valid_num_comps)))

    context.universe_pool = context.universes[context.codes[0]]['universe']
    for code in context.codes:
        context.universe_pool = context.universe_pool | context.universes[code]['universe']    

    for i in range(MAX_KALMAN_STOCKS+diff):
        price_history = get_price_history(data, context.universe_pool[i], context.max_lookback)
        if RUN_KALMAN_FILTER:
            kf_stock = KalmanFilter(transition_matrices = [1],
                                    observation_matrices = [1],
                                    initial_state_mean = price_history.values[0],
                                    initial_state_covariance = 1,
                                    observation_covariance=1,
                                    transition_covariance=.05)

            price_history,_ = kf_stock.filter(price_history.values)
            price_history = price_history.flatten()
        context.price_histories[context.universe_pool[i]] = price_history

def choose_pairs(context, data):
    if not context.universe_set:
        return
    
    context.universe_set = False
    print(("CHOOSING " + str(context.num_pairs) + " PAIRS"))
    context.test_data = {}
    context.passing_pairs = {}
    context.pairs = []
    
    pair_counter = 0
    for code in context.codes:
        for i in range (context.universes[code]['size']):
            for j in range (i+1, context.universes[code]['size']):
                s1 = context.universes[code]['universe'][i]
                s2 = context.universes[code]['universe'][j]
                s1_price = context.price_histories[s1]
                s2_price = context.price_histories[s2]

                if (SET_PAIR_LIMIT and pair_counter > MAX_PROCESSABLE_PAIRS):
                    break
                context.curr_price_history = (s1_price, s2_price)
                if passed_all_tests(context, data, s1, s2):
                    context.passing_pairs[(s1,s2)] = context.test_data[(s1,s2)]
                    context.passing_pairs[(s1,s2)]['code'] = code
                pair_counter += 1

                if (SET_PAIR_LIMIT and pair_counter > MAX_PROCESSABLE_PAIRS):
                    break
                context.curr_price_history = (s2_price, s1_price)
                if passed_all_tests(context, data, s2, s1):
                    context.passing_pairs[(s2,s1)] = context.test_data[(s2,s1)]
                    context.passing_pairs[(s2,s1)]['code'] = code
                pair_counter += 1

    #sort pairs from highest to lowest cointegrations
    rev = (RANK_BY == 'correlation')
    passing_pair_keys = sorted(context.passing_pairs, key=lambda kv: context.passing_pairs[kv][RANK_BY],
                                     reverse=rev)
    passing_pair_list = []
    for k in passing_pair_keys:
        passing_pair_list.append(k)

    total_code_list = []
    for pair in passing_pair_list:
        if (context.passing_pairs[pair]['code'] in total_code_list):
            passing_pair_keys.remove(pair)
            del context.passing_pairs[pair]
        else:
            total_code_list.append(context.passing_pairs[pair]['code'])          

    #select top num_pairs pairs
    if (context.num_pairs > len(passing_pair_keys)):
        context.num_pairs = len(passing_pair_keys)
    print(("Pairs found: " + str(context.num_pairs)))
    for i in range(context.num_pairs):
        context.pairs.append(passing_pair_keys[i])
        report = "TOP PAIR "+str(i+1)+": "+str(passing_pair_keys[i])
        for test in context.passing_pairs[passing_pair_keys[i]]:
            report += "\n\t\t\t" + str(test) + ": \t" + str(context.passing_pairs[passing_pair_keys[i]][test])
        print(report)
    context.pairs_chosen = True

    for pair in context.pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False

    context.spread = np.ndarray((context.num_pairs, 0))

def check_pair_status(context, data):
    if (not context.pairs_chosen):
        return
    
    

    new_spreads = np.ndarray((context.num_pairs, 1))

    for i in range(context.num_pairs):
        if (len(context.pairs) == 0):
            month = get_datetime('US/Eastern').month
            context.curr_month = month + 1 - 12*(month == 12)
            context.universe_set = False
            context.pairs_chosen = False
            break
        elif (i == len(context.pairs)):
            break
        pair = context.pairs[i]
        s1 = pair[0]
        s2 = pair[1]

        s1_price_test = get_price_history(data, s1, context.max_lookback)
        s2_price_test = get_price_history(data, s2, context.max_lookback)
        context.curr_price_history = (s1_price_test, s2_price_test)
        if not passed_all_tests(context, data, s1, s2, loose_screens=True):
            print("Closing " + str((s1,s2)) + ". Failed tests.")
            context.pairs.remove(pair)
            
            order_target_percent(s1, 0)
            order_target_percent(s2, 0)
            context.target_weights.loc[s1] = 0.0
            context.target_weights.loc[s2] = 0.0
            #context.target_weights = context.target_weights.drop([s1,s2])
            #context.universe_pool = context.universe_pool.drop([s1,s2])
            continue

        s1_price = data.history(s1, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::]
        if RUN_KALMAN_FILTER:
            kf_stock = KalmanFilter(transition_matrices = [1],
                                    observation_matrices = [1],
                                    initial_state_mean = s1_price.values[0],
                                    initial_state_covariance = 1,
                                    observation_covariance=1,
                                    transition_covariance=.05)

            price_history,_ = kf_stock.filter(s1_price.values)
            price_history = price_history.flatten()
            s1_price = price_history
        s2_price = data.history(s2, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::]
        if RUN_KALMAN_FILTER:
            kf_stock = KalmanFilter(transition_matrices = [1],
                                    observation_matrices = [1],
                                    initial_state_mean = s2_price.values[0],
                                    initial_state_covariance = 1,
                                    observation_covariance=1,
                                    transition_covariance=.05)

            price_history,_ = kf_stock.filter(s2_price.values)
            price_history = price_history.flatten()
            s2_price = price_history

        try:
            hedge = hedge_ratio(s1_price, s2_price, add_const=True)
        except ValueError as e:
            log.debug(e)
            return

        context.target_weights = get_current_portfolio_weights(context, data)
        for k in context.target_weights.keys():
            if not data.can_trade(k):
                context.target_weights = context.target_weights.drop([k])
        new_spreads[i, :] = s1_price[-1] - hedge * s2_price[-1]
        if context.spread.shape[1] > Z_WINDOW:

            spreads = context.spread[i, -Z_WINDOW:]
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()

            if context.pair_status[pair]['currently_short'] and zscore < EXIT:
                stocks = get_allocated_stocks(context, context.target_weights)
                n = float(len(stocks))
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        context.target_weights[stock] = context.target_weights[stock]*n/(n-2)
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        scale_stock_to_leverage(context, stock, pair_weight=2/(n-2))

                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False

                if not RECORD_LEVERAGE:
                    record(Y_pct=0, X_pct=0)
                allocate(context, data)
                return

            if context.pair_status[pair]['currently_long'] and zscore > -EXIT:
                stocks = get_allocated_stocks(context, context.target_weights)
                n = float(len(stocks))
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        context.target_weights[stock] = context.target_weights[stock]*n/(n-2)
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        scale_stock_to_leverage(context, stock, pair_weight=2/(n-2))

                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False

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

                stocks = get_allocated_stocks(context, context.target_weights)
                n = float(len(stocks))
                for stock in stocks:
                    context.target_weights[stock] = context.target_weights[stock]*n/(n+2)
                for stock in stocks:
                    scale_stock_to_leverage(context, stock, pair_weight=(2/(n+2)))

                context.target_weights[s2] = LEVERAGE * x_target_pct * (2/(n+2))
                context.target_weights[s1] = LEVERAGE * y_target_pct * (2/(n+2))

                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)

                allocate(context, data)
                return

            if zscore > ENTRY and (not context.pair_status[pair]['currently_short']):
                context.pair_status[pair]['currently_short'] = True
                context.pair_status[pair]['currently_long'] = False
                y_target_shares = -1
                X_target_shares = hedge
                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )

                stocks = get_allocated_stocks(context, context.target_weights)
                n = float(len(stocks))
                for stock in stocks:
                    context.target_weights[stock] = context.target_weights[stock]*n/(n+2)
                for stock in stocks:
                    scale_stock_to_leverage(context, stock, pair_weight=(2/(n+2)))

                context.target_weights[s2] = LEVERAGE * x_target_pct * (2/(n+2))
                context.target_weights[s1] = LEVERAGE * y_target_pct * (2/(n+2))

                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)

                allocate(context, data)
                return

    context.spread = np.hstack([context.spread, new_spreads])

def allocate(context, data):
    if RECORD_LEVERAGE:
        record(market_exposure=context.account.net_leverage, leverage=context.account.leverage)
    print ("ALLOCATING...")
    for s in list(context.target_weights.keys()):
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
            if partner:
                del context.passing_pairs[(s, partner)]
            if partner in context.target_weights:
                context.target_weights = context.target_weights.drop([partner])
                context.universe_pool = context.universe_pool.drop([partner])
                print(("--> Removed partner " + str(partner)))

    print ("Target weights:")
    for s in list(context.target_weights.keys()):
        if context.target_weights.loc[s] != 0:
            print(("\t" + str(s) + ":\t" + str(round(context.target_weights.loc[s],3))))
    # print(context.target_weights.keys())
    objective = opt.TargetWeights(context.target_weights)


    # Define constraints
    constraints = []
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )


def handle_data(context, data):
    pass
    #check_pair_status(context, data)
    # if context.account.leverage>LEVERAGE or context.account.leverage < 0:
    #     warn_leverage(context, data)