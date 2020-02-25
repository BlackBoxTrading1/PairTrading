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
MARKET_CAP             = 1000 #millions
INTERVAL               = 3
DESIRED_PAIRS          = 2
HEDGE_LOOKBACK         = 20 # used for regression
Z_WINDOW               = 20 # used for zscore calculation, must be <= HEDGE_LOOKBACK
ENTRY                  = 2.0
EXIT                   = 0.5
RECORD_LEVERAGE        = True
STOPLOSS               = 0.20

# Quantopian constraints
SET_PAIR_LIMIT         = True
SET_KALMAN_LIMIT       = True
MAX_PROCESSABLE_PAIRS  = 19000
MAX_KALMAN_STOCKS      = 150

# REAL_UNIVERSE             = [
#                                30947102, 31169147, 10428070, 10325059, 10321053, 10428068, 30951106,
#                                 31165133, 31052107, 10320050, 31061119, 31054109, 31165131, 20744096,
#                                 31166135, 31168144, 20635084, 10323057, 20636086, 20637087, 10320051,
#                                 20532078, 10322056, 10103004, 10217033, 10212027, 10104005, 10218039,
#                                 10211024, 10212026, 10106011, 10210023, 10216032, 10428069, 10209018,
#                                 10217037, 10212028, 10106010, 20744097, 20641092, 31167140, 10102002,
#                                 30845100, 20642093, 31058114, 31062125, 31062126, 30950105, 10428065
#                             ]

REAL_UNIVERSE = [

    # 10110010, 10120010, 10130010, 10130020, 10140010, 10140020, 10150010, 10150020, 10150030,
    # 10150040, 10150050, 10150060, 10160010, 10160020, 10200010, 10200020, 10200030, 10200040,
    # 10220010, 10230010, 10240010, 10240020, 10240030, 10250010, 10260010, 10270010, 10280010,
    # 10280020, 10280030, 10280040, 10280050, 10280060, 10290010, 10290020, 10290030, 10290040,
    # 10290050, 10310010, 10320010, 10320020, 10320030, 10330010, 10330020, 10340010, 10340020,
    # 10340030, 10340040, 10340050, 10340060, 10350010, 10350020, 10360010, 10410010, 10410020,
    # 10410030, 10420010, 10420020, 10420030, 10420040, 10420050, 10420060, 10420070, 10420080,
    # 10420090, 20510010, 20510020, 20520010, 20525010, 20525020, 20525030, 20525040, 20540010,
    # 20550010, 20550020, 20550030, 20560010, 20610010, 20620010, 20620020, 20630010, 20645010,
    # 20645020, 20645030, 20650010, 20650020, 20660010, 20670010, 20710010, 20710020, 20720010,
    # 20720020, 20720030, 20720040, 30810010, 30820010, 30820020, 30820030, 30820040, 30830010,
    # 30830020, 30910010, 30910020, 30910030, 30910040, 30910050, 30910060, 30920010, 30920020,
    # 31010010, 31020010, 31020020, 31020030, 31020040, 31020050, 31030010, 31040010, 31040020, 
    # 31040030, 31050010, 31060010, 31070010, 31070020, 31070030, 31070040, 31070050, 31070060,
    # 31080010, 31080020, 31080030, 31080040, 31080050, 31080060, 31090010, 31110010, 31110020,
    # 31110030, 31120010, 31120020, 31120030, 31120040, 31120050, 31120060, 31130010, 31130020,
    # 31130030,

    10101001, 10102002, 10103003, 10103004, 10104005, 10105006, 10105007, 10106008, 10106009,
    10106010, 10106011, 10106012, 10107013, 10208014, 10208015, 10209016, 10209017, 10209018,
    10209019, 10209020, 10210021, 10210022, 10210023, 10211024, 10211025, 10212026, 10212027,
    10212028, 10213029, 10214030, 10215031, 10216032, 10217033, 10217034, 10217035, 10217036,
    10217037, 10218038, 10218039, 10218040, 10218041, 10319042, 10320043, 10320044, 10320045,
    10320046, 10320047, 10320048, 10320049, 10320050, 10320051, 10320052, 10321053, 10321054,
    10321055, 10322056, 10323057, 10324058, 10325059, 10326060, 10326061, 10427062, 10427063,
    10428064, 10428065, 10428066, 10428067, 10428068, 10428069, 10428070, 20529071, 20529072,
    20530073, 20531074, 20531075, 20531076, 20531077, 20532078, 20533079, 20533080, 20533081,
    20533082, 20534083, 20635084, 20636085, 20636086, 20637087, 20638088, 20638089, 20639090,
    20640091, 20641092, 20642093, 20743094, 20744095, 20744096, 20744097, 20744098, 30845099,
    30845100, 30946101, 30947102, 30948103, 30949104, 30950105, 30951106, 31052107, 31053108,
    31054109, 31055110, 31056111, 31056112, 31057113, 31058114, 31058115, 31059116, 31060117,
    31061118, 31061119, 31061120, 31061121, 31061122, 31062123, 31062124, 31062125, 31062126,
    31062127, 31063128, 31064129, 31165130, 31165131, 31165132, 31165133, 31165134, 31166135,
    31167136, 31167137, 31167138, 31167139, 31167140, 31167141, 31167142, 31167143, 31168144,
    31169145, 31169146, 31169147

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

# REAL_UNIVERSE             = [ 30947102, 31169147, 31167140]

#Choose tests
RUN_CORRELATION_TEST      = True
RUN_COINTEGRATION_TEST    = True
RUN_ADFULLER_TEST         = True
RUN_HURST_TEST            = True
RUN_HALF_LIFE_TEST        = True
RUN_SHAPIROWILKE_TEST     = True

RUN_BONFERRONI_CORRECTION = True
RUN_KALMAN_FILTER         = True

#Ranking metric: select key from TEST_PARAMS
RANK_BY                   = 'hurst h-value'
DESIRED_PVALUE            = 0.01
PVALUE_TESTS              = ['Cointegration','ADFuller','Shapiro-Wilke']
TEST_PARAMS               = { #Used when choosing pairs
            'Correlation':      {'lookback': 365, 'min': 0.90, 'max': 1.00,           'key': 'correlation'  },
            'Cointegration':    {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'coint p-value'},
            'ADFuller':         {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'adf p-value'  },
            'Hurst':            {'lookback': 365, 'min': 0.00, 'max': 0.30,           'key': 'hurst h-value'},
            'Half-life':        {'lookback': 365, 'min': 0,    'max': 50,             'key': 'half-life'    },
            'Shapiro-Wilke':    {'lookback': 365, 'min': 0.00, 'max': DESIRED_PVALUE, 'key': 'sw p-value'   }

                             }
LOOSE_PVALUE              = 0.05
LOOSE_PARAMS              = { #Used when checking pair quality
            'Correlation':      {'min': 0.00, 'max': 1.00,         'run': False},
            'Cointegration':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
            'ADFuller':         {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
            'Hurst':            {'min': 0.00, 'max': 0.40,         'run': True },
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


    # my_pipe = make_pipeline()
    # algo.attach_pipeline(my_pipe, 'my_pipeline')
    
    context.num_pipes = (int)(len(REAL_UNIVERSE)/50) + (len(REAL_UNIVERSE)%50 > 0)*1
    for i in range(context.num_pipes):
        pipe = make_pipeline(50*i, 50*i+50)
        algo.attach_pipeline(pipe, "pipe" + str(i))
    

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
    day = day - (int)(2*day/7) - 3
    day = 0 if (day < 0 or day > 19) else day

    schedule_function(choose_pairs, date_rules.month_start(day), time_rules.market_open(hours=0, minutes=30))
    schedule_function(set_universe, date_rules.month_start(day), time_rules.market_open(hours=0, minutes=1))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))

def make_pipeline(start, end):

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()
    industry_code = ms.asset_classification.morningstar_industry_code.latest
    sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30, mask=base_universe)

    columns = {}
    securities = (ms.valuation.market_cap.latest < 0 )

    # for universe in REAL_UNIVERSE:
    #     columns[str(universe)] = (sma_short<15) & industry_code.eq(universe) & (ms.valuation.market_cap.latest<1000000000)
    #     securities = securities | columns[str(universe)]
    
    for i in range(start, end):
        if (i >= len(REAL_UNIVERSE)):
            continue
        universe = REAL_UNIVERSE[i]
        columns[str(universe)] = (sma_short>5) & industry_code.eq(universe) & (ms.valuation.market_cap.latest>MARKET_CAP*(10**6))
        securities = securities | columns[str(universe)]

    return Pipeline(
        columns = columns,
        screen=(securities),
    )

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
            if not k in current_weights:
                current_weights.append(k)
            if not partner in current_weights:
                current_weights.append(partner)
    return current_weights

def scale_stock_to_leverage(context, stock, pair_weight):
    partner = get_stock_partner(context, stock)
    stock_weight = 0
    if stock in context.target_weights.keys():
        stock_weight = context.target_weights.loc[stock]
    partner_weight = 0
    if partner in context.target_weights.keys():
        partner_weight = context.target_weights.loc[partner]
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
                corr = np.corrcoef(s1_price, s2_price)[0][1]

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


def calculate_price_histories(context, data):
    sorted_codes = context.remaining_codes
    if not context.remaining_codes:
        context.desired_pairs = 0
        return
    context.remaining_codes = []
    total = 0
    size_str = ""
    for code in sorted_codes:
        total += context.universes[code]['size']
        size_str = size_str + " " + str(context.universes[code]['size'])
    diff = total-MAX_KALMAN_STOCKS
    kalman_overflow = (SET_KALMAN_LIMIT and diff > 0)
    while (SET_KALMAN_LIMIT and diff > 0):
        diff = diff - context.universes[sorted_codes[0]]['size']
        #del context.universes[sorted_codes[0]]
        context.remaining_codes.append(sorted_codes[0])
        sorted_codes.pop(0)
    context.codes = sorted_codes

    #context.codes = sorted(context.universes, key=lambda kv: context.universes[kv]['size'], reverse=True)
    context.codes.reverse()

    updated_sizes_str = ""
    new_sizes = []
    for code in context.codes:
        if kalman_overflow:
            updated_sizes_str = updated_sizes_str + str(code) + " (" + str(context.universes[code]['size']) + ")  "
        new_sizes.append(context.universes[code]['size'])

    comps = 0
    for size in new_sizes:
        comps += size*(size+1)

    valid_num_comps = (comps <= MAX_PROCESSABLE_PAIRS or (not SET_PAIR_LIMIT))

    print(("SETTING UNIVERSE " + " (running Kalman Filters)"*RUN_KALMAN_FILTER + 
           "...\nUniverse sizes:" + size_str + "\nTotal stocks: " + str(total)
           + (" > " + str(MAX_KALMAN_STOCKS) + " --> removing smallest universes" 
           + "\nUniverse sizes: " + str(updated_sizes_str))*(kalman_overflow)
           + "\nProcessed pairs: " + str(comps) + (" > " + str(MAX_PROCESSABLE_PAIRS)
           + " --> processing first " + str(MAX_PROCESSABLE_PAIRS) + " pairs") * (not valid_num_comps)))

    # context.universe_pool = context.universes[context.codes[0]]['universe']
    context.universe_pool = []
    for code in context.codes:
        # context.universe_pool = context.universe_pool | context.universes[code]['universe']
        context.universe_pool = context.universe_pool + context.universes[code]['universe']

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
    context.universe_set = True ##########################REVISIT##############################

def set_universe(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.curr_month < 0:
        context.curr_month = this_month
    context.next_month = context.curr_month + INTERVAL - 12*(context.curr_month + INTERVAL > 12)
    if (this_month != context.curr_month):
        return
    context.curr_month = context.next_month

    context.pairs_chosen = False
    context.desired_pairs = DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)
    context.desired_pairs = int(round(context.desired_pairs))

    context.passing_pairs = {}
    context.pairs = []

    context.purchase_prices = {}

    empty_target_weights(context)
    context.target_weights = get_current_portfolio_weights(context, data)

    context.universes = {}
    context.price_histories = {}
    #pipe_output = algo.pipeline_output('my_pipeline')
    
    pipe_output = algo.pipeline_output('pipe0')
    for i in range(1, context.num_pipes):
        pipe_output = pipe_output.append(algo.pipeline_output("pipe"+str(i)))
    pipe_output = pipe_output.fillna(False)
    
    total = 0
    for code in REAL_UNIVERSE:
        context.universes[code] = {}
        context.universes[code]['universe'] = pipe_output[pipe_output[str(code)]].index.tolist()
        context.universes[code]['size'] = len(context.universes[code]['universe'])
        if context.universes[code]['size'] > 1:
            context.universe_set = True
        else:
            del context.universes[code]
            continue
        total += context.universes[code]['size']

    if (not context.universes):
        print("No substantial universe found. Waiting until next cycle")
        context.universe_set = False
        return

    context.remaining_codes = sorted(context.universes, key=lambda kv: context.universes[kv]['size'], reverse=False)

    calculate_price_histories(context, data)

def choose_pairs(context, data):
    if not context.universe_set:
        return

    context.universe_set = False
    print(("CHOOSING " + str(context.desired_pairs) + " PAIRS"))
    context.test_data = {}

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
        if pair in context.pairs:
            passing_pair_keys.remove(pair)
            continue

        if (context.passing_pairs[pair]['code'] in total_code_list):
            passing_pair_keys.remove(pair)
            del context.passing_pairs[pair]
        else:
            total_code_list.append(context.passing_pairs[pair]['code'])          

    #select top num_pairs pairs
    context.num_pairs = len(passing_pair_keys)
    if (context.num_pairs > context.desired_pairs):
        context.num_pairs = context.desired_pairs
    context.desired_pairs = context.desired_pairs - context.num_pairs

    print(("Pairs found: " + str(context.num_pairs)))
    for i in range(context.num_pairs):
        context.pairs.append(passing_pair_keys[i])
        report = "TOP PAIR "+str(i+1)+": "+str(passing_pair_keys[i])
        for test in context.passing_pairs[passing_pair_keys[i]]:
            report += "\n\t\t\t" + str(test) + ": \t" + str(context.passing_pairs[passing_pair_keys[i]][test])
        print(report)
        context.purchase_prices[passing_pair_keys[i][0]] = {'price': 0, 'long': False}
        context.purchase_prices[passing_pair_keys[i][1]] = {'price': 0, 'long': False}
    context.pairs_chosen = True
    context.num_pairs = len(context.pairs)

    for pair in context.pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False

    context.spread = np.ndarray((context.num_pairs, 0))

def check_pair_status(context, data):
    if (not context.pairs_chosen):
        return

    if (context.desired_pairs != 0):
        calculate_price_histories(context, data)
        choose_pairs(context, data)
        return

    new_spreads = np.ndarray((context.num_pairs, 1))
    
    pairs_to_dump = []
    for stock in context.purchase_prices.keys():
        initial_price = context.purchase_prices[stock]['price']
        is_long = context.purchase_prices[stock]['long']
        current_price = data.current(stock, 'price')
        if (initial_price == 0):
            continue
        if ((is_long and current_price< (1-STOPLOSS)*initial_price) or (not is_long and current_price> (1+STOPLOSS)*initial_price)):
            partner = get_stock_partner(context, stock)
            print ("Dumping " + str(stock) + ". Purchase price: " + str(initial_price) + ", Current price: " + str(current_price))
            order_target_percent(stock, 0)
            order_target_percent(partner, 0)
            if (stock in context.target_weights.keys()):
                context.target_weights.loc[stock] = 0.0
            if (partner in context.target_weights.keys()):
                context.target_weights.loc[partner] = 0.0

            pair = (stock, partner)
            if not (pair in context.pairs):
                pair = (partner, stock)
            if not (pair in pairs_to_dump):
                pairs_to_dump.append(pair)
            
    for pair in pairs_to_dump:
        del context.purchase_prices[pair[0]]
        del context.purchase_prices[pair[1]]
        context.pairs.remove(pair)

    for i in range(context.num_pairs):
        if (len(context.pairs) == 0):
            month = get_datetime('US/Eastern').month
            context.curr_month = month + 1 - 12*(month == 12)
            context.universe_set = False
            context.pairs_chosen = False
            break
        elif (i >= len(context.pairs)):
            break
        pair = context.pairs[i]
        s1 = pair[0]
        s2 = pair[1]

        s1_price_test = get_price_history(data, s1, context.max_lookback)
        s2_price_test = get_price_history(data, s2, context.max_lookback)
        context.curr_price_history = (s1_price_test, s2_price_test)
        if not passed_all_tests(context, data, s1, s2, loose_screens=True):
            print("Closing " + str((s1,s2)) + ". Failed tests.")
            del context.purchase_prices[pair[0]]
            del context.purchase_prices[pair[1]]
            context.pairs.remove(pair)
            # context.num_pairs = context.num_pairs - 1

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
                    if stock != s1 and stock != s2 and stock in context.target_weights.keys():
                        context.target_weights[stock] = context.target_weights.loc[stock]*n/(n-2)
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        scale_stock_to_leverage(context, stock, pair_weight=2/(n-2))

                context.purchase_prices[s1]['price'] = 0
                context.purchase_prices[s2]['price'] = 0
                        
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
                    if stock != s1 and stock != s2 and stock in context.target_weights.keys():
                        context.target_weights[stock] = context.target_weights.loc[stock]*n/(n-2)
                for stock in stocks:
                    if stock != s1 and stock != s2:
                        scale_stock_to_leverage(context, stock, pair_weight=2/(n-2))

                        
                context.purchase_prices[s1]['price'] = 0
                context.purchase_prices[s2]['price'] = 0        
                        
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
                    if stock in context.target_weights:
                        context.target_weights[stock] = context.target_weights.loc[stock]*n/(n+2)
                for stock in stocks:
                    scale_stock_to_leverage(context, stock, pair_weight=(2/(n+2)))

                s1_weight = LEVERAGE * y_target_pct * (2/(n+2))
                s2_weight = LEVERAGE * x_target_pct * (2/(n+2))
                if (context.purchase_prices[s1]['price'] == 0):
                    context.purchase_prices[s1]['price'] = data.current(s1, 'price')
                    context.purchase_prices[s1]['long'] = True if s1_weight > 0 else False
                else:
                    is_long = context.purchase_prices[s1]['long']
                    if ((is_long and s1_weight < 0) or (not is_long and s1_weight > 0)):
                        context.purchase_prices[s1]['long'] = not context.purchase_prices[s1]['long']
                        context.purchase_prices[s1]['price'] = data.current(s1, 'price')
                
                if (context.purchase_prices[s2]['price'] == 0):
                    context.purchase_prices[s2]['price'] = data.current(s2, 'price')
                    context.purchase_prices[s2]['long'] = True if s2_weight > 0 else False
                else:
                    is_long = context.purchase_prices[s2]['long']
                    if ((is_long and s2_weight < 0) or (not is_long and s2_weight > 0)):
                        context.purchase_prices[s2]['long'] = not context.purchase_prices[s2]['long']
                        context.purchase_prices[s2]['price'] = data.current(s2, 'price')
                    
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
                    if stock in context.target_weights:
                        context.target_weights[stock] = context.target_weights.loc[stock]*n/(n+2)
                for stock in stocks:
                    scale_stock_to_leverage(context, stock, pair_weight=(2/(n+2)))

                    
                s1_weight = LEVERAGE * y_target_pct * (2/(n+2))
                s2_weight = LEVERAGE * x_target_pct * (2/(n+2))
                if (context.purchase_prices[s1]['price'] == 0):
                    context.purchase_prices[s1]['price'] = data.current(s1, 'price')
                    context.purchase_prices[s1]['long'] = True if s1_weight > 0 else False
                else:
                    is_long = context.purchase_prices[s1]['long']
                    if ((is_long and s1_weight < 0) or (not is_long and s1_weight > 0)):
                        context.purchase_prices[s1]['long'] = not context.purchase_prices[s1]['long']
                        context.purchase_prices[s1]['price'] = data.current(s1, 'price')
                
                if (context.purchase_prices[s2]['price'] == 0):
                    context.purchase_prices[s2]['price'] = data.current(s2, 'price')
                    context.purchase_prices[s2]['long'] = True if s2_weight > 0 else False
                else:
                    is_long = context.purchase_prices[s2]['long']
                    if ((is_long and s2_weight < 0) or (not is_long and s2_weight > 0)):
                        context.purchase_prices[s2]['long'] = not context.purchase_prices[s2]['long']
                        context.purchase_prices[s2]['price'] = data.current(s2, 'price')    
                    
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
            print("Not in weights")
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
        order_target_percent(s, context.target_weights.loc[s])