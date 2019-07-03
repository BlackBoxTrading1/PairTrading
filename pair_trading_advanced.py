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
import math

COMMISSION         = 0.0035
LEVERAGE           = 2.0
MAX_GROSS_EXPOSURE = LEVERAGE
INTERVAL           = 3
DESIRED_PAIRS      = 2
SAMPLE_UNIVERSE    = [(symbol('ABGB'), symbol('FSLR'))]
REAL_UNIVERSE      = [10209016, 10209017, 10209018, 10209019, 10209020, 30946101, 30947102, 30948103, 30949104,
                      30950105, 30951106, 10428064, 10428065, 10428066, 10428067, 10428068, 10428069, 10428070,
                      31167136, 31167137, 31167138, 31167139, 31167140, 31167141, 31167142, 31167143]

#Data settings
#Cointegration / correlation
COINT_LOOKBACK     = 730
COINT_P_MAX        = 0.05
CORR_MIN           = 0.8
#ADFuller Test
ADF_LOOKBACK       = 63
ADF_P_MAX          = 0.05
#Hurst Test
HURST_H_MIN        = 0.0
HURST_H_MAX        = 0.4
#Half-life test
HALF_LIFE_LOOKBACK = 126
HALF_LIFE_MIN      = 1
HALF_LIFE_MAX      = 42
HEDGE_LOOKBACK     = 20 # used for regression
Z_WINDOW           = 20 # used for zscore calculation, must be <= lookback

#Choose tests
RUN_SAMPLE_PAIRS         = True
RUN_CORRELATION_TEST     = True
RUN_COINTEGRATION_TEST   = True
RUN_ADFULLER_TEST        = True
RUN_HURST_TEST           = True
RUN_HALF_LIFE_TEST       = True

#Rank pairs by (select key): 'coint', 'adf', 'corr', 'half-life', 'hurst'
RANK_BY = 'half-life'

#Display graphs
RECORD_LEVERAGE = True

def initialize(context):
    
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=COMMISSION))
    context.industry_code = ms.asset_classification.morningstar_industry_code.latest
    #ENTER DESIRED SECTOR CODES:
    context.codes = REAL_UNIVERSE
    context.num_universes = len(context.codes)
    context.universes = {}
    
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
    
    context.coint_data = {}
    context.coint_pairs = {}
    context.real_yield_keys = []
    context.pair_status = {}
    context.total_stock_list = []
    context.universe_pool = []
    
    context.target_weights = {}
 
    context.interval_mod = -1
    
    if RUN_SAMPLE_PAIRS:
        schedule_function(sample_comparison_test, date_rules.month_start(), time_rules.market_open(hours=0, minutes=1))
    else:
        schedule_function(choose_pairs, date_rules.month_start(), time_rules.market_open(hours=0, minutes=1))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))

def empty_data(context):
    context.coint_data = {}
    context.coint_pairs = {}
    context.real_yield_keys = []
    context.top_yield_pairs = []
    context.total_stock_list = []
    
def empty_target_weights(context):
    for s in context.target_weights.keys():
        context.target_weights.loc[s] = 0.0
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)

def get_stock_partner(context, stock):
    partner = 0
    for pair in context.coint_pairs.keys():
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

#return correlation and cointegration pvalue
def get_corr_coint(data, s1, s2, length):
    s1_price = data.history(s1, "price", length, '1d')
    s2_price = data.history(s2, "price", length, '1d')
    score, pvalue, _ = sm.coint(s1_price, s2_price)
    correlation = s1_price.corr(s2_price)
    
    return correlation, pvalue

#return long and short moving avg
def get_mvg_averages(data, s1, s2, long_length, short_length):
    prices = data.history([s1, s2], "price", long_length, '1d')
    short_prices = prices.iloc[-short_length:]
    long_ma = np.mean(prices[s1] - prices[s2])
    short_ma = np.mean(short_prices[s1] - short_prices[s2])
    return long_ma, short_ma

#calculate std
def get_std(data, s1, s2, length):
    prices = data.history([s1, s2], "price", length, '1d')
    std = np.std(prices[s1] - prices[s2])
    return std

def hedge_ratio(Y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model.params[1]
    model = sm.OLS(Y, X).fit()
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

def get_spreads(data, s1, s2, length):
    s1_price = data.history(s1, "price", length, '1d')
    s2_price = data.history(s2, "price", length, '1d')
    try:
        hedge = hedge_ratio(s1_price, s2_price, add_const=True)      
    except ValueError as e:
        log.debug(e)
        return
    spreads = []
    for i in range(length):
        spreads = np.append(spreads, s1_price[i] - hedge*s2_price[i])

    return spreads
def get_adf_pvalue(spreads):
    return sm.adfuller(spreads,1)[1]

def get_half_life(spreads): 
    lag = np.roll(spreads, 1)
    lag[0] = 0
    ret = spreads - lag
    ret[0] = 0
    
    lag2 = sm.add_constant(lag)
    model = sm.OLS(ret, lag2)
    res = model.fit()
    return (-np.log(2) / res.params[1])

def get_hurst_hvalue(spreads):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(spreads[lag:], spreads[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
    return poly[0]*2.0
    
    

#REMOVE EVENTUALLY*****************************************************************************************
def sample_comparison_test(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.interval_mod < 0:
        context.interval_mod = this_month % INTERVAL
    if (this_month % INTERVAL) != context.interval_mod:
        return
    
    context.num_pairs = DESIRED_PAIRS
    empty_data(context)
    
    context.universe_pool = pd.Index([])
    for pair in SAMPLE_UNIVERSE:
        context.coint_pairs[pair] = {}
        context.universe_pool.append(pd.Index([pair[0], pair[1]]))
        corr, coint = get_corr_coint(data, pair[0], pair[1], COINT_LOOKBACK)
        adf_p = 'N/A'
        hl = 'N/A'
        hurst_h = 'N/A'
        if RUN_ADFULLER_TEST:
            spreads = get_spreads(data, pair[0], pair[1], ADF_LOOKBACK)
            adf_p = get_adf_pvalue(spreads)
            
        if RUN_HALF_LIFE_TEST or RUN_HURST_TEST:
            spreads = get_spreads(data, pair[0], pair[1], HALF_LIFE_LOOKBACK)
            if RUN_HALF_LIFE_TEST:
                hl = get_half_life(spreads)
            if RUN_HURST_TEST:
                hurst_h = get_hurst_hvalue(spreads)
        
        context.coint_pairs[pair]['corr'] = corr
        context.coint_pairs[pair]['coint'] = coint
        context.coint_pairs[pair]['adf'] = adf_p
        context.coint_pairs[pair]['half-life'] = hl
        context.coint_pairs[pair]['hurst'] = hurst_h
        
    
    context.target_weights = get_current_portfolio_weights(context, data)
    empty_target_weights(context)
    
    
    context.real_yield_keys = sorted(context.coint_pairs, key=lambda kv: context.coint_pairs[kv][RANK_BY], reverse=False)
    
    temp_real_yield_keys = context.real_yield_keys
    for pair in temp_real_yield_keys:
        if (pair[0] in context.total_stock_list) or (pair[1] in context.total_stock_list):
            context.real_yield_keys.remove(pair)
            del context.coint_pairs[pair]
        else:
            context.total_stock_list.append(pair[0])
            context.total_stock_list.append(pair[1])
            
    if (context.num_pairs > len(context.real_yield_keys)):
        context.num_pairs = len(context.real_yield_keys) 
    for i in range(context.num_pairs):
        context.top_yield_pairs.append(context.real_yield_keys[i])
        coint = context.coint_pairs[context.real_yield_keys[i]]['coint']
        corr = context.coint_pairs[context.real_yield_keys[i]]['corr']
        adf_p = context.coint_pairs[context.real_yield_keys[i]]['adf']
        hl = context.coint_pairs[context.real_yield_keys[i]]['half-life']
        hurst_h = context.coint_pairs[context.real_yield_keys[i]]['hurst']
        
        print("TOP PAIR " + str(i+1) + ": " + str(context.real_yield_keys[i]) 
              + "\n\t\t\tcorrelation: \t" + str(round(corr,3)) 
              + "\n\t\t\tcointegration: \t" + str(coint)
              + "\n\t\t\tadf p-value: \t" + str(adf_p)
              + "\n\t\t\thalf-life: \t" + str(hl)
              + "\n\t\t\thurst h-value: \t" + str(hurst_h) + "\n")
    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False
    
    context.universe_set = True
    context.spread = np.ndarray((context.num_pairs, 0))
#*************************************************************************************************************
    
def choose_pairs(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.interval_mod < 0:
        context.interval_mod = this_month % INTERVAL
    if (this_month % INTERVAL) != context.interval_mod:
        return
    
    context.num_pairs = DESIRED_PAIRS
    
    empty_data(context)
    size_str = ""
    for code in context.codes:
        context.universes[code]['universe'] = algo.pipeline_output(str(code))
        context.universes[code]['universe'] = context.universes[code]['universe'].index
        context.universes[code]['size'] = len(context.universes[code]['universe'])
        if context.universes[code]['size'] > 1:
            context.universe_set = True
        size_str = size_str + " " + str(context.universes[code]['size'])
    print ("CHOOSING PAIRS...\nUniverse sizes:" + size_str)
    context.universe_pool = context.universes[context.codes[0]]['universe']
    for code in context.codes:
        context.universe_pool = context.universe_pool | context.universes[code]['universe']
    
    context.target_weights = get_current_portfolio_weights(context, data)
    empty_target_weights(context)
    #context.spread = np.ndarray((context.num_pairs, 0))
    
    #SCREENING
    for code in context.codes:
        for i in range (context.universes[code]['size']):
            for j in range (i+1, context.universes[code]['size']):
                s1 = context.universes[code]['universe'][i]
                s2 = context.universes[code]['universe'][j]
                correlation, coint_pvalue = get_corr_coint(data, s1, s2, COINT_LOOKBACK)
                context.coint_data[(s1,s2)] = {"corr": correlation, "coint": coint_pvalue, 
                                               "adf": "N/A", "half-life": "N/A", "hurst": "N/A"}
                
                passed_corr = (not RUN_CORRELATION_TEST) or (abs(correlation) > CORR_MIN)
                passed_coint = (not RUN_COINTEGRATION_TEST) or (coint_pvalue < COINT_P_MAX)

                if (passed_corr and passed_coint):
                    adf_p = 100
                    if RUN_ADFULLER_TEST:
                        spreads = get_spreads(data, s1, s2, ADF_LOOKBACK)
                        adf_p = get_adf_pvalue(spreads)
                        context.coint_data[(s1,s2)]['adf'] = adf_p       
                    if (not RUN_ADFULLER_TEST) or (adf_p < ADF_P_MAX):
                        spreads = get_spreads(data, s1, s2, HALF_LIFE_LOOKBACK)
                        hurst_h = -1
                        if RUN_HURST_TEST:
                            hurst_h = get_hurst_hvalue(spreads)
                            context.coint_data[(s1,s2)]['hurst'] = hurst_h
                        if (not RUN_HURST_TEST) or (hurst_h < HURST_H_MAX and hurst_h > HURST_H_MIN):
                            hl = -1
                            if RUN_HALF_LIFE_TEST:
                                hl = get_half_life(spreads)
                                context.coint_data[(s1,s2)]['half-life'] = hl
                            if (not RUN_HALF_LIFE_TEST) or (hl > HALF_LIFE_MIN and hl < HALF_LIFE_MAX):
                                context.coint_pairs[(s1,s2)] = context.coint_data[(s1,s2)]
    
    #sort pairs from highest to lowest cointegrations
    context.real_yield_keys = sorted(context.coint_pairs, key=lambda kv: context.coint_pairs[kv][RANK_BY], reverse=False)
    
    temp_real_yield_keys = context.real_yield_keys
    for pair in temp_real_yield_keys:
        if (pair[0] in context.total_stock_list) or (pair[1] in context.total_stock_list):
            context.real_yield_keys.remove(pair)
            del context.coint_pairs[pair]
        else:
            context.total_stock_list.append(pair[0])
            context.total_stock_list.append(pair[1])
        
    
    #select top num_pairs pairs
    if (context.num_pairs > len(context.real_yield_keys)):
        context.num_pairs = len(context.real_yield_keys)
    for i in range(context.num_pairs):
        context.top_yield_pairs.append(context.real_yield_keys[i])
        u_code = 0
        for code in context.codes:
            if context.real_yield_keys[i][0] in context.universes[code]['universe']:
                u_code = code
        coint = context.coint_pairs[context.real_yield_keys[i]]['coint']
        corr = context.coint_pairs[context.real_yield_keys[i]]['corr']
        adf_p = context.coint_pairs[context.real_yield_keys[i]]['adf']
        hl = context.coint_pairs[context.real_yield_keys[i]]['half-life']
        hurst_h = context.coint_pairs[context.real_yield_keys[i]]['hurst']
        
        print("TOP PAIR " + str(i+1) + ": " + str(context.real_yield_keys[i]) 
              + "\n\t\t\tsector: \t" + str(u_code) + "\n\t\t\tcorrelation: \t" + str(round(corr,3)) 
              + "\n\t\t\tcointegration: \t" + str(coint) 
              + "\n\t\t\tadf p-value: \t" + str(adf_p) 
              + "\n\t\t\thalf-life: \t" + str(hl) 
              + "\n\t\t\thurst h-value: \t" + str(hurst_h) + "\n")
    
    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False
    
    context.spread = np.ndarray((context.num_pairs, 0))
    
def check_pair_status(context, data):
    if (not context.universe_set):
        return
    
    new_spreads = np.ndarray((context.num_pairs, 1))
    numPairs = context.num_pairs
    for i in range(numPairs):
        pair = context.top_yield_pairs[i]
        # print pair
        s1 = pair[0]
        s2 = pair[1]
        
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

            if context.pair_status[pair]['currently_short'] and zscore < 0.0:
                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                #set_pair_status(context, data, s1,s2,s1_price,s2_price, 0, 0, False, False)
                if not RECORD_LEVERAGE:
                    record(Y_pct=0, X_pct=0)
                allocate(context, data)
                return
            
            if context.pair_status[pair]['currently_long'] and zscore > 0.0:
                context.target_weights[s1] = 0.0
                context.target_weights[s2] = 0.0
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                #set_pair_status(context, data, s1,s2,s1_price,s2_price, 0, 0, False, False)
                if not RECORD_LEVERAGE:
                    record(Y_pct=0, X_pct=0)
                allocate(context, data)
                return
            
            if zscore < -1.0 and (not context.pair_status[pair]['currently_long']):
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = True
                y_target_shares = 1
                X_target_shares = -hedge
                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )
                
                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_pairs)
    
                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)
                #set_pair_status(context,s1,s2,s1_price,s2_price, 1, -hedge, True, False)
                allocate(context, data)
                return
            
            if zscore > 1.0 and (not context.pair_status[pair]['currently_short']):
                context.pair_status[pair]['currently_short'] = True
                context.pair_status[pair]['currently_long'] = False
                y_target_shares = -1
                X_target_shares = hedge
                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )
                
                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_pairs)
    
                if not RECORD_LEVERAGE:
                    record(Y_pct=y_target_pct, X_pct=x_target_pct)
                #set_pair_status(context,s1,s2,s1_price,s2_price, -1, hedge, False, True)
                allocate(context, data)
                return
            
    context.spread = np.hstack([context.spread, new_spreads])

def allocate(context, data):
    if RECORD_LEVERAGE:
        record(leverage=context.account.leverage)
    print ("ALLOCATING...")
    for s in context.target_weights.keys():
        error = ""
        if (not s in context.target_weights):
            continue
        elif (not data.can_trade(s)):
            error = "Cannot trade " + str(s)
        elif(np.isnan(context.target_weights.loc[s])):
            error = "Invalid target weight " + str(s)
        if error:
            print(error)
            # context.universe_set = False
            # return
            partner = get_stock_partner(context, s)
            if not partner in context.target_weights:
                context.target_weights = context.target_weights.drop([s])
                context.universe_pool = context.universe_pool.drop([s])
            else:
                context.target_weights = context.target_weights.drop([s, partner])
                context.universe_pool = context.universe_pool.drop([s, partner])
                print("--> Removing partner " + str(partner) + "...")
            
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
    # if context.account.leverage>LEVERAGE or context.account.leverage < 0:
    #     warn_leverage(context, data)
