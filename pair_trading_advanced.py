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

COMMISSION = 0.0035
LEVERAGE = 1.0
MAX_GROSS_EXPOSURE = LEVERAGE

def initialize(context):
    
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=COMMISSION))
    context.industry_code = ms.asset_classification.morningstar_industry_code.latest
    #ENTER DESIRED SECTOR CODES HERE
    context.codes = [30946101, 30947102, 30948103, 30949104, 30950105, 30951106, 10428064, 
                     10428065, 10428066, 10428067, 10428068, 10428069, 10428070]
    context.num_universes = len(context.codes)
    context.universes = {}
    
    for code in context.codes:
        context.universes[code] = {}
        context.universes[code]['pipe'] = Pipeline()
        context.universes[code]['pipe'] = algo.attach_pipeline(context.universes[code]['pipe'], 
                                                          name = str(code))
        context.universes[code]['pipe'].set_screen(QTradableStocksUS() & 
                                           context.industry_code.eq(code))
    
    context.price_history_length = 365
    context.long_ma_length = 30
    context.short_ma_length = 1
    context.entry_threshold = 0.2
    context.exit_threshold = 0.1
    
    context.desired_num_pairs = 1
    context.num_pairs = context.desired_num_pairs
    context.pvalue_th = 1
    context.corr_th = 0
    context.top_yield_pairs = []
    context.universe_set = False
    
    context.coint_data = {}
    context.coint_pairs = {}
    context.real_yield_keys = []
    context.pair_status = {}
    
    context.target_weights = {}
    
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback   
    #context.spread = np.ndarray((context.num_pairs, 0))
    
    schedule_function(choose_pairs, date_rules.month_start(), time_rules.market_open(hours=0, minutes=1))  
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))
    
#TODO: LEVERAGE HANDLING
# def warn_leverage(context, data):
#     log.warn('Leverage Exceeded: '+str(context.account.leverage))
#     context.open_orders = get_open_orders()
#     if context.open_orders:
#         for orders,_ in context.open_orders.iteritems():
#             cancel_order(orders)
#     for equity in context.portfolio.positions:  
#         order_target_percent(equity, 0)
    
#empty all data structures
def empty_data(context):
    context.coint_data = {}
    context.coint_pairs = {}
    context.real_yield_keys = []
    context.top_yield_pairs = []
   
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
   
def choose_pairs(context, data):
    this_month = get_datetime('US/Eastern').month  
    if this_month not in [3, 6, 9, 12]:  
        return 
    
    empty_data(context)
   
    for code in context.codes:
        context.universes[code]['universe'] = algo.pipeline_output(str(code))
        context.universes[code]['universe'] = context.universes[code]['universe'].index
        context.universes[code]['size'] = len(context.universes[code]['universe'])
        if context.universes[code]['size'] > 1:
            context.universe_set = True
        print("Universe " + str(code) + " size: " + str(context.universes[code]['size']))
        #print(context.universes[code]['universe'])
    
    #TODO: FIGURE THIS OUT
    # if (context.universe_size < 2):
    #     return
    # if (context.desired_num_pairs > context.universe_size/2.0):
    #     context.num_pairs = 1
    
    context.spread = np.ndarray((context.num_pairs, 0))
    
    for code in context.codes:
        for i in range (context.universes[code]['size']):
            for j in range (i+1, context.universes[code]['size']):
                s1 = context.universes[code]['universe'][i]
                s2 = context.universes[code]['universe'][j]
                correlation, coint_pvalue = get_corr_coint(data, s1, s2,
                                                           context.price_history_length)
                context.coint_data[(s1,s2)] = {"corr": correlation, "coint": coint_pvalue}
                if (coint_pvalue < context.pvalue_th and abs(correlation) > context.corr_th):
                    context.coint_pairs[(s1,s2)] = context.coint_data[(s1,s2)]      

    #sort pairs from highest to lowest cointegrations
    context.real_yield_keys = sorted(context.coint_pairs, key=lambda kv: context.coint_pairs[kv]['coint'], reverse=False)
    
    #select top num_pairs pairs
    npairs = context.num_pairs
    if (npairs > len(context.real_yield_keys)):
        npairs = len(context.real_yield_keys)
    for i in range(npairs):
        context.top_yield_pairs.append(context.real_yield_keys[i])
        u_code = 0
        for code in context.codes:
            if context.real_yield_keys[i][0] in context.universes[code]['universe']:
                u_code = code
        coint = context.coint_pairs[context.real_yield_keys[i]]['coint']
        corr = context.coint_pairs[context.real_yield_keys[i]]['corr']
        print("TOP PAIR " + str(i+1) + ": " + str(context.real_yield_keys[i]) 
              + "\n\t\t\tuniverse: " + str(u_code) + "\n\t\t\tcorrelation: " + str(corr) 
              + "\n\t\t\tcointegration: " + str(coint) + "\n")
    
    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False
    
#INCOMPLETE
def check_pair_status(context, data):
    if (not context.universe_set):
        return
    
    new_spreads = np.ndarray((context.num_pairs, 1))
    numPairs = context.num_pairs
    
    if (numPairs > len(context.top_yield_pairs)):
        numPairs = len(context.top_yield_pairs)
    for i in range(numPairs):
        pair = context.top_yield_pairs[i]
        s1 = pair[0]
        s2 = pair[1]
        
        s1_price = data.history(s1, 'price', 35, '1d').iloc[-context.lookback::]
        s2_price = data.history(s2, 'price', 35, '1d').iloc[-context.lookback::]
        
        
        try:
            hedge = hedge_ratio(s1_price, s2_price, add_const=True)      
        except ValueError as e:
            log.debug(e)
            return
        
        context.target_weights = get_current_portfolio_weights(context, data)
        #print(context.target_weights)
        new_spreads[i, :] = s1_price[-1] - hedge * s2_price[-1]  
        if context.spread.shape[1] > context.z_window:
            
            spreads = context.spread[i, -context.z_window:]
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()

            if context.pair_status[pair]['currently_short'] and zscore < 0.0:
                context.target_weights[s1] = 0
                context.target_weights[s2] = 0
                
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                
                #record(X_pct=0, Y_pct=0)
                allocate(context, data)
                return
            
            if context.pair_status[pair]['currently_long'] and zscore > 0.0:
                context.target_weights[s1] = 0
                context.target_weights[s2] = 0
                
                context.pair_status[pair]['currently_short'] = False
                context.pair_status[pair]['currently_long'] = False
                
                #record(X_pct=0, Y_pct=0)
                allocate(context, data)
                return
            
            if zscore < -1.0 and (not context.pair_status[pair]['currently_long']):
                # Only trade if NOT already in a trade 
                y_target_shares = 1
                X_target_shares = -hedge
                context.pair_status[pair]['currently_long'] = True
                context.pair_status[pair]['currenlty_short'] = False

                (y_target_pct, x_target_pct) = computeHoldingsPct(y_target_shares,X_target_shares, s1_price[-1], s2_price[-1])
                
                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_pairs)
                
                #record(Y_pct=y_target_pct, X_pct=x_target_pct)
                allocate(context, data)
                return
            
            if zscore > 1.0 and (not context.pair_status[pair]['currently_short']):
                # Only trade if NOT already in a trade
                y_target_shares = -1
                X_target_shares = hedge
                context.pair_status[pair]['currently_short'] = True
                context.pair_status[pair]['currently_long'] = False

                (y_target_pct, x_target_pct) = computeHoldingsPct( y_target_shares, X_target_shares, s1_price[-1], s2_price[-1] )
                
                context.target_weights[s1] = LEVERAGE * y_target_pct * (1.0/context.num_pairs)
                context.target_weights[s2] = LEVERAGE * x_target_pct * (1.0/context.num_pairs)
                
                #record(Y_pct=y_target_pct, X_pct=x_target_pct)
                
                allocate(context, data)
                return
            
    context.spread = np.hstack([context.spread, new_spreads])                                 


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
    universe_pool = context.universes[context.codes[0]]['universe']
    for t in range(1, context.num_universes):
        universe_pool = universe_pool | context.universes[context.codes[t]]['universe']
    
    return current_weights.reindex(positions_index.union(universe_pool), fill_value=0.0)  
    
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)    

def allocate(context, data):
    record(leverage=context.account.leverage)
    
    for s in context.target_weights.keys():
        if (not data.can_trade(s)):
            print("Cannot trade " + str(s))
            context.universe_set = False
            return
        if(np.isnan(context.target_weights.loc[s])):
            print("Invalid target weight " + str(s))
            context.universe_set = False
            return
        if context.target_weights.loc[s] != 0:
            print (str(s) + " " + str(context.target_weights.loc[s]))
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
    
    #TRASHED CODE
    
    # tw_divider = 2*context.num_pairs*1.0
    # if (2*context.num_pairs > context.universe_size):
    #     tw_divider = context.universe_size*1.0
    
    #context.target_weights = pd.Series(index=context.universe, data=(1/tw_divider))
    #print(context.target_weights)
    
    
    #industry_group = ms.asset_classification.morningstar_industry_group_code.latest
    #pipe.set_screen(QTradableStocksUS() & industry_group.element_of([30947]))  5 digit
    #pipe.set_screen(QTradableStocksUS() & Sector().eq(309)) 3 digit
