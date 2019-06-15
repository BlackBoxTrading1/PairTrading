#Pair Trading Algorithm

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline,CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
from quantopian.pipeline.filters import Q500US, Q3000US
from quantopian.pipeline.data import Fundamentals
from quantopian.algorithm import attach_pipeline, pipeline_output
from collections import OrderedDict
from quantopian.pipeline.classifiers.morningstar import Sector
import quantopian.pipeline.classifiers.morningstar
import quantopian.pipeline.data.morningstar as ms
import random
import math

COMMISSION = 0.0035
LEVERAGE = 1.0
MAX_GROSS_EXPOSURE = LEVERAGE


def initialize(context):
    
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=COMMISSION))
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name = 'pairs')
    
    industry_code = ms.asset_classification.morningstar_industry_code.latest
    pipe.set_screen(QTradableStocksUS() & industry_code.element_of([30948103]))
    #industry_group = ms.asset_classification.morningstar_industry_group_code.latest
    #pipe.set_screen(QTradableStocksUS() & industry_group.element_of([30947]))  5 digit
    #pipe.set_screen(QTradableStocksUS() & Sector().eq(309)) 3 digit
    context.universe = []
    context.q3000 = []
         
    context.price_history_length = 365
    context.long_ma_length = 30
    context.short_ma_length = 1
    context.entry_threshold = 0.2
    context.exit_threshold = 0.1
    context.universe_size = 100
    
    context.desired_num_pairs = 1
    context.num_pairs = context.desired_num_pairs
    context.pvalue_th = 1
    context.corr_th = 0
    context.top_yield_pairs = []
    context.universe_set = False
    
    context.coint_data = {}
    context.coint_pairs = {}
    context.spreads = {}
    context.real_yields = {}
    context.real_yield_keys = []
    context.pair_status = {}
    context.pair_weights = {}
    
    context.target_weights = {}
    
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback
    
    #context.spread = np.ndarray((context.num_pairs, 0))
    
    schedule_function(choose_pairs, date_rules.month_start(), time_rules.market_open(hours=0, minutes=1))  
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))
    
    
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
    context.spreads = {}
    context.real_yields = {}
    context.real_yield_keys = []
    context.top_yield_pairs = []
    context.pair_weights = {}
   
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
    context.universe = pipeline_output('pairs')
    context.universe = context.universe.index
    
   
    #print(context.universe)
    context.universe_set = True
    
    

    if (context.universe_size > len(context.universe)):
        context.universe_size = len(context.universe)
    print("Universe size: " + str(context.universe_size))
    
    if (context.universe_size < 2):
        return
    
    # tw_divider = 2*context.num_pairs*1.0
    # if (2*context.num_pairs > context.universe_size):
    #     tw_divider = context.universe_size*1.0
    
    #context.target_weights = pd.Series(index=context.universe, data=(1/tw_divider))
    #print(context.target_weights)
    
    if (context.desired_num_pairs > context.universe_size/2.0):
        context.num_pairs = 1
    
    context.spread = np.ndarray((context.num_pairs, 0))
    
    for i in range (context.universe_size):
        for j in range (i+1, context.universe_size):
            s1 = context.universe[i]
            s2 = context.universe[j]
            #get correlation cointegration values
            correlation, coint_pvalue = get_corr_coint(data, s1, s2, context.price_history_length)
            context.coint_data[(s1,s2)] = {"corr": correlation, "coint": coint_pvalue}
            if (coint_pvalue < context.pvalue_th and abs(correlation) > context.corr_th):
                context.coint_pairs[(s1,s2)] = context.coint_data[(s1,s2)]      
    
    #print(context.coint_pairs)
    for pair in context.coint_pairs:
        long_ma, short_ma = get_mvg_averages(data, pair[0], pair[1], context.long_ma_length, context.short_ma_length)
        #calculate spread and zscore
        context.spreads[pair] = (short_ma-long_ma)/long_ma
    
        port_val = context.portfolio.portfolio_value
        top_commission = get_commission(data, s1, port_val*0.5/context.num_pairs)
        bottom_commission = get_commission(data, s2, port_val*0.5/context.num_pairs)
        pair_commission = top_commission + bottom_commission
        context.real_yields[pair] = {}
        #subtract total commission of pair from % of portfolio value to get real yield
        context.real_yields[pair]['yield'] = abs(context.spreads[pair]) * port_val-pair_commission
        context.real_yields[pair]['corr'] = context.coint_data[pair]['corr']
        context.real_yields[pair]['coint'] = context.coint_data[pair]['coint']
    
    #sort pairs from highest to lowest correlations
    #context.real_yield_keys = sorted(context.real_yields, key=lambda kv: context.real_yields[kv]['corr'], reverse=True)
    context.real_yield_keys = sorted(context.real_yields, key=lambda kv: context.real_yields[kv]['coint'], reverse=False)
    
    #select top num_pairs pairs
    npairs = context.num_pairs
    if (npairs > len(context.real_yield_keys)):
        npairs = len(context.real_yield_keys)
    for i in range(npairs):
        context.top_yield_pairs.append(context.real_yield_keys[i])
        
        coint = context.real_yields[context.real_yield_keys[i]]['coint']
        corr = context.real_yields[context.real_yield_keys[i]]['corr']
        print("TOP PAIR: " + str(context.real_yield_keys[i]) +"\n\t\t\tcorrelation: " + str(corr) + "\n\t\t\tcointegration: " + str(coint) + "\n")
        
    #determine weights of each pair based on correlation
    total_corr = 0
    for pair in context.top_yield_pairs:
        total_corr += context.real_yields[pair]['corr']
    
    for pair in context.top_yield_pairs:
        context.pair_weights[pair] = context.real_yields[pair]['corr']/total_corr
    
    #print final data
    #print(context.real_yields) #prints yield and correlation of every pair that passed 1st screen
    #print(context.pair_weights) #prints weight of every pair in final list of top pairs
    
    #context.top_yield_pairs = [(symbol('ABGB'), symbol('FSLR')), (symbol('CSUN'), symbol('ASTI'))]
    
    for pair in context.top_yield_pairs:
        context.pair_status[pair] = {}
        context.pair_status[pair]['currently_short'] = False
        context.pair_status[pair]['currently_long'] = False
    
#INCOMPLETE
def check_pair_status(context, data):
    if (not context.universe_set):
        return
    
    #prices = data.history(context.universe, 'price', 35, '1d').iloc[-context.lookback::]
    #prices = data.history(context.universe.index, 'price', 35, '1d').iloc[-context.lookback::]
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
        
        #print(hedge)
        context.target_weights = get_current_portfolio_weights(context, data)
        #print(context.target_weights)
        
        new_spreads[i, :] = s1_price[-1] - hedge * s2_price[-1]
        
        if context.spread.shape[1] > context.z_window:
            #print(context.spread)
            spreads = context.spread[i, -context.z_window:]
            
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()
            #print (zscore)
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
    return current_weights.reindex(positions_index.union(context.universe), fill_value=0.0)  
    
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
    # print(context.target_weights.keys())
    # print(context.target_weights)
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
