#Pair Trading Algorithm 
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline,CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.pipeline.data.morningstar as ms

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
from scipy.stats import shapiro
from pykalman import KalmanFilter
import math

LEVERAGE               = 1.0
MARKET_CAP             = 25
INTERVAL               = 1
DESIRED_PAIRS          = 10
HEDGE_LOOKBACK         = 21 
Z_WINDOW               = 21 
ENTRY                  = 1.25
EXIT                   = 0.50
Z_STOP                 = 3
STOPLOSS               = 0.15
MIN_SHARE              = 1.00
Z_PROTECT              = 0.05

# Quantopian constraints
MAX_PROCESSABLE_PAIRS  = 19000
MAX_KALMAN_STOCKS      = 150

REAL_UNIVERSE = [
    10101001, 10102002, 10103003, 10103004, 10104005, 10105006, 10105007, 10106008, 10106009, 10106010, 
    10106011, 10106012, 10107013, 10208014, 10208015, 10209016, 10209017, 10209018, 10209019, 10209020, 
    10210021, 10210022, 10210023, 10211024, 10211025, 10212026, 10212027, 10212028, 10213029, 10214030, 
    10215031, 10216032, 10217033, 10217034, 10217035, 10217036, 10217037, 10218038, 10218039, 10218040, 
    10218041, 10319042, 10320043, 10320044, 10320045, 10320046, 10320047, 10320048, 10320049, 10320050, 
    10320051, 10320052, 10321053, 10321054, 10321055, 10322056, 10323057, 10324058, 10325059, 10326060, 
    10326061, 10427062, 10427063, 10428064, 10428065, 10428066, 10428067, 10428068, 10428069, 10428070, 
    20529071, 20529072, 20530073, 20531074, 20531075, 20531076, 20531077, 20532078, 20533079, 20533080, 
    20533081, 20533082, 20534083, 20635084, 20636085, 20636086, 20637087, 20638088, 20638089, 20639090, 
    20640091, 20641092, 20642093, 20743094, 20744095, 20744096, 20744097, 20744098, 30845099, 30845100, 
    30946101, 30947102, 30948103, 30949104, 30950105, 30951106, 31052107, 31053108, 31054109, 31055110, 
    31056111, 31056112, 31057113, 31058114, 31058115, 31059116, 31060117, 31061118, 31061119, 31061120, 
    31061121, 31061122, 31062123, 31062124, 31062125, 31062126, 31062127, 31063128, 31064129, 31165130, 
    31165131, 31165132, 31165133, 31165134, 31166135, 31167136, 31167137, 31167138, 31167139, 31167140, 
    31167141, 31167142, 31167143, 31168144, 31169145, 31169146, 31169147
]

#Ranking metric: select key from TEST_PARAMS
RANK_BY                   = 'Hurst'
RANK_DESCENDING           = False
DESIRED_PVALUE            = 0.01
LOOKBACK                  = 253
LOOSE_PVALUE              = 0.05
PVALUE_TESTS              = ['Cointegration','ADFuller','Shapiro-Wilke']
RUN_BONFERRONI_CORRECTION = True
TEST_PARAMS               = {
    'Correlation':  {'lookback': LOOKBACK, 'min': 0.80, 'max': 1.00,           'type': 'price',  'run': False},
    'Cointegration':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'price',  'run': True },
    'Hurst':        {'lookback': LOOKBACK, 'min': 0.00, 'max': 0.49,           'type': 'spread', 'run': True },
    'ADFuller':     {'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'spread', 'run': True },
    'Half-life':    {'lookback': LOOKBACK, 'min': 1,    'max': INTERVAL*21,    'type': 'spread', 'run': True },
    'Shapiro-Wilke':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'spread', 'run': True },
    'Zscore':       {'lookback': Z_WINDOW, 'min': ENTRY*(1+Z_PROTECT),'max': ENTRY*(1+(2*Z_PROTECT)),'type': 'spread', 'run': True},
    'Alpha':        {'lookback': HEDGE_LOOKBACK, 'min': 0.00, 'max': 10**9,           'type': 'price', 'run':  True}
                             }

LOOSE_PARAMS              = {
    'Correlation':      {'min': 0.00,     'max': 1.00,         'run': False},
    'Cointegration':    {'min': 0.00,     'max': LOOSE_PVALUE, 'run': True },
    'ADFuller':         {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'Hurst':            {'min': 0.00,     'max': 0.49,         'run': False},
    'Half-life':        {'min': 1,        'max': INTERVAL*21,  'run': False},
    'Shapiro-Wilke':    {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'Zscore':           {'min': -Z_STOP,  'max': Z_STOP,       'run': True },
    'Alpha':            {'min': 0.00,     'max': 10**9,        'run': True }
                             }

class Stock:
    def __init__(self, equity, price_history):
        self.equity = equity
        self.price_history = price_history
        self.purchase_price = {'price': 0, 'long': False}

    def update_purchase_price(self, price, is_long):
        self.purchase_price['price'] = price
        self.purchase_price['long'] = is_long

    def test_stoploss(self, data):
        is_long = self.purchase_price['long']
        initial_price = self.purchase_price['price']
        current_price = data.current(self.equity, 'price')
        if initial_price == 0:
            return True
        return not ((is_long and current_price< (1-STOPLOSS)*initial_price) or (not is_long and current_price> (1+STOPLOSS)*initial_price))
    
class Pair:
    def __init__(self, data, s1, s2, industry):
        self.left = s1
        self.right= s2
        self.industry = industry
        self.spreads = []
        self.latest_test_results = {}
        self.currently_long = False
        self.currently_short = False

    def is_tradable(self, data):
        untradable = []
        if not data.can_trade(self.left.equity):
            untradable.append(self.left)
        if not data.can_trade(self.right.equity):
            untradable.append(self.right)
        if len(untradable) > 0:
            return False, untradable
        return True, []

    def test(self, context, data, loose_screens=False, test_type="spread"):
        for test in TEST_PARAMS:
            if (not TEST_PARAMS[test]['run']) or (loose_screens and not LOOSE_PARAMS[test]['run']) or (TEST_PARAMS[test]['type'] != test_type):
                continue
            current_test = get_test_by_name(test)
            result = "N/A"
            if TEST_PARAMS[test]['type'] == "price":
                try:
                    result = current_test(self.left.price_history[-TEST_PARAMS[test]['lookback']:], self.right.price_history[-TEST_PARAMS[test]['lookback']:])
                except:
                    pass
            elif TEST_PARAMS[test]['type'] == "spread":
                if self.spreads == []:
                    return False
                try:
                    result = current_test(self.spreads)
                except:
                    pass
            if result == 'N/A':
                return False
            self.latest_test_results[test] = round(result,6)
            upper_bound = TEST_PARAMS[test]['max'] if (not loose_screens) else LOOSE_PARAMS[test]['max']
            lower_bound = TEST_PARAMS[test]['min'] if (not loose_screens) else LOOSE_PARAMS[test]['min']
            if RUN_BONFERRONI_CORRECTION and test in PVALUE_TESTS:
                upper_bound /= len(PVALUE_TESTS)
            if not (result >= lower_bound and result <= upper_bound):
                return False

            if (test == RANK_BY) and (len(context.industries[self.industry]['top']) >= context.desired_pairs):
                bottom_result = context.industries[self.industry]['top'][0].latest_test_results[test]
                if (RANK_DESCENDING and result < bottom_result) or (not RANK_DESCENDING and result > bottom_result):
                    return False
                
        if TEST_PARAMS[RANK_BY]['type'] == test_type:
            context.industries[self.industry]['top'].append(self)
            context.industries[self.industry]['top'] = sorted(context.industries[self.industry]['top'], key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
            stock_list = []
            new_list = []
            for pair in context.industries[self.industry]['top']:
                if (not pair.left.equity in stock_list) and (not pair.right.equity in stock_list):
                    new_list.append(pair)
                    stock_list.append(pair.left.equity)
                    stock_list.append(pair.right.equity)
            
            context.industries[self.industry]['top'] = new_list
            if len(context.industries[self.industry]['top']) > context.desired_pairs:
                context.industries[self.industry]['top'].pop(0)
                
        return True

def initialize(context):
    context.num_pipes = (int)(len(REAL_UNIVERSE)/50) + (len(REAL_UNIVERSE)%50 > 0)*1
    for i in range(context.num_pipes):
        algo.attach_pipeline(make_pipeline(50*i, 50*i+50), "pipe" + str(i))

    context.initial_portfolio_value = context.portfolio.portfolio_value
    context.universe_set = False
    context.pairs_chosen = False
    context.curr_month = -1
    context.target_weights = {}
    context.industries = []
    context.weight_change = False
    day = get_datetime().day - (int)(2*get_datetime().day/7) - 3

    schedule_function(set_universe, date_rules.month_start(day * (not (day < 0 or day > 19))), time_rules.market_open(hours=0, minutes=1))
    schedule_function(calculate_price_histories, date_rules.every_day(), time_rules.market_open(hours=0, minutes=30))
    schedule_function(create_pairs, date_rules.every_day(), time_rules.market_open(hours=0, minutes=45))
    schedule_function(choose_pairs, date_rules.every_day(), time_rules.market_open(hours=1, minutes=0))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(hours = 1))

def make_pipeline(start, end):
    base_universe = QTradableStocksUS()
    industry_code = ms.asset_classification.morningstar_industry_code.latest
    sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30, mask=base_universe)
    columns = {}
    securities = (ms.valuation.market_cap.latest < 0 )
    for i in range(start, end):
        if (i >= len(REAL_UNIVERSE)):
            continue
        columns[str(REAL_UNIVERSE[i])] = (sma_short>MIN_SHARE) & industry_code.eq(REAL_UNIVERSE[i]) & (ms.valuation.market_cap.latest>MARKET_CAP*(10**6))
        securities = securities | columns[str(REAL_UNIVERSE[i])]
    return Pipeline(columns = columns, screen=(securities),)

def set_universe(context, data):
    this_month = get_datetime('US/Eastern').month 
    if context.curr_month < 0:
        context.curr_month = this_month
    context.next_month = context.curr_month + INTERVAL - 12*(context.curr_month + INTERVAL > 12)
    if (this_month != context.curr_month):
        return
    context.curr_month = context.next_month
    context.pairs = []
    context.universes = {}
    context.industries = {}
    context.target_weights = {}
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)

    context.pairs_chosen = False
    context.desired_pairs = int(round(DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)))

    pipe_output = algo.pipeline_output('pipe0')
    for i in range(1, context.num_pipes):
        pipe_output = pipe_output.append(algo.pipeline_output("pipe"+str(i)))
    pipe_output = pipe_output.fillna(False)

    print ("="*60)
    total = 0
    industry_pool = []
    context.max_kalman = 0
    for code in REAL_UNIVERSE:
        stock_obj_list = []
        stock_list = pipe_output[pipe_output[str(code)]].index.tolist()
        for stock in stock_list:
            new_stock = Stock(stock, [])
            stock_obj_list.append(new_stock)
        industry_pool = industry_pool + stock_obj_list
        if len(stock_obj_list) > 1:
            context.universe_set = True
            context.industries[code] = {'list': stock_obj_list, 'top': [], 'size': len(stock_obj_list)}
            total += len(stock_obj_list)
            if (len(stock_obj_list) > context.max_kalman):
                context.max_kalman = len(stock_obj_list) + 1
    context.max_kalman = MAX_KALMAN_STOCKS
    if not context.industries:
        print("No substantial universe found. Waiting until next cycle")
        context.universe_set = False
        return

    for stock in industry_pool:
        context.target_weights[stock.equity] = 0.0
    context.remaining_codes = sorted(context.industries, key=lambda kv: context.industries[kv]['size'], reverse=False)
    context.spread = np.ndarray((0, Z_WINDOW))
    context.delisted = []

def calculate_price_histories(context, data):
    record(market_exposure=context.account.net_leverage, leverage=context.account.leverage)
    if (not context.remaining_codes) or (not context.universe_set) or context.desired_pairs==0:
        context.desired_pairs = 0
        return

    sorted_codes = context.remaining_codes
    count = 0
    i = len(sorted_codes) - 1
    context.codes = []
    while (i >= 0):
        diff = context.max_kalman - count
        if context.industries[sorted_codes[i]]['size'] < diff:
            context.codes.append(sorted_codes[i])
            count += context.industries[sorted_codes[i]]['size']
            context.remaining_codes.remove(sorted_codes[i])
        i = i - 1

    report = "SETTING UNIVERSE"
    comps = 0
    for i in range(len(context.codes)):
        report += ("\n\t\t\t" if i%3 == 0 else "\t") + str(context.codes[i]) + " (" + str(context.industries[context.codes[i]]['size']) + ")"
        comps += context.industries[context.codes[i]]['size']*(context.industries[context.codes[i]]['size'] + 1)
    
    comps = comps if comps < MAX_PROCESSABLE_PAIRS else MAX_PROCESSABLE_PAIRS
    print (report + "\n\t\t\t--> Processed pairs = " + str(comps))

    for code in context.codes:
        for stock in context.industries[code]['list']:
            stock.price_history = data.history(stock.equity, "price", LOOKBACK+HEDGE_LOOKBACK, '1d')

def create_pairs(context, data):
    if not context.universe_set or context.desired_pairs == 0:
        return
    pair_counter = 0
    context.all_pairs = {}
    for code in context.codes:
        context.all_pairs[code] = []
        for i in range (context.industries[code]['size']):
            for j in range (i+1, context.industries[code]['size']):
                if (pair_counter > MAX_PROCESSABLE_PAIRS):
                    break
                pair_forward = Pair(data, context.industries[code]['list'][i], context.industries[code]['list'][j], code)
                pair_reverse = Pair(data, context.industries[code]['list'][j], context.industries[code]['list'][i], code)
                if pair_forward.test(context, data, test_type="price"):
                    pair_forward.spreads = get_spreads(data, pair_forward.left.price_history, pair_forward.right.price_history, LOOKBACK)
                    context.all_pairs[code].append(pair_forward)

                if pair_reverse.test(context, data, test_type="price"):
                    pair_reverse.spreads = get_spreads(data, pair_reverse.left.price_history, pair_reverse.right.price_history, LOOKBACK)
                    context.all_pairs[code].append(pair_reverse)
                pair_counter += 2
                
def choose_pairs(context, data):
    if not context.universe_set or context.desired_pairs == 0:
        return

    report = "CHOOSING " + str(context.desired_pairs) + " PAIR" + "S"*(context.desired_pairs > 1)
    new_pairs = []
    for code in context.all_pairs:
        for pair in context.all_pairs[code]:
            pair.test(context, data, test_type="spread")
        new_pairs = new_pairs + context.industries[code]['top']
        # if not (context.industries[code]['top'] == None):
        #     new_pairs.append(context.industries[code]['top'])
    new_pairs = sorted(new_pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
    num_pairs = context.desired_pairs if (len(new_pairs) > context.desired_pairs) else len(new_pairs)
    context.desired_pairs = context.desired_pairs - num_pairs
    report += " --> FOUND " + str(num_pairs)
    for i in range(num_pairs):
        report += ("\n\t\t\t"+str(len(context.pairs)+i+1)+") "+str(new_pairs[i].left.equity)+" & "+str(new_pairs[i].right.equity)
        +"\n\t\t\t\tIndustry Code:\t"+ str(new_pairs[i].industry))
        for test in new_pairs[i].latest_test_results:
            report += "\n\t\t\t\t" + str(test) + ": \t" + "\t"*(len(test) <= 5 ) + str(new_pairs[i].latest_test_results[test])
        context.pairs.append(new_pairs[i])
        context.pairs_chosen = True
    
    num_spreads = context.spread.shape[1]
    for index in range(num_pairs):
        new_spreads = np.ndarray((1, num_spreads))
        for i in range(num_spreads):
            diff = num_spreads-Z_WINDOW
            new_spreads[0][i] = new_pairs[index].spreads[-HEDGE_LOOKBACK:][i-diff] if i >= diff else 0
        context.spread = np.vstack((context.spread, new_spreads))
    print (report)

def check_pair_status(context, data):
    if (not context.pairs_chosen):
        return   
    num_pairs = len(context.pairs)
    if (num_pairs == 0):
        month = get_datetime('US/Eastern').month
        context.curr_month = month + 1 - 12*(month == 12)
        context.universe_set = False
        context.pairs_chosen = False
        return
    
    for i in range(num_pairs):
        if (i >= len(context.pairs)):
            break
        pair = context.pairs[i]
        
        if (not pair.left.test_stoploss(data)) or (not pair.right.test_stoploss(data)):
            print (str(pair.left.equity) + " & " + str(pair.right.equity) + " failed stoploss --> X")
            remove_pair(context, pair, index=i)
            i = i-1        
        is_tradable = pair.is_tradable(data)
        if not is_tradable[0]:
            print ("cannot trade  " + str(pair.left.equity) + " & " + str(pair.right.equity))
            remove_pair(context, pair, index=i)
            i = i-1
            del context.target_weights[pair.left.equity]
            del context.target_weights[pair.right.equity]
            context.delisted = context.delisted + is_tradable[1]

    temp_delisted = context.delisted
    for stock in temp_delisted:
        if (not stock.equity in context.portfolio.positions):
            context.delisted.remove(stock)
    if len(context.delisted) > 0:
        print("Stock not liquidated. Returning")
        return

    num_pairs = len(context.pairs)
    pair_index = 0
    new_spreads = np.ndarray((num_pairs, 1))
    while (pair_index < num_pairs):
        if (pair_index >= len(context.pairs)):
            break
        pair = context.pairs[pair_index]
        (s1, s2) = (pair.left, pair.right)
        (s1_price_test, s2_price_test) = (data.history(s1.equity, "price", LOOKBACK, '1d'), data.history(s2.equity, "price", LOOKBACK, '1d'))
        pair.left.price_history = s1_price_test
        pair.right.price_history = s2_price_test

        if not pair.test(context,data,loose_screens=True):
            print(str(s1.equity) + " & " + str(s2.equity) + " failed tests --> X")
            remove_pair(context, pair, index=pair_index)
            new_spreads = np.delete(new_spreads, pair_index, 0)
            continue
    
        s1_price = run_kalman(data.history(s1.equity, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::])
        s2_price = run_kalman(data.history(s2.equity, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::])

        hedge = np.polynomial.polynomial.polyfit(s2_price,s1_price,1)[1]
        new_spreads[pair_index, :] = s1_price[-1] - hedge * s2_price[-1]
        
        if context.spread.shape[1] >= Z_WINDOW:
            spreads = context.spread[pair_index, -Z_WINDOW:]
            context.pairs[pair_index].spreads = spreads
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()

            if (pair.currently_short and zscore < EXIT) or (pair.currently_long and zscore > -EXIT):     
                sell_pair(context, data, pair)
            elif pair.currently_short:
                y_target_shares = -1
                X_target_shares = hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price, new_pair=False)
            elif pair.currently_long:
                y_target_shares = 1
                X_target_shares = -hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price, new_pair=False)

            if zscore < -ENTRY and (not pair.currently_long):
                y_target_shares = 1
                X_target_shares = -hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price, new_pair=True)
                
            if zscore > ENTRY and (not pair.currently_short):
                y_target_shares = -1
                X_target_shares = hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price, new_pair=True)

        pair_index = pair_index+1

    context.spread = np.hstack([context.spread, new_spreads])
    allocate(context, data)

def sell_pair(context, data, pair):
    n = num_allocated_stocks(context)
    if n > 2:
        scale_stocks(context, n/(n-2))
        scale_pair_pct(context, 2/(n-2))
    pair.left.update_purchase_price(0, False)
    context.target_weights[pair.left.equity] = 0.0
    pair.right.update_purchase_price(0, False)
    context.target_weights[pair.right.equity] = 0.0
    pair.currently_short = False
    pair.currently_long = False
    context.weight_change = True

def buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price, new_pair=True):
    n = num_allocated_stocks(context)
    notionalDol =  abs(y_target_shares * s1_price[-1]) + abs(X_target_shares * s2_price[-1])
    (y_target_pct, x_target_pct) = (y_target_shares * s1_price[-1] / notionalDol, X_target_shares * s2_price[-1] / notionalDol)
    if new_pair:
        pair.currently_short = (y_target_shares < 0)
        pair.currently_long = (y_target_shares > 0)
        scale_stocks(context, n/(n+2))
        scale_pair_pct(context, 2/(n+2))
        update_target_weight(context, data, pair.left, LEVERAGE * y_target_pct * (2/(n+2)))
        update_target_weight(context, data, pair.right, LEVERAGE * x_target_pct * (2/(n+2)))
    else:
        update_target_weight(context, data, pair.left, LEVERAGE * y_target_pct* 2/n)
        update_target_weight(context, data, pair.right, LEVERAGE * x_target_pct* 2/n)

def allocate(context, data):
    if (not context.weight_change):
        return
    context.weight_change = False
    table = ""
    for pair in context.pairs:
        if (context.target_weights[pair.left.equity] != 0 or context.target_weights[pair.right.equity] != 0):
            table += ("\n\t|\t"+str(pair.left.equity)+"\t"+"+"*(1 if context.target_weights[pair.left.equity] >= 0 else 0) 
                  +str(round(context.target_weights[pair.left.equity],3)*100)+"%  "+"+"*(1 if context.target_weights[pair.right.equity] >= 0 else 0)
                  +str(round(context.target_weights[pair.right.equity],3)*100)+"%\t "+str(pair.right.equity) + "\t|")
        order_target_percent(pair.left.equity, context.target_weights[pair.left.equity])
        order_target_percent(pair.right.equity, context.target_weights[pair.right.equity])
    table = table if len(table) > 0 else "\n\t|\t\t\t\tall weights 0\t\t\t\t|"
    print ("ALLOCATING...\n\t " + "_"*71 + table + "\n\t|" + "_"*71 + "|")

def get_spreads(data, s1_price, s2_price, length):
    spreads = []
    for i in range(length):
        start_index = len(s1_price)-length+i
        try:
            hedge = np.polynomial.polynomial.polyfit(s2_price[start_index-HEDGE_LOOKBACK:start_index],s1_price[start_index-HEDGE_LOOKBACK:start_index],1)[1]
        except:
            return []
        spreads = np.append(spreads, s1_price[start_index-1] - hedge*s2_price[start_index-1])
    return spreads

def num_allocated_stocks(context):
    total = 0
    for pair in context.pairs:
        total = total + (1 if context.target_weights[pair.left.equity] != 0 else 0) + (1 if context.target_weights[pair.right.equity] != 0 else 0)
    return total

def scale_stocks(context, factor):
    for k in context.target_weights:
        context.target_weights[k] = context.target_weights[k]*factor

def scale_pair_pct(context, factor):
    for pair in context.pairs:
        s1_weight = context.target_weights[pair.left.equity]
        s2_weight = context.target_weights[pair.right.equity]
        total = abs(s1_weight) + abs(s2_weight)
        if (total != 0) and (total != LEVERAGE*factor):
            context.target_weights[pair.left.equity] = LEVERAGE * factor * s1_weight / total
            context.target_weights[pair.right.equity] = LEVERAGE * factor * s2_weight / total

def run_kalman(price_history):
    kf_stock = KalmanFilter(transition_matrices = [1], observation_matrices = [1], initial_state_mean = price_history.values[0], 
                            initial_state_covariance = 1, observation_covariance=1, transition_covariance=.05)

    return kf_stock.filter(price_history.values)[0].flatten()

def update_target_weight(context, data, stock, new_weight):
    if (stock.purchase_price['price'] == 0):
        is_long = True if new_weight > 0 else False
        stock.update_purchase_price(data.current(stock.equity, 'price'), is_long)        
    else:
        is_long = stock.purchase_price['long']
        if ((is_long and new_weight < 0) or (not is_long and new_weight > 0)):
            stock.update_purchase_price(data.current(stock.equity, 'price'), not is_long)
    context.target_weights[stock.equity] = new_weight
    context.weight_change = True

def remove_pair(context, pair, index):
    order_target_percent(pair.left.equity, 0)
    order_target_percent(pair.right.equity, 0)
    context.target_weights[pair.left.equity] = 0.0
    context.target_weights[pair.right.equity] = 0.0
    context.pairs.remove(pair)
    context.spread = np.delete(context.spread, index, 0)

def get_test_by_name(name):
    def correlation(a,b):
        return abs(np.corrcoef(a,b)[0][1])
    def cointegration(a, b):
        score, pvalue, _ = sm.coint(a, b)
        return pvalue
    def adf_pvalue(spreads):
        return sm.adfuller(spreads,1)[1]
    def hurst_hvalue(ts):
        # ts = np.asarray(ts)
        # lagvec = []
        # tau = []
        # lags = list(range(2, 100))
        # for lag in lags:
        #     pdiff = np.subtract(ts[lag:],ts[:-lag])
        #     lagvec.append(lag)
        #     tau.append(np.sqrt(np.std(pdiff)))
        # m = np.polynomial.polynomial.polyfit(np.log(np.asarray(lagvec)), np.log(np.asarray(tau)), 1)
        # return m[0]*2.0
        
        npspreads = np.array(ts)
        return compute_Hc(npspreads)[0]
    def half_life(spreads): 
        lag = np.roll(spreads, 1)
        lag[0] = 0
        ret = spreads - lag
        ret[0] = 0
        return(-np.log(2) / np.polynomial.polynomial.polyfit(lag, ret, 1)[1])
    
    def shapiro_pvalue(spreads):
        w, p = shapiro(spreads)
        return p
    
    def zscore(spreads):
        spreads = spreads[-Z_WINDOW:]
        return abs((spreads[-1]-spreads.mean())/spreads.std())
    
    def alpha(price1, price2):
        return np.polynomial.polynomial.polyfit(price2,price1,1)[1]
    
    def default(a=0, b=0):
        return a
    
    if (name.lower() == "correlation"):
        return correlation
    elif (name.lower() == "cointegration"):
        return cointegration
    elif (name.lower() == "adfuller"):
        return adf_pvalue
    elif (name.lower() == "hurst"):
        return hurst_hvalue
    elif (name.lower() == "half-life"):
        return half_life
    elif (name.lower() == "shapiro-wilke"):
        return shapiro_pvalue
    elif (name.lower() == "zscore"):
        return zscore
    elif (name.lower() == "alpha"):
        return alpha
    return default


def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs

def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts

def __get_simplified_RS(series, kind):
    """
    Simplified version of rescaled range

    Parameters
    ----------

    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        R = max(series) - min(series)  # range in absolute values
        S = np.std(incs, ddof=1)
    elif kind == 'price':
        pcts = __to_pct(series)
        R = max(series) / min(series) - 1. # range in percent
        S = np.std(pcts, ddof=1)
    elif kind == 'change':
        incs = series
        _series = np.hstack([[0.],np.cumsum(incs)])
        R = max(_series) - min(_series)  # range in absolute values
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio

    return R / S

def __get_RS(series, kind):
    """
    Get rescaled range (using the range of cumulative sum
    of deviations instead of the range of a series as in the simplified version
    of R/S) from a time-series of values.

    Parameters
    ----------

    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        mean_inc = (series[-1] - series[0]) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'price':
        incs = __to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'change':
        incs = series
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due undefined R/S

    return R / S

def compute_Hc(series, kind="random_walk", min_window=10, max_window=None, simplified=True):
    """
    Compute H (Hurst exponent) and C according to Hurst equation:
    E(R/S) = c * T^H

    Refer to:
    https://en.wikipedia.org/wiki/Hurst_exponent
    https://en.wikipedia.org/wiki/Rescaled_range
    https://en.wikipedia.org/wiki/Random_walk

    Parameters
    ----------

    series : array-like
        (Time-)series

    kind : str
        Kind of series
        possible values are 'random_walk', 'change' and 'price':
        - 'random_walk' means that a series is a random walk with random increments;
        - 'price' means that a series is a random walk with random multipliers;
        - 'change' means that a series consists of random increments
            (thus produced random walk is a cumulative sum of increments);

    min_window : int, default 10
        the minimal window size for R/S calculation

    max_window : int, default is the length of series minus 1
        the maximal window size for R/S calculation

    simplified : bool, default True
        whether to use the simplified or the original version of R/S calculation

    Returns tuple of
        H, c and data
        where H and c — parameters or Hurst equation
        and data is a list of 2 lists: time intervals and R/S-values for correspoding time interval
        for further plotting log(data[0]) on X and log(data[1]) on Y
    """

    # if len(series)<100:
    #     raise ValueError("Series length must be greater or equal to 100")

    # ndarray_likes = [np.ndarray]
    # if "pandas.core.series" in sys.modules.keys():
    #     ndarray_likes.append(pd.core.series.Series)

    # # convert series to numpy array if series is not numpy array or pandas Series
    # if type(series) not in ndarray_likes:
    #     series = np.array(series)

    # if "pandas.core.series" in sys.modules.keys() and type(series) == pd.core.series.Series:
    #     if series.isnull().values.any():
    #         raise ValueError("Series contains NaNs")
    #     series = series.values  # convert pandas Series to numpy array
    # elif np.isnan(np.min(series)):
    #     raise ValueError("Series contains NaNs")

    if simplified:
        RS_func = __get_simplified_RS
    else:
        RS_func = __get_RS


    err = np.geterr()
    np.seterr(all='raise')

    max_window = max_window or len(series)-1
    window_sizes = list(map(
        lambda x: int(10**x),
        np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start+w)>len(series):
                break
            _ = RS_func(series[start:start+w], kind)
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    np.seterr(**err)

    c = 10**c
    return H, c, [window_sizes, RS]

def random_walk(length, proba=0.5, min_lookback=1, max_lookback=100, cumprod=False):
    """
    Generates a random walk series

    Parameters
    ----------

    proba : float, default 0.5
        the probability that the next increment will follow the trend.
        Set proba > 0.5 for the persistent random walk,
        set proba < 0.5 for the antipersistent one

    min_lookback: int, default 1
    max_lookback: int, default 100
        minimum and maximum window sizes to calculate trend direction
    cumprod : bool, default False
        generate a random walk as a cumulative product instead of cumulative sum
    """

    assert(min_lookback>=1)
    assert(max_lookback>=min_lookback)

    if max_lookback > length:
        max_lookback = length
        # warnings.warn("max_lookback parameter has been set to the length of the random walk series.")

    if not cumprod:  # ordinary increments
        series = [0.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                direction = np.sign(series[i-1] - series[i-1-lookback]) * np.sign(proba - np.random.uniform())
            series[i] = series[i-1] + np.fabs(np.random.randn()) * direction
    else:  # percent changes
        series = [1.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                direction = np.sign(series[i-1] / series[i-1-lookback] - 1.) * np.sign(proba - np.random.uniform())
            series[i] = series[i-1] * np.fabs(1 + np.random.randn()/1000. * direction)

    return series


# if __name__ == '__main__':

#     # Use random_walk() function or generate a random walk series manually:
#     # series = random_walk(99999, cumprod=True)
#     np.random.seed(42)
#     random_changes = 1. + np.random.randn(99999) / 1000.
#     series = np.cumprod(random_changes)  # create a random walk from random changes

#     # Evaluate Hurst equation
#     H, c, data = compute_Hc(series, kind='price', simplified=True)

#     # Plot
#     # uncomment the following to make a plot using Matplotlib:
#     """
#     import matplotlib.pyplot as plt

#     f, ax = plt.subplots()
#     ax.plot(data[0], c*data[0]**H, color="deepskyblue")
#     ax.scatter(data[0], data[1], color="purple")
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel('Time interval')
#     ax.set_ylabel('R/S ratio')
#     ax.grid(True)
#     plt.show()
#     """

#     print("H={:.4f}, c={:.4f}".format(H,c))
#     assert H<0.6 and H>0.4