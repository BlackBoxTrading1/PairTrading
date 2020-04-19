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
import statsmodels.stats.diagnostic as sd
from scipy.stats import shapiro
from pykalman import KalmanFilter

LEVERAGE               = 1.0
MARKET_CAP             = 50
INTERVAL               = 1
DESIRED_PAIRS          = 2
HEDGE_LOOKBACK         = 21 
Z_WINDOW               = 21 
ENTRY                  = 1.5
EXIT                   = 0.5
Z_STOP                 = 3
STOPLOSS               = 0.20
MIN_SHARE              = 2
Z_PROTECT              = 0.20

# Quantopian constraints
MAX_PROCESSABLE_PAIRS  = 19000
MAX_KALMAN_STOCKS      = 150

REAL_UNIVERSE = [
    10101001, 10102002, 10103003, 10103004, 10104005, 10105006, 10105007, 10106008, 10106009, 10106010, 10106011, 10106012, 10107013, 10208014, 
    10208015, 10209016, 10209017, 10209018, 10209019, 10209020, 10210021, 10210022, 10210023, 10211024, 10211025, 10212026, 10212027, 10212028, 
    10213029, 10214030, 10215031, 10216032, 10217033, 10217034, 10217035, 10217036, 10217037, 10218038, 10218039, 10218040, 10218041, 10319042, 
    10320043, 10320044, 10320045, 10320046, 10320047, 10320048, 10320049, 10320050, 10320051, 10320052, 10321053, 10321054, 10321055, 10322056, 
    10323057, 10324058, 10325059, 10326060, 10326061, 10427062, 10427063, 10428064, 10428065, 10428066, 10428067, 10428068, 10428069, 10428070, 
    20529071, 20529072, 20530073, 20531074, 20531075, 20531076, 20531077, 20532078, 20533079, 20533080, 20533081, 20533082, 20534083, 20635084, 
    20636085, 20636086, 20637087, 20638088, 20638089, 20639090, 20640091, 20641092, 20642093, 20743094, 20744095, 20744096, 20744097, 20744098, 
    30845099, 30845100, 30946101, 30947102, 30948103, 30949104, 30950105, 30951106, 31052107, 31053108, 31054109, 31055110, 31056111, 31056112, 
    31057113, 31058114, 31058115, 31059116, 31060117, 31061118, 31061119, 31061120, 31061121, 31061122, 31062123, 31062124, 31062125, 31062126,
    31062127, 31063128, 31064129, 31165130, 31165131, 31165132, 31165133, 31165134, 31166135, 31167136, 31167137, 31167138, 31167139, 31167140, 
    31167141, 31167142, 31167143, 31168144, 31169145, 31169146, 31169147
]

#Ranking metric: select key from TEST_PARAMS
RANK_BY                   = 'Correlation'
RANK_DESCENDING           = True
DESIRED_PVALUE            = 0.01
LOOKBACK                  = 253
LOOSE_PVALUE              = 0.15
PVALUE_TESTS              = ['Cointegration','ADFuller','Shapiro-Wilke']
RUN_BONFERRONI_CORRECTION = True
TEST_PARAMS               = {
    'Correlation':  {'lookback': LOOKBACK, 'min': 0.90, 'max': 1.00,           'type': 'price',  'run': True},
    'Cointegration':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'price',  'run': True },
    'Hurst':        {'lookback': LOOKBACK, 'min': 0.00, 'max': 0.50,           'type': 'spread', 'run': False },
    'ADFuller':     {'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'spread', 'run': False},
    
    'Half-life':    {'lookback': LOOKBACK, 'min': 1,    'max': INTERVAL*21,    'type': 'spread', 'run': False},
    'Shapiro-Wilke':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE, 'type': 'spread', 'run': False},
    'Zscore':       {'lookback': Z_WINDOW, 'min': ENTRY*(1+Z_PROTECT),'max': Z_STOP*(1-Z_PROTECT),'key': 'zscore', 'run': False}
                             }

LOOSE_PARAMS              = {
    'Correlation':      {'min': 0.00, 'max': 1.00,         'run': False},
    'Cointegration':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
    'ADFuller':         {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False},
    'Hurst':            {'min': 0.00, 'max': 0.50,         'run': False},
    'Half-life':        {'min': 1,    'max': 63,           'run': False},
    'Shapiro-Wilke':    {'min': 0.00, 'max': LOOSE_PVALUE, 'run': False}
                             }

class Stock:
    def __init__(self, equity, price_history, partner):
        self.equity = equity
        self.price_history = price_history
        self.purchase_price = {'price': 0, 'long': False}
        self.partner_equity = partner
    
    def update_purchase_price(self, price, is_long):
        self.purchase_price['price'] = price
        self.purchase_price['long'] = is_long
        
    def set_price_history(self, price_history):
        self.price_history = price_history
        
    def test_stoploss(self, data):
        is_long = self.purchase_price['long']
        initial_price = self.purchase_price['price']
        current_price = data.current(self.equity, 'price')
        if initial_price == 0:
            return True
        return not ((is_long and current_price< (1-STOPLOSS)*initial_price) or (not is_long and current_price> (1+STOPLOSS)*initial_price))

class Industry:
    def __init__(self, code, stock_list):
        self.code = code
        self.stock_list = stock_list
        self.size = len(stock_list)
        self.top_pair = None

class Pair:
    def __init__(self, data, s1, s2, industry, s1_price, s2_price):
        self.left = Stock(s1, s1_price, s2)
        self.right= Stock(s2, s2_price, s1)
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
    
    def test(self, context, data, loose_screens=False):
        for test in TEST_PARAMS:
            if not TEST_PARAMS[test]['run'] or (loose_screens and not LOOSE_PARAMS[test]['run']):
                continue
            
            current_test = get_test_by_name(test)
            result = "N/A"
            (s1_price, s2_price) = (self.left.price_history, self.right.price_history)
            
            if TEST_PARAMS[test]['type'] == "price":
                try:
                    result = current_test(s1_price[-LOOKBACK:], s2_price[-LOOKBACK:])
                except:
                    pass
            elif TEST_PARAMS[test]['type'] == "spread":
                if self.spreads == []:
                    self.spreads = get_spreads(data, s1_price, s2_price, LOOKBACK)
                try:
                    result = current_test(self.spreads)
                except:
                    pass

            if result == 'N/A':
                return False

            self.latest_test_results[test] = round(result,3)
            upper_bound = TEST_PARAMS[test]['max'] if (not loose_screens) else LOOSE_PARAMS[test]['max']
            lower_bound = TEST_PARAMS[test]['min'] if (not loose_screens) else LOOSE_PARAMS[test]['min']
            if RUN_BONFERRONI_CORRECTION and test in PVALUE_TESTS:
                upper_bound /= len(PVALUE_TESTS)
            if not (result >= lower_bound and result <= upper_bound):
                return False
            if (test == RANK_BY) and (not self.industry.top_pair == None):
                top_result = self.industry.top_pair.latest_test_results[test]
                if (RANK_DESCENDING and result < top_result) or (not RANK_DESCENDING and result > top_result):
                    return False          

        current_result = self.latest_test_results[RANK_BY]
        top_result = self.industry.top_pair.latest_test_results[RANK_BY] if self.industry.top_pair != None else 'N/A'
        if (top_result == 'N/A') or (RANK_DESCENDING and current_result > top_result) or (not RANK_DESCENDING and current_result < top_result):
            self.industry.top_pair = self
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
    context.price_histories = {}
    day = get_datetime().day - (int)(2*get_datetime().day/7) - 3

    schedule_function(count_pairs, date_rules.every_day(), time_rules.market_open(hours=0, minutes=30))
    schedule_function(set_universe, date_rules.month_start(day * (not (day < 0 or day > 19))), time_rules.market_open(hours=0, minutes=1))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))

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

def calculate_price_histories(context, data):
    if not context.remaining_codes:
        context.desired_pairs = 0
        return

    sorted_codes = context.remaining_codes
    context.remaining_codes = []
    total = sum(code.size for code in sorted_codes)

    diff = total-context.max_kalman
    kalman_overflow = (diff > 0)
    while (diff > 0):
        diff = diff - sorted_codes[0].size
        context.remaining_codes.append(sorted_codes[0])
        sorted_codes.pop(0)
    context.industries = sorted_codes
    context.industries.reverse()

    updated_sizes_str = ""
    comps = 0
    for industry in context.industries:
        if kalman_overflow:
            updated_sizes_str = updated_sizes_str + str(industry.code) + " (" + str(industry.size) + ")  "
        comps += industry.size*(industry.size + 1)

    valid_num_comps = (comps <= MAX_PROCESSABLE_PAIRS )
    print ("SETTING UNIVERSE " + 
           "\n\t\t\tTotal stocks: " + str(total) + (" > " + str(context.max_kalman) + " --> removing smallest universes" 
           + "\n\t\t\tUniverse sizes: " + str(updated_sizes_str))*(kalman_overflow)
           + "\n\t\t\tProcessed pairs: " + str(comps) + (" > " + str(MAX_PROCESSABLE_PAIRS))*(not valid_num_comps))

    current_pool = []
    for industry in context.industries:
         current_pool = current_pool + industry.stock_list
    for i in range(context.max_kalman+diff):
        price_history = data.history(current_pool[i], "price", LOOKBACK+HEDGE_LOOKBACK, '1d')
        context.price_histories[current_pool[i]] = run_kalman(price_history)

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
    context.price_histories = {}
    context.industries = []
    context.target_weights = {}
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)

    context.pairs_chosen = False
    context.desired_pairs = int(round(DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)))

    pipe_output = algo.pipeline_output('pipe0')
    for i in range(1, context.num_pipes):
        pipe_output = pipe_output.append(algo.pipeline_output("pipe"+str(i)))
    pipe_output = pipe_output.fillna(False)

    print ("*"*70)
    total = 0
    context.industry_pool = []
    context.max_kalman = 0
    for code in REAL_UNIVERSE:
        new_industry = Industry(code, pipe_output[pipe_output[str(code)]].index.tolist())
        context.industry_pool = context.industry_pool + new_industry.stock_list
        if new_industry.size > 1:
            context.universe_set = True
            context.industries.append(new_industry)
            total += new_industry.size
            if (new_industry.size > context.max_kalman):
                context.max_kalman = new_industry.size + 1
    if not context.industries:
        print("No substantial universe found. Waiting until next cycle")
        context.universe_set = False
        return

    for stock in context.industry_pool:
        context.target_weights[stock] = 0.0

    context.remaining_codes = sorted(context.industries, key=lambda kv: kv.size, reverse=False)
    context.spread = np.ndarray((0, Z_WINDOW))
    context.delisted = []

def count_pairs(context, data):
    record(market_exposure=context.account.net_leverage, leverage=context.account.leverage)
    if context.universe_set and (context.desired_pairs != 0):
        calculate_price_histories(context, data)
        choose_pairs(context, data)
    
def choose_pairs(context, data):
    if not context.universe_set or context.desired_pairs == 0:
        return
    
    print(("CHOOSING " + str(context.desired_pairs) + " PAIR" + "S"*(context.desired_pairs > 1)))
    new_pairs = []
    context.code_max = {}
    pair_counter = 0
    for industry in context.industries:
        for i in range (industry.size):
            for j in range (i+1, industry.size):
                if (pair_counter > MAX_PROCESSABLE_PAIRS):
                    break
                (s1, s2) = (industry.stock_list[i], industry.stock_list[j])
                (s1_price, s2_price) = (context.price_histories[s1], context.price_histories[s2])
                pair_forward = Pair(data, s1, s2, industry, s1_price, s2_price)
                pair_reverse = Pair(data, s2, s1, industry, s2_price, s1_price)

                pair_forward.test(context, data)
                pair_reverse.test(context, data)
                pair_counter += 2
        if not industry.top_pair == None:
            new_pairs.append(industry.top_pair)
            
    new_pairs = sorted(new_pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
    num_pairs = context.desired_pairs if (len(new_pairs) > context.desired_pairs) else len(new_pairs)
    context.desired_pairs = context.desired_pairs - num_pairs
    print(("Pairs found: " + str(num_pairs)))
    for i in range(num_pairs):
        context.pairs.append(new_pairs[i])
        report = ("TOP PAIR "+str(i+1)+":  "+str(new_pairs[i].left.equity)+" & "+str(new_pairs[i].right.equity)
        +"\n\t\t\t\tIndustry Code:\t"+ str(new_pairs[i].industry.code))
        for test in new_pairs[i].latest_test_results:
            report += "\n\t\t\t\t" + str(test) + ": \t" + str(new_pairs[i].latest_test_results[test])
        print(report)
        context.pairs_chosen = True
        
    num_spreads = context.spread.shape[1]
    for index in range(num_pairs):
        (s1_price, s2_price) = (new_pairs[index].left.price_history, new_pairs[index].right.price_history)
        new_spreads = np.ndarray((1, num_spreads))
        for i in range(num_spreads):
            if (i >= num_spreads-Z_WINDOW):
                start_index = len(s1_price) - (num_spreads-i)
                (price1, price2) = (s1_price[(start_index-HEDGE_LOOKBACK): start_index], s2_price[(start_index-HEDGE_LOOKBACK): start_index])
                hedge = np.polynomial.polynomial.polyfit(price2,price1,1)[1]
                new_spreads[0][i] = price1[-1] - hedge*price2[-1]
            else:
                new_spreads[0][i] = 0
        context.spread = np.vstack((context.spread, new_spreads))    

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
            print ("Dumping  " + str(pair.left.equity) + " & " + str(pair.right.equity) + ".  Failed price stoploss")
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
    if (len(context.delisted) > 0):
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
        pair.left.set_price_history(s1_price_test)
        pair.right.set_price_history(s2_price_test)
        if not pair.test(context,data,loose_screens=True):
            print("Closing  " + str(s1.equity) + " & " + str(s2.equity) + ".  Failed tests.")
            remove_pair(context, pair, index=pair_index)
            continue
    
        s1_price = run_kalman(data.history(s1.equity, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::])
        s2_price = run_kalman(data.history(s2.equity, 'price', 35, '1d').iloc[-HEDGE_LOOKBACK::])

        hedge = np.polynomial.polynomial.polyfit(s2_price,s1_price,1)[1]
        new_spreads[pair_index, :] = s1_price[-1] - hedge * s2_price[-1]
        
        if context.spread.shape[1] >= Z_WINDOW:
            spreads = context.spread[pair_index, -Z_WINDOW:]
            zscore = (spreads[-1] - spreads.mean()) / spreads.std()
            if (zscore > Z_STOP) or (zscore < -Z_STOP):
                print("Removing " + str(pair.left.equity) + " & " + str(pair.right.equity) + "  failed Z Stop: " + str(round(zscore, 3)))
                remove_pair(context, pair, index=pair_index)
                new_spreads = np.delete(new_spreads, pair_index, 0)
                continue

            if (pair.currently_short and zscore < EXIT) or (pair.currently_long and zscore > -EXIT):     
                sell_pair(context, data, pair)

            if zscore < -ENTRY and (not pair.currently_long):
                y_target_shares = 1
                X_target_shares = -hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price)
                
            if zscore > ENTRY and (not pair.currently_short):
                y_target_shares = -1
                X_target_shares = hedge
                buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price)
            
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
    
def buy_pair(context, data, pair, y_target_shares, X_target_shares, s1_price, s2_price):
    pair.currently_short = (y_target_shares < 0)
    pair.currently_long = (y_target_shares > 0)
    n = num_allocated_stocks(context)
    scale_stocks(context, n/(n+2))
    scale_pair_pct(context, 2/(n+2))
    notionalDol =  abs(y_target_shares * s1_price[-1]) + abs(X_target_shares * s2_price[-1])
    (y_target_pct, x_target_pct) = (y_target_shares * s1_price[-1] / notionalDol, X_target_shares * s2_price[-1] / notionalDol)
    update_target_weight(context, data, pair.left, LEVERAGE * y_target_pct * (2/(n+2)))
    update_target_weight(context, data, pair.right, LEVERAGE * x_target_pct * (2/(n+2)))
    
def allocate(context, data):
    if (not context.weight_change):
        return
    context.weight_change = False
    print ("ALLOCATING...")
    for pair in context.pairs:
        if (context.target_weights[pair.left.equity] != 0 or context.target_weights[pair.right.equity] != 0):
            print("\t"+str(pair.left.equity)+":\t"+"+"*(1 if context.target_weights[pair.left.equity] >= 0 else 0) 
                  +str(round(context.target_weights[pair.left.equity],3)*100)+"%  "+"+"*(1 if context.target_weights[pair.right.equity] >= 0 else 0)
                  +str(round(context.target_weights[pair.right.equity],3)*100)+"%\t "+str(pair.right.equity))
        order_target_percent(pair.left.equity, context.target_weights[pair.left.equity])
        order_target_percent(pair.right.equity, context.target_weights[pair.right.equity])
    print ("\t" + "-"*60)
        
def get_spreads(data, s1_price, s2_price, length):
    spreads = []
    for i in range(length):
        try:
            start_index = len(s1_price)-length+i
            hedge = np.polynomial.polynomial.polyfit(s2_price[start_index-HEDGE_LOOKBACK:start_index],s1_price[start_index-HEDGE_LOOKBACK:start_index],1)[1]
        except ValueError as e:
            log.debug(e)
            return
        spreads = np.append(spreads, s1_price[i] - hedge*s2_price[i])
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
        s1_weight = 0
        if pair.left in context.target_weights:
            s1_weight = context.target_weights[pair.left.equity]
        s2_weight = 0
        if pair.right in context.target_weights:
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
        ts = np.asarray(ts)
        lagvec = []
        tau = []
        lags = list(range(2, 100))
        for lag in lags:
            pdiff = np.subtract(ts[lag:],ts[:-lag])
            lagvec.append(lag)
            tau.append(np.sqrt(np.std(pdiff)))
        m = np.polynomial.polynomial.polyfit(np.log(np.asarray(lagvec)), np.log(np.asarray(tau)), 1)
        return m[0]*2.0
    def half_life(spreads): 
        lag = np.roll(spreads, 1)
        lag[0] = 0
        ret = spreads - lag
        ret[0] = 0
        lag2 = sm.add_constant(lag)
        model = sm.OLS(ret, lag2)
        res = model.fit_regularized()
        return (-np.log(2) / res.params[1])
    def shapiro_pvalue(spreads):
        w, p = shapiro(spreads)
        return p
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
    return default