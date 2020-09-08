from zipline.api import symbol, order_target_percent, schedule_function, get_datetime, time_rules, date_rules

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
import statsmodels.stats.diagnostic as sd
from scipy.stats import shapiro, jarque_bera, pearsonr, linregress
import math
from pykalman import KalmanFilter

import requests
import json
import alpaca_trade_api_fixed as tradeapi
import progressbar
import copy

MAX_COMPANY = 50
EXCHANGES = ['New York Stock Exchange', 'Nasdaq Global Select', 'NYSE American', 'NASDAQ Global Market', 'NASDAQ Capital Market']
BASE_URL = "https://paper-api.alpaca.markets"
KEY_ID = "PKSL6HFOBBRWI3ZYB3CE"
SECRET_KEY = "oFil1E/0DN1WTatQMGoo6YahQXudVRED9t6dBNbV"
EXCLUDED_INDUSTRIES = ['Banks', 'Insurance', 'Insurance - Life', 'Insurance - Specialty', 'Brokers & Exchanges', 'Insurance - Property & Casualty', 'Asset Management', 'REITs', 'Conglomerates', 'Credit Services', 'Utilities - Independent Power Producers', 'Utilities - Regulated', 'Health Care Plans', 'Real Estate Services']

LEVERAGE               = 1.0
INTERVAL               = 1
DESIRED_PAIRS          = 20
HEDGE_LOOKBACK         = 21  #usually 15-300
ENTRY                  = 1.5 #usually 1.5
EXIT                   = 0.1 #usually 0.0
Z_STOP                 = 4.0 #usually >4.0
STOPLOSS               = 0.15
MIN_SHARE              = 1.00
MIN_WEIGHT             = 0.25
BETA_LOWER             = 0.0
BETA_UPPER             = 1.0

#Ranking metric: select key from TEST_PARAMS
RANK_BY                   = 'Hurst'
RANK_DESCENDING           = False
DESIRED_PVALUE            = 0.01
LOOKBACK                  = 253 # usually set to 253
LOOSE_PVALUE              = 0.05
PVALUE_TESTS              = ['Cointegration', 'ADFuller','Shapiro-Wilke', 'Ljung-Box', 'Jarque-Bera']
RUN_BONFERRONI_CORRECTION = True
TEST_ORDER                = ['Cointegration', 'Alpha', 'Correlation', 'ADF-Prices', 'Hurst', 'Half-life', 'Zscore', 'ADFuller', 'Shapiro-Wilke', 'Jarque-Bera', 'Ljung-Box']

TEST_PARAMS               = {
    'Correlation':  {'lookback': LOOKBACK, 'min': 0.80, 'max': 1.00,                   'type': 'price',  'run': True },
    'Cointegration':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE,         'type': 'price',  'run': False},
    'Hurst':        {'lookback': LOOKBACK, 'min': 0.00, 'max': 0.49,                   'type': 'spread', 'run': True },
    'ADFuller':     {'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE,         'type': 'spread', 'run': True },
    'Half-life':    {'lookback': HEDGE_LOOKBACK, 'min': 2, 'max': HEDGE_LOOKBACK*2,    'type': 'spread', 'run': True },
    'Shapiro-Wilke':{'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE,         'type': 'spread', 'run': True },
    'Jarque-Bera':  {'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE,         'type': 'spread', 'run': False},
    'Zscore':       {'lookback': LOOKBACK, 'min': ENTRY,'max': Z_STOP,                 'type': 'spread', 'run': True },
    'Alpha':        {'lookback': HEDGE_LOOKBACK,   'min': 0.00, 'max': np.inf,         'type': 'price',  'run': True },
    'Ljung-Box':    {'lookback': LOOKBACK, 'min': 0.00, 'max': DESIRED_PVALUE,         'type': 'spread', 'run': False},
    'ADF-Prices':   {'lookback': LOOKBACK, 'min': LOOSE_PVALUE, 'max': 1.00,         'type': 'price',  'run': True }
    }
    
LOOSE_PARAMS              = {
    'Correlation':      {'min': 0.80,     'max': 1.00,         'run': False },
    'Cointegration':    {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'ADFuller':         {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'Hurst':            {'min': 0.00,     'max': 0.49,         'run': False},
    'Half-life':        {'min': 1,        'max': HEDGE_LOOKBACK*2,'run': False },
    'Shapiro-Wilke':    {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'Jarque-Bera':      {'min': 0.00,     'max': LOOSE_PVALUE, 'run': False},
    'Zscore':           {'min': 0,        'max': Z_STOP,       'run': True },
    'Alpha':            {'min': 0.00,     'max': np.inf,       'run': True },
    'Ljung-Box':        {'min': 0.00,     'max': np.inf,       'run': False},
    'ADF-Prices':       {'min': LOOSE_PVALUE, 'max': 1.00,     'run': False}
    }

def write_to_file(filename, data):
    file = open(filename, 'w')
    print(data, file=file)
    file.close()

def log(message):
    print(str(get_datetime()) + " > " + str(message))
    with open("logs.txt", "a") as myfile:
        myfile.write("\n"+ str(get_datetime()) + " > " + str(message))

class Stock:
    def __init__(self, equity):
        self.equity = equity
        self.name = str(self.equity.sid) + " " + str(equity.symbol)
        self.price_history = []
        self.filtered_price_history = []
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
        self.to_string = "([" + str(s1.name) + "] & [" + str(s2.name) + "])"
        self.industry = industry
        self.spreads = []
        self.filtered_spreads = []
        self.latest_test_results = {}
        self.latest_failed_test = ""
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
        for test in TEST_ORDER:
            if (not TEST_PARAMS[test]['run']) or (loose_screens and not LOOSE_PARAMS[test]['run']) or (TEST_PARAMS[test]['type'] != test_type):
                continue
            current_test = get_test_by_name(test)
            result = "N/A"
            if TEST_PARAMS[test]['type'] == "price":
                try:
                    if test == "Alpha":
                        result = current_test(self.left.price_history[-HEDGE_LOOKBACK:], self.right.price_history[-HEDGE_LOOKBACK:])
                    else:
                        result = current_test(self.left.filtered_price_history[-HEDGE_LOOKBACK:], self.right.filtered_price_history[-HEDGE_LOOKBACK:])
                except:
                    pass
                # try:
                #     if loose_screens and test == "Alpha":
                #         hl = int(round(self.latest_test_results['Half-life'], 0))
                #         result = current_test(self.left.price_history[-hl:], self.right.price_history[-hl:])
                #     else:
                #         result = current_test(self.left.price_history[-HEDGE_LOOKBACK:], self.right.price_history[-HEDGE_LOOKBACK:])
                # except:
                #     pass
            elif TEST_PARAMS[test]['type'] == "spread":
                if self.spreads == [] or self.filtered_spreads == []:
                    return False
                # result = current_test(self.spreads)
                try:
                    if test == "Zscore":
                        result = current_test(self.spreads[-HEDGE_LOOKBACK:])
                    if test == "Half-life":
                        result = current_test(self.filtered_spreads[-HEDGE_LOOKBACK:])
                    else:
                        result = current_test(self.filtered_spreads)
                except:
                    pass
            if result == 'N/A':
                self.latest_failed_test = test + " " + str(result)
                # if TEST_PARAMS[test]['type'] == "spread":
                #     log(str(test) + " " + str(result))
                return False
            self.latest_test_results[test] = result #round(result,6)
            upper_bound = TEST_PARAMS[test]['max'] if (not loose_screens) else LOOSE_PARAMS[test]['max']
            lower_bound = TEST_PARAMS[test]['min'] if (not loose_screens) else LOOSE_PARAMS[test]['min']
            if RUN_BONFERRONI_CORRECTION and test in PVALUE_TESTS:
                upper_bound /= len(PVALUE_TESTS)
            if not (result >= lower_bound and result <= upper_bound):
                self.latest_failed_test = test + " " + str(result)
                # if TEST_PARAMS[test]['type'] == "spread":
                #     log(str(test) + " " + str(result))
                return False

            if (not loose_screens) and (test == RANK_BY) and (len(context.industries[self.industry]['top']) >= context.desired_pairs):
                bottom_result = context.industries[self.industry]['top'][-1].latest_test_results[test]
                if (RANK_DESCENDING and result < bottom_result) or (not RANK_DESCENDING and result > bottom_result):
                    self.latest_failed_test = test + " " + str(result) + " no space"
                    # if TEST_PARAMS[test]['type'] == "spread":
                    #     log(str(test) + " ranking " + str(result))
                    return False

        if (not loose_screens) and test_type == "spread":
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
                del context.industries[self.industry]['top'][-1]

        return True

def collect_polygon_tickers(base_url, key_id, secret_key):
    
    API = tradeapi.REST(base_url=base_url, key_id=key_id, secret_key=secret_key)
    assets = API.list_assets()
    symbols = [asset.symbol for asset in assets if asset.tradable]

    num_requests = int(len(symbols)/MAX_COMPANY)+(1 if len(symbols) % MAX_COMPANY > 0 else 0)

    log("Pulling all Polygon companies")
    bar = progressbar.ProgressBar(maxval=num_requests, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', '<',progressbar.Timer(),'>'])
    bar.start()
    companies = {}
    for r in range(num_requests):
        new_companies = {}
        if r == num_requests - 1:
            new_companies = API.polygon.company(symbols[r*MAX_COMPANY:])
        else:
            new_companies = API.polygon.company(symbols[r*MAX_COMPANY:(r+1)*MAX_COMPANY])
        companies.update(new_companies)
        bar.update(r+1)
    bar.finish()

    industries = {}
    for ticker, company in companies.items():
        if not (company.exchange in EXCHANGES):
            continue
        if not hasattr(company, 'type') or company.type != 'CS':
            continue
        if ('Trust' in company.description.split()) or ('Fund' in company.description.split()):
            continue

        if not (company.industry in industries):
            industries[company.industry] = []
        industries[company.industry].append(ticker)

    delete = [key for key in industries if len(industries[key]) < 2 or key in EXCLUDED_INDUSTRIES]

    for key in delete:
        del industries[key]

    num_tickers = sum(len(industries[key]) for key in industries)
    log("Pulled {0} tickers".format(num_tickers))
    return industries

def initialize(context):
    write_to_file('logs.txt', "BACKTEST LOGS")
    context.initial_portfolio_value = context.portfolio.portfolio_value
    context.universe_set = False
    context.pairs_chosen = False
    context.curr_month = -1
    context.target_weights = {}
    context.pairs = []
    context.weight_change = False
    day = get_datetime().day - (int)(2*get_datetime().day/7) - 3

    schedule_function(set_universe, date_rules.month_start(day * (not (day < 0 or day > 19))), time_rules.market_open(hours=0, minutes=1))
    #schedule_function(calculate_price_histories, date_rules.every_day(), time_rules.market_open(hours=0, minutes=30))
    #schedule_function(create_pairs, date_rules.every_day(), time_rules.market_open(hours=0, minutes=45))
    #schedule_function(choose_pairs, date_rules.every_day(), time_rules.market_open(hours=1, minutes=0))
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(hours = 1))

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
    context.target_weights = {}
    for equity in context.portfolio.positions:  
        order_target_percent(equity, 0)
    
    context.industries = collect_polygon_tickers(BASE_URL, KEY_ID, SECRET_KEY)
    context.pairs_chosen = False
    context.desired_pairs = int(round(DESIRED_PAIRS * (context.portfolio.portfolio_value / context.initial_portfolio_value)))
    total = 0
    industry_pool = []
    context.max_kalman = 0

    log("Setting Universe...")
    for industry in context.industries:
        stock_list = []
        for ticker in context.industries[industry]:
            try:
                stock_list.append(Stock(symbol(ticker)))
            except:
                pass
        industry_pool = industry_pool+stock_list
        if len(stock_list) > 1:
            context.universe_set = True
        context.industries[industry] = {'list': stock_list, 'top': [], 'size': len(stock_list)}
        total += len(stock_list)

    temp = copy.deepcopy(context.industries)
    for i in temp:
        if temp[i]['size'] < 1:
            del context.industries[i]

    if not context.industries:
        log("No substantial universe found. Waiting until next cycle")
        context.universe_set = False
        return

    for stock in industry_pool:
        context.target_weights[stock.equity] = 0.0
    context.remaining_codes = sorted(context.industries, key=lambda kv: context.industries[kv]['size'], reverse=False)
    context.spread = np.ndarray((0, LOOKBACK))
    context.delisted = []


    # CALCULATE PRICE HISTORIES**************************************************
    if (not context.universe_set) or context.desired_pairs==0:
        context.desired_pairs = 0
        return

    context.codes = context.remaining_codes

    report = "Calculating Price Histories..."
    comps = 0
    count = 0
    for i in range(len(context.codes)):
        report += ("\n\t\t\t" if i%3 == 0 else "\t") + str(context.codes[i]) + " (" + str(context.industries[context.codes[i]]['size']) + ")"
        comps += context.industries[context.codes[i]]['size']*(context.industries[context.codes[i]]['size'] + 1)
        count += context.industries[context.codes[i]]['size']
    log (report + "\n\n\t\t\tTotal Tickers = " + str(count) + "\n\t\t\tTotal Pairs = " + str(comps))

    bar = progressbar.ProgressBar(maxval=count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', '<',progressbar.Timer(),'>'])
    bar.start()
    count = 0
    for code in context.codes:
        for stock in context.industries[code]['list']:
            stock.price_history = data.history(stock.equity, "price", LOOKBACK+HEDGE_LOOKBACK, '1d')
            stock.price_history = stock.price_history.values.tolist()
            stock.filtered_price_history = run_kalman(stock.price_history)
            bar.update(count+1)
            count += 1
    bar.finish()

    # PRICE TEST PAIRS**********************************************
    pair_counter = 0
    context.all_pairs = {}
    log("Price Testing Pairs...")
    counter = 0
    bar = progressbar.ProgressBar(maxval=comps, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', '<',progressbar.Timer(),'>'])
    bar.start()
    for code in context.codes:
        context.all_pairs[code] = []
        for i in range (context.industries[code]['size']):
            for j in range (i+1, context.industries[code]['size']):
                
                pair_forward = Pair(data, context.industries[code]['list'][i], context.industries[code]['list'][j], code)
                pair_reverse = Pair(data, context.industries[code]['list'][j], context.industries[code]['list'][i], code)
                if pair_forward.test(context, data, test_type="price"):
                    counter += 1
                    pair_forward.spreads = get_spreads(data, pair_forward.left.price_history, pair_forward.right.price_history, LOOKBACK)
                    try:
                        pair_forward.filtered_spreads = run_kalman(pair_forward.spreads)
                        context.all_pairs[code].append(pair_forward)
                    except:
                        log("forward pair failed kalman")
                bar.update(pair_counter+1)
                pair_counter += 1
                    
                if pair_reverse.test(context, data, test_type="price"):
                    counter += 1
                    pair_reverse.spreads = get_spreads(data, pair_reverse.left.price_history, pair_reverse.right.price_history, LOOKBACK)
                    try:
                        pair_reverse.filtered_spreads = run_kalman(pair_reverse.spreads)
                        context.all_pairs[code].append(pair_reverse)
                    except:
                        log("reverse pair failed kalman")
                bar.update(pair_counter+1)
                pair_counter += 1
    bar.finish()
    log ("Price test passed: {0}".format(counter))

    # CHOOSE PAIRS************************************************************************
    report = "CHOOSING " + str(context.desired_pairs) + " PAIR" + "S"*(context.desired_pairs > 1)
    
    new_pairs = []
    bar = progressbar.ProgressBar(maxval=counter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', '<',progressbar.Timer(),'>'])
    bar.start()
    count = 0
    for code in context.all_pairs:
        for pair in context.all_pairs[code]:
            pair.test(context, data, test_type="spread")
            bar.update(count+1)
            count += 1
        new_pairs = new_pairs + context.industries[code]['top']
    bar.finish()

    new_pairs = sorted(new_pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
    num_pairs = context.desired_pairs if (len(new_pairs) > context.desired_pairs) else len(new_pairs)
    context.desired_pairs = context.desired_pairs - num_pairs
    report += " --> FOUND " + str(num_pairs)
    for i in range(num_pairs):
        report += ("\n\t\t\t"+str(len(context.pairs)+1)+") "+str(new_pairs[i].to_string)
        +"\n\t\t\t\tIndustry Code:\t"+ str(new_pairs[i].industry))
        for test in new_pairs[i].latest_test_results:
            report += "\n\t\t\t\t" + str(test) + ": \t" + "\t"*(len(test) <= 5 ) + str(new_pairs[i].latest_test_results[test])
        context.pairs.append(new_pairs[i])
        context.pairs_chosen = True
    
    num_spreads = context.spread.shape[1]
    for index in range(num_pairs):
        new_spreads = np.ndarray((1, num_spreads))
        for i in range(num_spreads):
            diff = num_spreads-LOOKBACK
            new_spreads[0][i] = new_pairs[index].spreads[-LOOKBACK:][i-diff] if i >= diff else float("nan")
        context.spread = np.vstack((context.spread, new_spreads))
    log (report)

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
            log (pair.to_string + " failed stoploss --> X")
            if context.target_weights[pair.left.equity] != 0 or context.target_weights[pair.right.equity] != 0:
                sell_pair(context, data, pair)
            remove_pair(context, pair, index=i)
            i = i-1
        is_tradable = pair.is_tradable(data)
        if not is_tradable[0]:
            log ("cannot trade  " + str(pair.left.equity) + " & " + str(pair.right.equity))
            if context.target_weights[pair.left.equity] != 0 or context.target_weights[pair.right.equity] != 0:
                sell_pair(context, data, pair)
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
        log("Stock not liquidated. Returning")
        return

    num_pairs = len(context.pairs)
    pair_index = 0
    new_spreads = np.ndarray((num_pairs, 1))
    while (pair_index < num_pairs):
        if (pair_index >= len(context.pairs)):
            break
        pair = context.pairs[pair_index]
        (s1, s2) = (pair.left, pair.right)
        (s1_price_test, s2_price_test) = (data.history(s1.equity, "price", LOOKBACK+HEDGE_LOOKBACK, '1d'), data.history(s2.equity, "price", LOOKBACK+HEDGE_LOOKBACK, '1d'))
        pair.left.price_history = s1_price_test.values.tolist()
        pair.left.filtered_price_history = run_kalman(pair.left.price_history)
        pair.right.price_history = s2_price_test.values.tolist()
        pair.right.filtered_price_history = run_kalman(pair.right.price_history)

        result= pair.test(context,data,loose_screens=True, test_type="price")
        if result:
            pair.spreads = get_spreads(data, pair.left.price_history, pair.right.price_history, LOOKBACK)
            try:
                pair.filtered_spreads = run_kalman(pair.spreads)
            except:
                log("cps")
            result = pair.test(context,data,loose_screens=True)
        if not result:
            log(pair.to_string + " failed tests --> X " + str(pair.latest_failed_test))
            if context.target_weights[s1.equity] != 0 or context.target_weights[s2.equity] != 0:
                sell_pair(context, data, pair)
            remove_pair(context, pair, index=pair_index)
            new_spreads = np.delete(new_spreads, pair_index, 0)
            continue
        slope, intercept = linreg(np.log(pair.right.price_history[-HEDGE_LOOKBACK:]),np.log(pair.left.price_history[-HEDGE_LOOKBACK:]))
        new_spreads[pair_index, :] = np.log(pair.left.price_history[-1]) -  slope * np.log(pair.right.price_history[-1]) - intercept
        
        spreads = context.spread[pair_index, -context.spread.shape[1]:]
        spreads = np.array([val for val in spreads if (not np.isnan(val))])
        context.pairs[pair_index].spreads = spreads
        zscore = (spreads[-1] - spreads.mean()) / spreads.std()

        if (pair.currently_short and zscore < EXIT) or (pair.currently_long and zscore > -EXIT):     
            sell_pair(context, data, pair)
        elif pair.currently_short:
            y_target_shares = -1
            X_target_shares = slope
            buy_pair(context, data, pair, y_target_shares, X_target_shares, pair.left.price_history, pair.right.price_history, new_pair=False)
        elif pair.currently_long:
            y_target_shares = 1    
            X_target_shares = -slope
            buy_pair(context, data, pair, y_target_shares, X_target_shares, pair.left.price_history, pair.right.price_history, new_pair=False)

        if zscore < -ENTRY and (not pair.currently_long):
            y_target_shares = 1
            X_target_shares = -slope
            buy_pair(context, data, pair, y_target_shares, X_target_shares, pair.left.price_history, pair.right.price_history, new_pair=True)

        if zscore > ENTRY and (not pair.currently_short):
            y_target_shares = -1
            X_target_shares = slope
            buy_pair(context, data, pair, y_target_shares, X_target_shares, pair.left.price_history, pair.right.price_history, new_pair=True)

        pair_index = pair_index+1

    context.spread = np.hstack([context.spread, new_spreads])
    allocate(context, data)

def run_kalman(price_history):
    kf_stock = KalmanFilter(transition_matrices = [1], observation_matrices = [1], initial_state_mean = price_history[0], 
                            initial_state_covariance = 1, observation_covariance=1, transition_covariance=.05)

    try:
        filtered_prices = kf_stock.smooth(price_history)[0].flatten()
    except:
        filtered_prices = kf_stock.smooth(price_history)[0].flatten()
    return filtered_prices

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
            table += ("\n\t\t\t|\t"+str(pair.left.name)+"\t"+ "\t"*(1 if len(pair.left.name) < 8 else 0) + "+"*(1 if context.target_weights[pair.left.equity] >= 0 else 0) 
                  +('%.1f' % round(context.target_weights[pair.left.equity]*100,1))+"%  "+"+"*(1 if context.target_weights[pair.right.equity] >= 0 else 0)
                  +('%.1f' % round(context.target_weights[pair.right.equity]*100,1))+"%\t     "+str(pair.right.name) + "\t\t|")
        order_target_percent(pair.left.equity, context.target_weights[pair.left.equity])
        order_target_percent(pair.right.equity, context.target_weights[pair.right.equity])
    table = table if len(table) > 0 else "\n\t\t\t|\t\t\tall weights 0\t\t\t\t|"
    log ("ALLOCATING...\n\t\t\t " + "_"*63 + table + "\n\t\t\t|" + "_"*63 + "|")

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
    context.desired_pairs += 1

def get_spreads(data, s1_price, s2_price, length):
    spreads = []
    for i in range(length):
        start_index = len(s1_price)-length+i
        hedge, intercept = linreg(np.log(s2_price[start_index-HEDGE_LOOKBACK:start_index]),np.log(s1_price[start_index-HEDGE_LOOKBACK:start_index]))
        spreads = np.append(spreads, np.log(s1_price[i]) - hedge*np.log(s2_price[i])-intercept)
    return spreads

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
                log('Linear Regression Failed')
                slope = float('NaN')
                intercept = float('NaN')
    return slope, intercept

def get_test_by_name(name):
    
    def correlation(a,b):
        r, p = pearsonr(a,b)
        if p<DESIRED_PVALUE:
            return r
        else:
            return float('NaN')
        return r
    
    def cointegration(s1_price, s2_price):
        score, pvalue, _ = sm.coint(s1_price, s2_price)
        return pvalue
    
    def adf_pvalue(spreads):
        return sm.adfuller(spreads,autolag='t-stat')[1]
    
    def hurst_hvalue(series):
        
        #From Ernie Chan
        
        # tau, lagvec = [], []
        # for lag in range(2,20):  
        #     pp = np.subtract(series[lag:],series[:-lag])
        #     lagvec.append(lag)
        #     tau.append(np.sqrt(np.std(pp)))
        # slope, intercept = linreg(np.log10(lagvec),np.log10(tau))
        # hurst = slope*2
        # return hurst
        
        #From Hurst Package
        
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

                # SIMPLIFIED
                R = max(series[start:start+w]) - min(series[start:start+w])  # range in absolute values
                S = np.std(incs, ddof=1)
                #NOT SIMPLIFIED
                # mean_inc = (series[start:start+w][-1] - series[start:start+w][0]) / len(incs)
                # deviations = incs - mean_inc
                # Z = np.cumsum(deviations)
                # R = max(Z) - min(Z)
                # S = np.std(incs, ddof=1)

                if R != 0 and S != 0:
                    rs.append(R/S)
            RS.append(np.mean(rs))
        A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
        H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
        return H
    
    def half_life(spreads): 
        lag = np.roll(spreads, 1)
        ret = spreads - lag
        slope, intercept = linreg(lag,ret)
        return(-np.log(2) / slope)
    
    def shapiro_pvalue(spreads):
        w, p = shapiro(spreads)
        return p
    
    def adf_prices(s1_price, s2_price):
        p1 = sm.adfuller(s1_price, autolag='t-stat')[1]
        p2 = sm.adfuller(s2_price, autolag='t-stat')[1]
        return min(p1,p2)
    
    def jb_pvalue(spreads):
        w, p = jarque_bera(spreads)
        return p
    
    def zscore(spreads):
        return abs((spreads[-1]-spreads.mean())/spreads.std())
    
    def alpha(price1, price2):
        slope, intercept = linreg(np.log(price2), np.log(price1))
        y_target_shares = 1
        x_target_shares = -slope
        notionalDol =  abs(y_target_shares * price1[-1]) + abs(x_target_shares * price2[-1])
        (y_target_pct, x_target_pct) = (y_target_shares * price1[-1] / notionalDol, x_target_shares * price2[-1] / notionalDol)
        if (min (abs(x_target_pct),abs(y_target_pct)) > MIN_WEIGHT):
            return slope
        else:
            return float('NaN')
    
    def ljung_box(spreads):
        return max(sd.acorr_ljungbox(spreads)[1])
    
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
    elif (name.lower() == "ljung-box"):
        return ljung_box
    elif (name.lower() == "jarque-bera"):
        return jb_pvalue
    elif (name.lower() == "adf-prices"):
        return adf_prices

    return default