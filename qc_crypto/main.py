import copy
from itertools import groupby
from math import ceil
import numpy as np
from statlib import StatsLibrary
import scipy.stats as ss
from params import *
 
class PairsTrader(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(ST_Y, ST_M, ST_D)
        self.SetEndDate(END_Y, END_M, END_D)
        self.SetCash(INITIAL_PORTFOLIO_VALUE)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetBenchmark("SPY")
        self.last_month = -1
        self.industries= []
        self.industry_map = {}
        self.interval = 1
        
        hedge_input = self.GetParameter("hedge-lookback")
        self.hedge_lookback = 15 if hedge_input is None else int(hedge_input)
        entry_input = self.GetParameter("entry-input")
        self.entry = 2.00 if entry_input is None else float(entry_input)
        
        self.library = StatsLibrary(hedge_lookback=self.hedge_lookback)
        self.strict_tester = PairTester(config=TEST_PARAMS, library=self.library, hedge_lookback=self.hedge_lookback)
        self.loose_tester = PairTester(config=LOOSE_PARAMS, library=self.library, hedge_lookback=self.hedge_lookback)
        self.Schedule.On(self.DateRules.EveryDay(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 1), Action(self.check_date))
        self.Schedule.On(self.DateRules.EveryDay(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 5), Action(self.choose_pairs))
        self.Schedule.On(self.DateRules.EveryDay(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 35), Action(self.check_pair_status))
    
    def OnData(self, data):
        pass
    
    def choose_pairs(self):
        if not self.industry_map:
            return
        
        self.reset_vars()
        self.Log("Creating Industries...")
        self.industries = self.create_industries()
        self.industry_map.clear()
        sizes = "".join([(("\n\t\t\t" if i%3 == 0 else "\t") + str(self.industries[i])) for i in range(len(self.industries))])
        self.Log(sizes + "\n\n\t\t\tTotal Tickers = {0}\n\t\t\tTotal Pairs = {1}".format(sum([i.size() for i in self.industries]), sum([len(i.pairs) for i in self.industries])))

        # Testing pairs
        pairs = [pair for industry in self.industries for pair in industry.pairs]
        if SIMPLE_SPREADS:
            pairs = [pair for pair in pairs if (self.strict_tester.test_pair(pair, spreads=False) and self.strict_tester.test_pair(pair.reverse_pair, spreads=False))]
        else:
            pairs = [pair for pair in pairs if self.strict_tester.test_pair(pair, spreads=False)]
        self.Log("Price Testing Pairs...\n\t\t\t{0}".format(self.strict_tester))
        for pair in pairs:
            pair.spreads = self.library.get_spreads(pair.left.ph, pair.right.ph, self.true_lookback-(self.hedge_lookback))
            pair.spreads_raw = self.library.get_spreads(pair.left.ph_raw, pair.right.ph_raw, self.true_lookback-(self.hedge_lookback))
        self.strict_tester.reset()
        pairs = [pair for pair in pairs if self.strict_tester.test_pair(pair, spreads=True)]
        self.Log("Spread Testing Pairs...\n\t\t\t{0}".format(self.strict_tester))
        if SIMPLE_SPREADS:
            self.strict_tester.reset()
            for pair in pairs:
                pair.reverse_pair.spreads = self.library.get_spreads(pair.reverse_pair.left.ph, pair.reverse_pair.right.ph, self.true_lookback-(self.hedge_lookback))
                pair.reverse_pair.spreads_raw = self.library.get_spreads(pair.reverse_pair.left.ph_raw, pair.reverse_pair.right.ph_raw, self.true_lookback-(self.hedge_lookback))
            pairs = [pair for pair in pairs if self.strict_tester.test_pair(pair.reverse_pair, spreads=True)]
            self.Log("Spread Testing Reverse Pairs...\n\t\t\t{0}".format(self.strict_tester))
            pairs.extend([pair.reverse_pair for pair in pairs])
            
        # Sorting and removing overlapping pairs
        pairs = sorted(pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
        final_pairs = []
        for pair in pairs:
            if not any((p.contains(pair.left.id) or p.contains(pair.right.id)) for p in final_pairs):
                final_pairs.append(pair)
        final_pairs = final_pairs[:min(len(final_pairs), self.desired_pairs)]
        self.Log("Pair List" + "".join(["\n\t{0}) {1} {2}".format(i+1, final_pairs[i], final_pairs[i].formatted_results()) for i in range(len(final_pairs))]))
        self.weight_mgr = WeightManager(max_pair_weight=MAX_PAIR_WEIGHT, pairs=final_pairs)

    def check_pair_status(self):
        if self.industries == []:
            return
        # Check validity
        for pair in list(self.weight_mgr.pairs):
            pair.left.ph_raw, pair.right.ph_raw = self.daily_close(pair.left.ticker, LOOKBACK+100), self.daily_close(pair.right.ticker, LOOKBACK+100)
            if self.weight_mgr.is_allocated(pair):
                pair.left.update_purchase_info(pair.left.ph_raw[-1], pair.left.long)
                pair.right.update_purchase_info(pair.right.ph_raw[-1], pair.right.long)
            if not self.loose_tester.test_stoploss(pair):
                self.Log("Removing {0}. Failed stoploss. \n\t\t\t{1}: {2}\n\t\t\t{3}: {4}".format(pair, pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.weight_mgr.zero(pair)
            if not self.istradable(pair):
                self.Log("Removing {0}. Not Tradable. \n\t\t\t{1}: {2}\n\t\t\t{3}: {4}".format(pair, pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.weight_mgr.zero(pair)
        # Run loose tests
        for pair in list(self.weight_mgr.pairs):
            pair.left.ph, pair.right.ph = self.library.run_kalman(pair.left.ph_raw), self.library.run_kalman(pair.right.ph_raw)
            pair.latest_test_results.clear()
            passed = self.loose_tester.test_pair(pair, spreads=False)
            if passed:
                pair.spreads = self.library.get_spreads(pair.left.ph, pair.right.ph, self.true_lookback-(self.hedge_lookback))
                pair.spreads_raw = self.library.get_spreads(pair.left.ph_raw, pair.right.ph_raw, self.true_lookback-(self.hedge_lookback))
                passed = self.loose_tester.test_pair(pair, spreads=True)
            if not passed:
                self.Log("Removing {0}. Failed tests.\n\t\t\tResults:{1} \n\t\t\t{2}: {3}\n\t\t\t{4}: {5}".format(pair, pair.formatted_results(), pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.weight_mgr.zero(pair)
                continue
            
            # spread adjustments
            if RSI:
                latest_rsi = self.library.get_rsi(pair.spreads_raw, RSI_LOOKBACK)
            slope = 1
            if EWA:
                zscore = self.library.ewa(pair.spreads_raw[-self.hedge_lookback:])
            else:
                zscore = ss.zscore(pair.spreads_raw[-self.hedge_lookback:], nan_policy='omit')[-1]
            if not SIMPLE_SPREADS:
                slope, _ = self.library.linreg(pair.right.ph_raw[-self.hedge_lookback:], pair.left.ph_raw[-self.hedge_lookback:])
                zscore = ss.zscore(pair.spreads_raw[-self.hedge_lookback:], nan_policy='omit')[-1]
            
            # trading logic
            if (pair.currently_short and (zscore < EXIT or latest_rsi < RSI_EXIT or zscore > Z_STOP)) or (pair.currently_long and (zscore > -EXIT or latest_rsi > -RSI_EXIT or zscore < -Z_STOP)):   
                self.weight_mgr.zero(pair)
            elif (self.Time.day < DAY_CUTOFF) and (zscore > self.entry and (not pair.currently_short)) and (self.weight_mgr.num_allocated/2 < MAX_ACTIVE_PAIRS) and latest_rsi>RSI_THRESHOLD:
                if CHECK_DOWNTICK:
                    pair.short_dt, pair.long_dt = True, False
                else:
                    self.weight_mgr.assign(pair=pair, y_target_shares=-1, X_target_shares=slope)
            elif (self.Time.day < DAY_CUTOFF) and (zscore < -self.entry and (not pair.currently_long)) and (self.weight_mgr.num_allocated/2 < MAX_ACTIVE_PAIRS) and latest_rsi<-RSI_THRESHOLD:
                if CHECK_DOWNTICK:
                    pair.long_dt, pair.short_dt = True, False
                else:
                    self.weight_mgr.assign(pair=pair, y_target_shares=1, X_target_shares=-slope)
            
            if (self.Time.day < DAY_CUTOFF) and CHECK_DOWNTICK and pair.short_dt and (zscore >= DOWNTICK) and (zscore <= self.entry) and (not pair.currently_short) and (self.weight_mgr.num_allocated/2 < MAX_ACTIVE_PAIRS):
                self.weight_mgr.assign(pair=pair, y_target_shares=-1, X_target_shares=slope)
            elif (self.Time.day < DAY_CUTOFF) and CHECK_DOWNTICK and pair.long_dt and (zscore <= -DOWNTICK) and (zscore >= -self.entry) and (not pair.currently_long) and (self.weight_mgr.num_allocated/2 < MAX_ACTIVE_PAIRS):
                self.weight_mgr.assign(pair=pair, y_target_shares=1, X_target_shares=-slope)


        # Place orders
        weights = self.weight_mgr.weights
        for pair in self.weight_mgr.updated:
            if weights[pair.left.id] > 0:
                self.SetHoldings(self.Symbol(pair.left.ticker), weights[pair.left.id])
            if weights[pair.right.id] > 0:
                self.SetHoldings(self.Symbol(pair.right.ticker), weights[pair.right.id])
        if len(self.weight_mgr.updated) > 0:
            self.Log("ALLOCATING\n\t\t\t{0}".format(self.weight_mgr))
        self.weight_mgr.reset()
    
    def create_industries(self):
        self.industry_map = {123: ['BTCUSD','LTCUSD', 'ETHUSD', 'ETCUSD']}
        tickers = [ticker for code in self.industry_map for ticker in self.industry_map[code]]
        for ticker in tickers:
            self.AddCrypto(ticker, Resolution.Daily)
        price_df = self.History([self.Symbol(ticker) for ticker in tickers], TimeSpan.FromDays(LOOKBACK+100), Resolution.Daily)
        
        industries = []    
        for code in self.industry_map:
            industry = Industry(code)
            for ticker in self.industry_map[code]:
                equity = self.Symbol(ticker)
                try:
                    price_history = price_df.loc[ticker]['close'].values.tolist()
                    if (len(price_history) >= self.true_lookback):
                        stock = Stock(ticker=ticker, id=equity.ID.ToString())
                        stock.ph_raw = price_history
                        stock.ph = self.library.run_kalman(stock.ph_raw)
                        industry.add_stock(stock)
                except:
                    pass
            if industry.size() > 1:
                industry.create_pairs(allow_reverse=(not SIMPLE_SPREADS))
                industries.append(industry)
        return sorted(industries, key=lambda x: x.size(), reverse=False)
        
    def reset_vars(self):
        self.industries.clear()
        self.strict_tester.reset()
        self.loose_tester.reset()
        self.Liquidate()
        
        self.true_lookback = len(self.daily_close("SPY", LOOKBACK + int(np.ceil(self.hedge_lookback*7/5))))
        self.desired_pairs = int(round(DESIRED_PAIRS * (self.Portfolio.TotalPortfolioValue / INITIAL_PORTFOLIO_VALUE)))
        return True
        
    def istradable(self, pair):
        left = self.Securities[self.Symbol(pair.left.ticker)].IsTradable
        right = self.Securities[self.Symbol(pair.right.ticker)].IsTradable
        return (left and right)
    
    def daily_close(self, ticker, length):
        history = []
        try:
            history = self.History(self.Symbol(ticker), TimeSpan.FromDays(length), Resolution.Daily).close.values.tolist()
        except:
            pass
        return history
        
    def check_date(self):
        if (self.last_month >= 0) and ((self.Time.month - 1) != ((self.last_month-1+self.interval+12) % 12)):
            return
        self.last_month = self.Time.month
        self.industry_map.clear()
        self.industry_map = {'Crypto': CRYPTO_TICKERS}
        
#####################
# CLASS DEFINITIONS #
####################################################################################################
class WeightManager:
    
    def __init__(self, max_pair_weight, pairs):
        self.weights = {}
        self.num_allocated = 0        
        self.max_pair_weight = max_pair_weight
        self.pairs = pairs
        self.updated = set()
        
        for pair in pairs:
            self.weights[pair.left.id] = 0.0
            self.weights[pair.right.id] = 0.0
        
    def __str__(self):
        lines = []
        for pair in self.pairs:
            lines.append("{0}\t{1}\t{2}\t{3}".format(pair, f"{round(self.weights[pair.left.id],2):+g}", f"{round(self.weights[pair.right.id],2):+g}", "U"*(pair in self.updated)))
        return "\n\t\t\t".join(lines)
    
    def reset(self):
        for pair in self.updated:
            if self.weights[pair.left.id] == 0 and self.weights[pair.right.id] == 0:
                del self.weights[pair.left.id]
                del self.weights[pair.right.id]
                self.pairs.remove(pair)
        self.updated.clear()
    
    def is_allocated(self, pair):
        return (self.weights[pair.left.id] != 0 and self.weights[pair.right.id] != 0)
    
    def zero(self, pair):
        self.updated.add(pair)
        if (self.weights[pair.left.id] == 0) and (self.weights[pair.right.id] == 0):
            return
        self.weights[pair.left.id] = 0.0
        self.weights[pair.right.id] = 0.0
        pair.left.update_purchase_info(0, False)
        pair.right.update_purchase_info(0, False)
        pair.currently_short, pair.currently_long = False, False
        pair.long_dt, pair.short_dt = False, False
        if (self.num_allocated/2) > (1/self.max_pair_weight):
            self.scale_keys(self.num_allocated/(self.num_allocated-2))
        self.num_allocated = self.num_allocated - 2
        
    def assign(self, pair, y_target_shares, X_target_shares):
        notionalDol =  abs(y_target_shares * pair.left.ph_raw[-1]) + abs(X_target_shares * pair.right.ph_raw[-1])
        (y_target_pct, x_target_pct) = (y_target_shares * pair.left.ph_raw[-1] / notionalDol, X_target_shares * pair.right.ph_raw[-1] / notionalDol)
        if SIMPLE_SPREADS:
            if x_target_pct<0:
                x_target_pct = -0.5
                y_target_pct = 0.5
            else:
                x_target_pct = 0.5
                y_target_pct = -0.5
        pair.currently_short = (y_target_pct < 0)
        pair.currently_long = (y_target_pct > 0)
        if (self.weights[pair.left.id] == 0) and (self.weights[pair.right.id] == 0):
            self.calculate_weights(pair, y_target_pct, x_target_pct, factor=2/(self.num_allocated+2), new=True)
            pair.left.update_purchase_info(pair.left.ph_raw[-1], y_target_pct > 0)
            pair.right.update_purchase_info(pair.right.ph_raw[-1], x_target_pct > 0)
            self.num_allocated = self.num_allocated+2
        else:
            self.calculate_weights(pair, y_target_pct, x_target_pct, factor=2/self.num_allocated, new=False)
        self.updated.add(pair)
    
    def calculate_weights(self, pair, y_target_pct, x_target_pct, factor, new=True):
        if (self.num_allocated/2) < (1/self.max_pair_weight):
            self.weights[pair.left.id] = y_target_pct * self.max_pair_weight
            self.weights[pair.right.id] = x_target_pct * self.max_pair_weight
        else:
            if new:
                self.scale_keys(self.num_allocated/(self.num_allocated+2))
            self.weights[pair.left.id] =  y_target_pct * factor
            self.weights[pair.right.id] =  x_target_pct * factor
    
    def get_pair_from_id(self, id):
        for pair in self.pairs:
            if id == pair.left.id or id == pair.right.id:
                return pair
    
    def scale_keys(self, factor):
        for key in self.weights:
            if self.weights[key] != 0:
                self.weights[key] = self.weights[key]*factor
                self.updated.add(self.get_pair_from_id(key))

class Stock:
    
    def __init__(self, ticker, id):
        self.ticker = ticker
        self.id = id
        self.ph_raw = []
        self.ph = []
        self.stop_price = 0
        self.long = False
        
    def __str__(self):
        return "{1}".format(self.id)
        
    def purchase_info(self):
        return {"long": self.long, "stop price": self.stop_price, "current": self.ph_raw[-1]}

    def update_purchase_info(self, price, is_long):
        if (is_long and price > self.stop_price) or ((not is_long) and (price < self.stop_price or self.stop_price == 0)):
            self.stop_price = price
        self.long = is_long

class Pair:
    
    def __init__(self, s1, s2, industry):
        self.left, self.right = s1, s2
        self.reverse_pair = None
        self.spreads, self.spreads_raw = [], []
        self.industry = industry
        self.latest_test_results = {}
        self.currently_long, self.currently_short = False, False
        self.long_dt, self.short_dt = False, False
    
    def __str__(self):
        pair = "([{0} & [{1}])".format(self.left.ticker, self.right.ticker)
        return "{0}{1}".format(self.industry.code, pair.rjust(18))
        
    def formatted_results(self):
        results = {}
        for key in self.latest_test_results:
            if self.latest_test_results[key] == None:
                results[key] = "None"
            else:
                results[key] = "{:.4f}".format(self.latest_test_results[key])
        return results

    def contains(self, id):
        return (self.left.id == id) or (self.right.id == id)

class Industry:
    
    def __init__(self, code):
        self.code = code
        self.stocks = []
        self.pairs = []
    
    def __str__(self):
        return "Industry {0}: {1} tickers".format(self.code, self.size())
    
    def size(self):
        return len(self.stocks)
    
    def add_stock(self, stock):
        self.stocks.append(stock)
        
    def create_pairs(self, allow_reverse=True):
        for i in range(len(self.stocks)):
            for j in range(i+1, len(self.stocks)):
                pair1 = Pair(self.stocks[i], self.stocks[j], self)
                pair2 = Pair(self.stocks[j], self.stocks[i], self)
                pair1.reverse_pair = pair2
                pair2.reverse_pair = pair1
                self.pairs.append(pair1)
                if allow_reverse:
                    self.pairs.append(pair2)
        
class PairTester:
    
    def __init__(self, config, library, hedge_lookback):
        self.price_tests = [name for name in config if ((not config[name]['spreads']) and config[name]['run'])]
        self.spread_tests = [name for name in config if (config[name]['spreads'] and config[name]['run'])]
        self.config = config
        self.library = library
        self.count = 0
        self.failures = {}
        self.hedge_lookback = hedge_lookback
        
    def __str__(self):
        if self.count == 0:
            return "No pairs tested."
        passed = self.count-sum(self.failures.values())
        return "Pairs Passed: {0}. Tester Failure Report: {1}. Pass Rate: {2}%".format(passed, self.failures, round(100*passed/self.count, 2))
        
    def reset(self):
        self.count = 0
        self.failures.clear()
    
    def test_pair(self, pair, spreads=False):
        self.count += 1
        tests = self.spread_tests if spreads else self.price_tests
        for test in tests:
            result = None
            test_function = self.library.get_func_by_name(test.lower())
            try:
                # if test == "ZScore":
                #     result = test_function(pair.spreads_raw)
                if test == "Alpha":
                    result = test_function(pair.left.ph_raw[-self.hedge_lookback:], pair.right.ph_raw[-self.hedge_lookback:])
                # elif test == "ShapiroWilke":
                #     result = test_function(pair.spreads_raw)
                elif spreads:
                    result = test_function(pair.spreads)
                else:
                    result = test_function(pair.left.ph, pair.right.ph)
            except:
                pass
            
            if (not result) or (not self.test_value_bounds(test, result)):
                pair.latest_test_results[test] = result
                self.failures[test] = self.failures.get(test, 0) + 1
                return False
            pair.latest_test_results[test] = round(result, 5)
        return True
    
    def test_value_bounds(self, test, result):
        return (result >= self.config[test]['min'] and result <= self.config[test]['max'])
        
    def test_stoploss(self, pair):
        if (pair.left.stop_price) == 0 and (pair.right.stop_price == 0):
            return True
        left_fail = (pair.left.long and pair.left.ph[-1] < (1-STOPLOSS)*pair.left.stop_price) or (not pair.left.long and pair.left.ph[-1] > (1+STOPLOSS)*pair.left.stop_price)
        right_fail = (pair.right.long and pair.right.ph[-1] < (1-STOPLOSS)*pair.right.stop_price) or (not pair.right.long and pair.right.ph[-1] > (1+STOPLOSS)*pair.right.stop_price)
        return not (left_fail or right_fail)
