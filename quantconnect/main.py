import copy
from itertools import groupby
from math import ceil
import numpy as np
from statlib import StatsLibrary
from params import *

class PairsTrader(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(START_YEAR, START_MONTH, START_DAY)
        self.SetEndDate(END_YEAR, END_MONTH, END_DAY)
        self.SetCash(INITIAL_PORTFOLIO_VALUE)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.curr_month, self.last_month = -1, -1
        self.industries, self.pairs = [], []
        self.industry_map, self.dv_by_symbol = {}, {}
        
        self.library = StatsLibrary(hedge_lookback=HEDGE_LOOKBACK, min_weight=MIN_WEIGHT)
        self.strict_tester = PairTester(config=TEST_PARAMS, library=self.library)
        self.loose_tester = PairTester(config=LOOSE_PARAMS, library=self.library)
        self.weight_mgr = WeightManager(max_pair_weight=MAX_PAIR_WEIGHT)
        self.AddUniverse(self.select_coarse, self.select_fine)
        self.Schedule.On(self.DateRules.MonthStart(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 5), Action(self.choose_pairs))
        self.Schedule.On(self.DateRules.EveryDay(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 5), Action(self.check_pair_status))
    
    def choose_pairs(self):
        if not self.reset_vars():
            return
        self.Log("Creating Industries...")
        self.industries = self.create_industries()
        sizes = "".join([(("\n\t\t\t" if i%3 == 0 else "\t") + str(self.industries[i])) for i in range(len(self.industries))])
        self.Log(sizes + "\n\n\t\t\tTotal Tickers = {0}\n\t\t\tTotal Pairs = {1}".format(sum([i.size() for i in self.industries]), sum([len(i.get_pairs()) for i in self.industries])))

        # Testing pairs
        pairs = [pair for industry in self.industries for pair in industry.get_pairs()]
        pairs = [pair for pair in pairs if self.strict_tester.test_pair(pair, spreads=False)]
        self.Log("Price Testing Pairs...\n\t\t\t{0}".format(self.strict_tester))
        for pair in pairs:
            spreads, _ = self.library.get_spreads(pair.left.ph, pair.right.ph, self.true_lookback-(2*HEDGE_LOOKBACK))
            spreads_raw, residuals = self.library.get_spreads(pair.left.ph_raw, pair.right.ph_raw, self.true_lookback-(2*HEDGE_LOOKBACK))
            pair.spreads, pair.spreads_raw, pair.latest_residuals = spreads, spreads_raw, residuals
        self.strict_tester.reset()
        pairs = [pair for pair in pairs if self.strict_tester.test_pair(pair, spreads=True)]
        self.Log("Spread Testing Pairs...\n\t\t\t{0}".format(self.strict_tester))

        # Sorting and removing overlapping pairs
        pairs = sorted(pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)
        for pair in pairs:
            if not any((p.contains(pair.left.id) or p.contains(pair.right.id)) for p in self.pairs):
                self.pairs.append(pair)
        self.pairs = self.pairs[:min(len(self.pairs), self.desired_pairs)]
        self.weight_mgr.add_pairs(self.pairs)
        self.Log("Pair List" + "".join(["\n\t{0}) {1} {2}".format(i+1, self.pairs[i], self.pairs[i].formatted_results()) for i in range(len(self.pairs))]))

    def check_pair_status(self):
        if self.pairs == []:
            return
        
        # Check validity
        for pair in list(self.pairs):
            if not self.loose_tester.test_stoploss(pair):
                self.Log("Removing {0}. Failed stoploss. \n\t\t\t{1}: {2}\n\t\t\t{3}: {4}".format(pair, pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.pairs.remove(pair)
                self.weight_mgr.zero(pair)
            if not self.istradable(pair):
                self.Log("Removing {0}. Not Tradable. \n\t\t\t{1}: {2}\n\t\t\t{3}: {4}".format(pair, pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.pairs.remove(pair)
                self.weight_mgr.zero(pair)
                self.weight_mgr.delete([pair.left.id, pair.right.id])

        # Run loose tests
        for pair in list(self.pairs):
            pair.left.ph_raw, pair.right.ph_raw = self.daily_close(pair.left.ticker, LOOKBACK+2*int(np.ceil(HEDGE_LOOKBACK*7/5))), self.daily_close(pair.right.ticker, LOOKBACK+2*int(np.ceil(HEDGE_LOOKBACK*7/5)))
            pair.left.ph, pair.right.ph = self.library.run_kalman(pair.left.ph_raw), self.library.run_kalman(pair.right.ph_raw)
            pair.latest_test_results.clear()
            passed = self.loose_tester.test_pair(pair, spreads=False)
            if passed:
                spreads, _ = self.library.get_spreads(pair.left.ph, pair.right.ph, self.true_lookback-(2*HEDGE_LOOKBACK))
                pair.spreads = spreads
                passed = self.loose_tester.test_pair(pair, spreads=True)
            if not passed:
                self.Log("Removing {0}. Failed tests.\n\t\t\tResults:{1} \n\t\t\t{2}: {3}\n\t\t\t{4}: {5}".format(pair, pair.formatted_results(), pair.left.ticker, pair.left.purchase_info(), pair.right.ticker, pair.right.purchase_info()))
                self.pairs.remove(pair)
                self.weight_mgr.zero(pair)
                continue
                
            slope, intercept = self.library.linreg(pair.right.ph_raw[-HEDGE_LOOKBACK:], pair.left.ph_raw[-HEDGE_LOOKBACK:])
            current_residual = pair.left.ph_raw[-1] - slope*pair.right.ph_raw[-1] + intercept
            pair.latest_residuals = np.delete(pair.latest_residuals, 0)
            pair.latest_residuals = np.append(pair.latest_residuals, current_residual)
            std = np.std(pair.latest_residuals)
            avg = np.mean(pair.latest_residuals)
            zscore = (current_residual-avg)/std
            pair.spreads_raw = np.append(pair.spreads_raw, zscore)
            
            if (pair.currently_short and zscore < EXIT) or (pair.currently_long and zscore > -EXIT):   
                self.weight_mgr.zero(pair)
            elif (zscore > ENTRY and (not pair.currently_short)):
                self.weight_mgr.assign(pair=pair, y_target_shares=-1, X_target_shares=slope)
            elif (zscore < -ENTRY and (not pair.currently_long)):
                self.weight_mgr.assign(pair=pair, y_target_shares=1, X_target_shares=-slope)

        # Place orders
        if self.weight_mgr.isupdated():
            weights = self.weight_mgr.get_weights()
            for pair in self.pairs:
                self.SetHoldings(pair.left.ticker, weights[pair.left.id])
                self.SetHoldings(pair.right.ticker, weights[pair.right.id])
            self.Log("ALLOCATING\n\t\t\t{0}".format(self.weight_mgr.__str__(self.pairs)))
            self.weight_mgr.updated = False
    
    def create_industries(self):
        if RUN_TEST_STOCKS:
            self.industry_map = {123: TEST_STOCKS}
        
        industries = []    
        for code in self.industry_map:
            industry = Industry(code)
            for ticker in self.industry_map[code]:
                equity = self.AddEquity(ticker)
                price_history = self.daily_close(ticker, LOOKBACK+2*int(np.ceil(HEDGE_LOOKBACK*7/5)))
                if (len(price_history) >= self.true_lookback):
                    stock = Stock(ticker=ticker, id=equity.Symbol.ID.ToString())
                    stock.ph_raw = price_history
                    stock.ph = self.library.run_kalman(stock.ph_raw)
                    industry.add_stock(stock)
            if industry.size() > 1:
                industry.create_pairs()
                industries.append(industry)
        return sorted(industries, key=lambda x: x.size(), reverse=False)
        
    def reset_vars(self):
        this_month = self.Time.month 
        if self.curr_month < 0:
            self.curr_month = this_month
        self.next_month = self.curr_month + INTERVAL - 12*(self.curr_month + INTERVAL > 12)
        if (this_month != self.curr_month):
            return False
        self.curr_month = self.next_month
        self.industries.clear()
        self.pairs.clear()
        self.weight_mgr.reset()
        self.strict_tester.reset()
        self.loose_tester.reset()
        self.Liquidate()
        
        self.true_lookback = len(self.daily_close("SPY", LOOKBACK + 2*int(np.ceil(HEDGE_LOOKBACK*7/5))))
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
        
    def select_coarse(self, coarse):
        self.industry_map.clear()
        if self.Time.month == self.last_month:
            return Universe.Unchanged

        sortedByDollarVolume = sorted([x for x in coarse if x.HasFundamentalData and x.Volume > MIN_VOLUME and x.Price > MIN_SHARE and x.Price < MAX_SHARE], key = lambda x: x.DollarVolume, reverse=True)[:COARSE_LIMIT]
        self.dv_by_symbol = {x.Symbol:x.DollarVolume for x in sortedByDollarVolume}
        if len(self.dv_by_symbol) == 0:
            return Universe.Unchanged

        return list(self.dv_by_symbol.keys())
        
    def select_fine(self, fine):
        sortedBySector = sorted([x for x in fine if x.CompanyReference.CountryId == "USA"
                                        and x.CompanyReference.PrimaryExchangeID in ["NYS","NAS"]
                                        and (self.Time - x.SecurityReference.IPODate).days > MIN_AGE
                                        and x.AssetClassification.MorningstarSectorCode != MorningstarSectorCode.FinancialServices
                                        # and x.AssetClassification.MorningstarSectorCode != MorningstarSectorCode.Utilities
                                        # and x.AssetClassification.MorningstarIndustryGroupCode != MorningstarIndustryGroupCode.MetalsAndMining
                                        and MKTCAP_MIN < x.MarketCap 
                                        and x.MarketCap < MKTCAP_MAX
                                        and not x.SecurityReference.IsDepositaryReceipt],
                               key = lambda x: x.CompanyReference.IndustryTemplateCode)

        count = len(sortedBySector)
        if count == 0:
            return Universe.Unchanged

        self.last_month = self.Time.month
        percent = FINE_LIMIT / count
        sortedByDollarVolume = []

        for code, g in groupby(sortedBySector, lambda x: x.CompanyReference.IndustryTemplateCode):
            y = sorted(g, key = lambda x: self.dv_by_symbol[x.Symbol], reverse = True)
            c = ceil(len(y) * percent)
            sortedByDollarVolume.extend(y[:c])

        sortedByDollarVolume = sorted(sortedByDollarVolume, key = lambda x: self.dv_by_symbol[x.Symbol], reverse=True)
        final_securities = sortedByDollarVolume[:FINE_LIMIT]
        for x in final_securities:
            if not (x.AssetClassification.MorningstarIndustryCode in self.industry_map):
                self.industry_map[x.AssetClassification.MorningstarIndustryCode] = []
            self.industry_map[x.AssetClassification.MorningstarIndustryCode].append(x.Symbol.Value)
        return [x.Symbol for x in final_securities]
        
#####################
# CLASS DEFINITIONS #
####################################################################################################
class WeightManager:
    
    def __init__(self, max_pair_weight):
        self.weights = {}
        self.num_allocated = 0        
        self.max_pair_weight = max_pair_weight
        self.updated = False
        
    def __str__(self, pairs):
        lines = []
        for pair in pairs:
            lines.append("{0}\t{1}\t{2}".format(pair, f"{round(self.weights[pair.left.id],2):+g}", f"{round(self.weights[pair.right.id],2):+g}"))
        return "\n\t\t\t".join(lines)
    
    def reset(self):
        self.weights.clear()
        self.num_allocated = 0
        
    def isupdated(self):
        return self.updated
    
    def add_pairs(self, pairs):
        for pair in pairs:
            self.weights[pair.left.id] = 0.0
            self.weights[pair.right.id] = 0.0
    
    def get_weights(self):
        return self.weights
    
    def delete(self, items):
        for item in items:
            del self.weights[item]
        
    def zero(self, pair):
        if (self.weights[pair.left.id] == 0) and (self.weights[pair.right.id] == 0):
            return
        self.weights[pair.left.id] = 0.0
        self.weights[pair.right.id] = 0.0
        pair.left.update_purchase_info(0, False)
        pair.right.update_purchase_info(0, False)
        pair.currently_short, pair.currently_long = False, False
        if (self.num_allocated/2) > (1/self.max_pair_weight):
            self.scale_keys(self.num_allocated/(self.num_allocated-2))
        self.num_allocated = self.num_allocated - 2
        self.updated = True
        
    def assign(self, pair, y_target_shares, X_target_shares):
        notionalDol =  abs(y_target_shares * pair.left.ph_raw[-1]) + abs(X_target_shares * pair.right.ph_raw[-1])
        (y_target_pct, x_target_pct) = (y_target_shares * pair.left.ph_raw[-1] / notionalDol, X_target_shares * pair.right.ph_raw[-1] / notionalDol)
        if EQUAL_WEIGHTS:
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
        self.updated = True
    
    def calculate_weights(self, pair, y_target_pct, x_target_pct, factor, new=True):
        if (self.num_allocated/2) < (1/self.max_pair_weight):
            self.weights[pair.left.id] = y_target_pct * self.max_pair_weight
            self.weights[pair.right.id] = x_target_pct * self.max_pair_weight
        else:
            if new:
                self.scale_keys(self.num_allocated/(self.num_allocated+2))
            self.weights[pair.left.id] =  y_target_pct * factor
            self.weights[pair.right.id] =  x_target_pct * factor
    
    def scale_keys(self, factor):
        for key in self.weights:
            self.weights[key] = self.weights[key]*factor

class Stock:
    
    def __init__(self, ticker, id):
        self.ticker = ticker
        self.id = id
        self.ph_raw = []
        self.ph = []
        self.purchase_price = 0
        self.long = False
        
    def __str__(self):
        return "{1}".format(self.id)
        
    def purchase_info(self):
        return {"long": self.long, "purchase": self.purchase_price, "current": self.ph_raw[-1]}

    def update_purchase_info(self, price, is_long):
        self.purchase_price = price
        self.long = is_long

class Pair:
    
    def __init__(self, s1, s2, industry):
        self.left, self.right = s1, s2
        self.spreads, self.spreads_raw = [], []
        self.industry = industry
        self.latest_residuals = []
        self.latest_test_results = {}
        self.currently_long, self.currently_short = False, False
    
    def __str__(self):
        pair = "([{0} & [{1}])".format(self.left.ticker, self.right.ticker)
        return "{0}{1}".format(self.industry.code, pair.rjust(18))
        
    def formatted_results(self):
        results = {}
        for key in self.latest_test_results:
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
        
    def get_pairs(self):
        return self.pairs
    
    def add_stock(self, stock):
        self.stocks.append(stock)
        
    def create_pairs(self):
        for i in range(len(self.stocks)):
            for j in range(i+1, len(self.stocks)):
                pair1 = Pair(self.stocks[i], self.stocks[j], self)
                pair2 = Pair(self.stocks[j], self.stocks[i], self)
                self.pairs.extend([pair1, pair2])
        
class PairTester:
    
    def __init__(self, config, library):
        self.price_tests = [name for name in config if ((not config[name]['spreads']) and config[name]['run'])]
        self.spread_tests = [name for name in config if (config[name]['spreads'] and config[name]['run'])]
        self.config = config
        self.library = library
        self.count = 0
        self.failures = {}
        
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
                if test == "HalfLife":
                    result = test_function(pair.spreads[-HEDGE_LOOKBACK:])
                elif test == "ZScore":
                    result = abs(pair.spreads_raw[-1])
                elif test == "Alpha":
                    result = test_function(pair.left.ph_raw[-HEDGE_LOOKBACK:], pair.right.ph_raw[-HEDGE_LOOKBACK:])
                elif test == "ADFPrices":
                    result = test_function(pair.left.ph, pair.right.ph_raw)    
                elif spreads:
                    result = test_function(pair.spreads)
                else:
                    result = test_function(pair.left.ph[-HEDGE_LOOKBACK:], pair.right.ph[-HEDGE_LOOKBACK:])
            except:
                pass
            
            if (not result) or (not self.test_value_bounds(test, result)):
                self.failures[test] = self.failures.get(test, 0) + 1
                return False
            pair.latest_test_results[test] = round(result, 5)
        return True
    
    def test_value_bounds(self, test, result):
        return (result >= self.config[test]['min'] and result <= self.config[test]['max'])
        
    def test_stoploss(self, pair):
        if (pair.left.purchase_price) == 0 and (pair.right.purchase_price == 0):
            return True
        left_fail = (pair.left.long and pair.left.ph[-1] < (1-STOPLOSS)*pair.left.purchase_price) or (not pair.left.long and pair.left.ph[-1] > (1+STOPLOSS)*pair.left.purchase_price)
        right_fail = (pair.right.long and pair.right.ph[-1] < (1-STOPLOSS)*pair.right.purchase_price) or (not pair.right.long and pair.right.ph[-1] > (1+STOPLOSS)*pair.right.purchase_price)
        return not (left_fail or right_fail)