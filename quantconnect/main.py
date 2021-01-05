import copy
from itertools import groupby
from math import ceil
from statlib import StatsLibrary
from params import *

class PairsTrader(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 11, 25)
        self.SetCash(INITIAL_PORTFOLIO_VALUE)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.curr_month, self.last_month = -1, -1
        self.industries, self.pairs = [], []
        self.industry_map, self.target_weights, self.dv_by_symbol = {}, {}, {}
        
        self.library = StatsLibrary(hedge_lookback=HEDGE_LOOKBACK, min_weight=MIN_WEIGHT)
        self.strict_tester = PairTester(config=TEST_PARAMS, library=self.library)
        self.loose_tester = PairTester(config=LOOSE_PARAMS, library=self.library)
        self.AddUniverse(self.SelectCoarse, self.SelectFine)
        self.Schedule.On(self.DateRules.MonthStart(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 5), Action(self.choose_pairs))
    
    def choose_pairs(self):
        if not self.reset_vars():
            return
        self.Log("New interval start. Portfolio Value: {0}".format(self.Portfolio.TotalPortfolioValue))
        self.Log("Creating Industries...")
        self.industries = self.create_industries()
        report = ""
        comps, tickers = 0, 0
        for i in range(len(self.industries)):
            report += ("\n\t\t\t" if i%3 == 0 else "\t") + str(self.industries[i])
            comps += self.industries[i].size() * (self.industries[i].size() - 1)
            tickers += self.industries[i].size()
        self.Log(report + "\n\n\t\t\tTotal Tickers = {0}\n\t\t\tTotal Pairs = {1}".format(tickers, comps))
   
        self.Log("Price Testing Pairs...")
        all_pairs, failure_counts = [], {}
        for c in range(len(self.industries)):
            industry = self.industries[c]
            for i in range (industry.size()):
                for j in range (industry.size()):
                    if i == j:
                        continue
                    pair = Pair(industry.stocks[i], industry.stocks[j], industry)
                    if self.strict_tester.test_pair(pair, spreads=False):
                        spreads, _ = self.library.get_spreads(pair.left.price_history, pair.right.price_history, self.true_lookback-42)
                        pair.update_spreads(spreads)
                        all_pairs.append(pair)
        self.Log(self.strict_tester)
        
        self.strict_tester.reset()
        self.Log("Spread Testing Pairs...")
        for pair in all_pairs:
            if self.strict_tester.test_pair(pair, spreads=True):
                pair.industry.add_good_pair(pair)
        self.pairs = [p for i in self.industries for p in i.good_pairs]
        self.Log(self.strict_tester)
        
        num_pairs = min(len(self.pairs), self.desired_pairs)
        self.pairs = sorted(self.pairs, key=lambda x: x.latest_test_results[RANK_BY], reverse=RANK_DESCENDING)[:num_pairs]
        self.Log("Pair List" + "".join(["\n\t{0}) {1} {2}".format(i+1, self.pairs[i], self.pairs[i].results()) for i in range(num_pairs)]))
    
    def SelectCoarse(self, coarse):
        self.industry_map = {}
        if self.Time.month == self.last_month:
            return Universe.Unchanged

        sortedByDollarVolume = sorted([x for x in coarse if x.HasFundamentalData and x.Volume > 0 and x.Price > 0], key = lambda x: x.DollarVolume, reverse=True)[:COARSE_LIMIT]
        self.dv_by_symbol = {x.Symbol:x.DollarVolume for x in sortedByDollarVolume}
        if len(self.dv_by_symbol) == 0:
            return Universe.Unchanged

        return list(self.dv_by_symbol.keys())
        
    def SelectFine(self, fine):
        sortedBySector = sorted([x for x in fine if x.CompanyReference.CountryId == "USA"
                                        and x.CompanyReference.PrimaryExchangeID in ["NYS","NAS"]
                                        and (self.Time - x.SecurityReference.IPODate).days > 540
                                        and x.MarketCap > 5e8],
                               key = lambda x: x.CompanyReference.IndustryTemplateCode)

        count = len(sortedBySector)
        if count == 0:
            return Universe.Unchanged

        self.last_month = self.Time.month
        percent = 500 / count
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
        
    def reset_vars(self):
        this_month = self.Time.month 
        if self.curr_month < 0:
            self.curr_month = this_month
        self.next_month = self.curr_month + INTERVAL - 12*(self.curr_month + INTERVAL > 12)
        if (this_month != self.curr_month):
            return False
        self.curr_month = self.next_month
        self.industries, self.pairs = [], []
        self.target_weights = {}
        self.strict_tester.reset()
        self.loose_tester.reset()
        self.Liquidate()
        
        spy_history = self.History(self.Symbol("SPY"), TimeSpan.FromDays(365 + 2*30), Resolution.Daily).close
        self.true_lookback = len(spy_history.values.tolist())
        self.desired_pairs = int(round(DESIRED_PAIRS * (self.Portfolio.TotalPortfolioValue / INITIAL_PORTFOLIO_VALUE)))
        return True
        
    def create_industries(self):
        if RUN_TEST_STOCKS:
            self.industry_map = {123: TEST_STOCKS}
        
        industries = []    
        for code in self.industry_map:
            industry = Industry(code)
            for ticker in self.industry_map[code]:
                equity = self.AddEquity(ticker)
                price_history = self.History(self.Symbol(ticker), TimeSpan.FromDays(365+2*30), Resolution.Daily).close
                if (len(price_history) >= self.true_lookback):
                    stock = Stock(ticker=ticker, id=equity.Symbol.ID.ToString())
                    stock.update_price_history(self.library.run_kalman(price_history.values.tolist()))
                    industry.add_stock(stock)
                    self.target_weights[equity.Symbol.ID.ToString()] = 0.0
            if industry.size() > 1:
                industries.append(industry)
        return sorted(industries, key=lambda x: x.size(), reverse=False)
        
#####################
# CLASS DEFINITIONS #
####################################################################################################
class Stock:
    
    def __init__(self, ticker, id):
        self.ticker = ticker
        self.id = id
        self.price_history = []
        self.purchase_price = 0
        self.long = False
        
    def __str__(self):
        return "{1}".format(self.id)
        
    def update_price_history(self, price_history):
        self.price_history = price_history

    def set_purchase_info(self, price, is_long):
        self.purchase_price = price
        self.long = is_long

class Pair:
    
    def __init__(self, s1, s2, industry):
        self.left = s1
        self.right= s2
        self.industry = industry
        self.spreads = []
        self.unfiltered_spreads = []
        self.latest_test_results = {}
        self.failed_test = ""
        self.currently_long = False
        self.currently_short = False
    
    def __str__(self):
        return "{0}: ([{1}] & [{2}])".format(self.industry.code, self.left.ticker, self.right.ticker)
        
    def results(self):
        return self.latest_test_results

    def contains(self, id):
        return (self.left.id == id) or (self.right.id == id)
    
    def update_spreads(self, spreads):
        self.spreads = spreads

class Industry:
    
    def __init__(self, code):
        self.code = code
        self.stocks = []
        self.good_pairs = []
    
    def __str__(self):
        return "Industry {0}: {1} tickers".format(self.code, self.size())
    
    def size(self):
        return len(self.stocks)
    
    def add_stock(self, stock):
        self.stocks.append(stock)
    
    def add_good_pair(self, pair):
        match = False
        for i in range(len(self.good_pairs)):
            if self.good_pairs[i].contains(pair.left.id) or self.good_pairs[i].contains(pair.right.id):
                match = True
                old_result = self.good_pairs[i].latest_test_results[RANK_BY]
                new_result = pair.latest_test_results[RANK_BY]
                if (RANK_DESCENDING and new_result > old_result) or ((not RANK_DESCENDING) and new_result < old_result):
                    self.good_pairs[i] = pair
        if (not match):
            self.good_pairs.append(pair)
        
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
        self.failures = {}
    
    def test_pair(self, pair, spreads=False):
        self.count += 1
        tests = self.spread_tests if spreads else self.price_tests
        for test in tests:
            result = None
            test_function = self.library.get_func_by_name(test.lower())
            try:
                if test == "Alpha":
                    result = 1
                elif test == "HalfLife":
                    result = test_function(pair.spreads[-HEDGE_LOOKBACK:])
                elif spreads:
                    result = test_function(pair.spreads)
                else:
                    result = test_function(pair.left.price_history[-HEDGE_LOOKBACK:], pair.right.price_history[-HEDGE_LOOKBACK:])
            except:
                pass
            
            if (not result) or (not self.test_value_bounds(test, result)):
                pair.failed_test = test
                self.failures[test] = self.failures.get(test, 0) + 1
                return False
            pair.latest_test_results[test] = round(result, 5)
        return True
    
    def test_value_bounds(self, test, result):
        return (result >= self.config[test]['min'] and result <= self.config[test]['max'])