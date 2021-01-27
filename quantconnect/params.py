### <summary>
### Parameters Class
###
### Parameters used in Pair Trading Algorithm.
### </summary>

# UNIVERSE PARAMS
RUN_TEST_STOCKS     = False
TEST_STOCKS         = ['AAPL', 'MSFT', 'NFLX', 'AMZN', 'TSLA', 'FB', 'YELP', 'AMD', 'NVDA', 'EA', 'INTC', 'XRX']
COARSE_LIMIT        = 1000000
FINE_LIMIT          = 500

# BACKTEST PARAMS
START_YEAR          = 2018
START_MONTH         = 1
START_DAY           = 1
END_YEAR            = 2018
END_MONTH           = 1
END_DAY             = 21

# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 10000
LEVERAGE               = 1.0
INTERVAL               = 1
DESIRED_PAIRS          = 10
LOOKBACK               = 365
HEDGE_LOOKBACK         = 21  #usually 15-300
ENTRY                  = 1.25 #usually 1.5
EXIT                   = 0.1 #usually 0.0
Z_STOP                 = 3.0 #usually >4.0
STOPLOSS               = 0.15
MIN_SHARE              = 2.00
MIN_AGE                = 365*3
MIN_WEIGHT             = 0.25
MAX_PAIR_WEIGHT        = 0.25
MIN_VOLUME             = 1e3
MKTCAP_MIN             = 30e6
MKTCAP_MAX             = 500e9

EQUAL_WEIGHTS          = False

# TESTING PARAMS
RANK_BY                   = 'Hurst' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = False

TEST_PARAMS               = {
    'Correlation':  {'min': 0.80,  'max': 1.00,            'spreads': 0,  'run': 1 },
    'Cointegration':{'min': 0.0,   'max': 0.01/3,          'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,            'spreads': 1,  'run': 1 },
    'ADFuller':     {'min': 0.00,  'max': 0.05,            'spreads': 1,  'run': 1 },
    'HalfLife':     {'min': 1.0,   'max': 1e2,             'spreads': 1,  'run': 1 },
    'ShapiroWilke': {'min': 0.00,  'max': 0.05,            'spreads': 1,  'run': 1 },
    'Zscore':       {'min': ENTRY, 'max': 2.0,             'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 0,     'max': 1e9,             'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.01,  'max': 1.00,            'spreads': 0,  'run': 1 }
}

LOOSE_PARAMS              = {
    'Correlation':  {'min': 0.8,     'max': 1.00,            'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.0,     'max': 0.05/3,          'spreads': 0,  'run': 0 },
    'ADFuller':     {'min': 0.0,     'max': 0.05/3,          'spreads': 1,  'run': 0 },
    'Hurst':        {'min': 0.0,     'max': 0.49,            'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 1.0,     'max': HEDGE_LOOKBACK,  'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.0,     'max': 0.05/3,          'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,     'max': Z_STOP,          'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 0.0,     'max': 1e9,             'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.05,    'max': 1.00,            'spreads': 0,  'run': 0 }
}
