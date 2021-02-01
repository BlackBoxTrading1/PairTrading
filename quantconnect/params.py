### <summary>
### Parameters Class
###
### Parameters used in Pair Trading Algorithm.
### </summary>

# UNIVERSE PARAMS
RUN_TEST_STOCKS     = False
TEST_STOCKS         = ['F', 'GM', 'FB', 'TWTR', 'KO', 'PEP']
COARSE_LIMIT        = 100000
FINE_LIMIT          = 10000

# BACKTEST PARAMS
START_YEAR          = 2020
START_MONTH         = 1
START_DAY           = 1
END_YEAR            = 2020
END_MONTH           = 12
END_DAY             = 30

# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 1e4
LEVERAGE               = 1.0
INTERVAL               = 1
DESIRED_PAIRS          = 10
LOOKBACK               = 365
HEDGE_LOOKBACK         = 21    #usually 15-300
ENTRY                  = 1.5   #usually 1.5
EXIT                   = 0.0   #usually 0.0
Z_STOP                 = 4.0   #usually >4.0
STOPLOSS               = 0.15
MIN_SHARE              = 5.00
MAX_SHARE              = (INITIAL_PORTFOLIO_VALUE/DESIRED_PAIRS)/2
MIN_AGE                = 365*3
MIN_WEIGHT             = 0.25
MAX_PAIR_WEIGHT        = 0.20
MIN_VOLUME             = 1e4
MKTCAP_MIN             = 1e8
MKTCAP_MAX             = 1e11

EQUAL_WEIGHTS          = True

# TESTING PARAMS
RANK_BY                   = 'HalfLife' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = False
PVALUE                    = 0.01/3

TEST_PARAMS               = {
    'Correlation':  {'min': 0.80,  'max': 1.00,                     'spreads': 0,  'run': 1 },
    'Cointegration':{'min': 0.00,  'max': 0.01,                     'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.45,                     'spreads': 1,  'run': 1 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 1 },
    'HalfLife':     {'min': 2,     'max': HEDGE_LOOKBACK*INTERVAL,  'spreads': 1,  'run': 1 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 1 },
    'Zscore':       {'min': ENTRY, 'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.10,  'max': 1.00,                     'spreads': 0,  'run': 1 }
}

LOOSE_PARAMS              = {
    'Correlation':  {'min': -1.00, 'max': 1.00,                     'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.00,  'max': 0.01,                     'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 0,     'max': HEDGE_LOOKBACK*INTERVAL,  'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,   'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.05,  'max': 1.00,                     'spreads': 0,  'run': 0 }
}