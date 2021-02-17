### <summary>
### Parameters Class
###
### Parameters used in Pair Trading Algorithm.
### </summary>

# UNIVERSE PARAMS
RUN_TEST_STOCKS     = False
TEST_STOCKS         = {123: ['F', 'GM', 'FB', 'TWTR', 'KO', 'PEP']}
COARSE_LIMIT        = 500
FINE_LIMIT          = 100

# BACKTEST PARAMS
ST_M, ST_D, ST_Y    = 1, 1, 2016
END_M, END_D, END_Y = 2, 25, 2016
SIMPLE_SPREADS      = True

# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 1e4
LEVERAGE               = 1.0
INTERVAL               = 1
DESIRED_PAIRS          = 10
LOOKBACK               = 365
HEDGE_LOOKBACK         = 21    #usually 15-300
ENTRY                  = 1.5   #usually 1.5
EXIT                   = 0.0   #usually 0.0
Z_STOP                 = 4.5   #usually >4.0
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
RANK_BY                   = 'ADFuller' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = False
PVALUE                    = 0.01

TEST_PARAMS               = {
    'Correlation':  {'min': 0.80,  'max': 1.00,                     'spreads': 0,  'run': 1 },
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': -1e9,  'max': -2.8723451788613157,      'spreads': 1,  'run': 1 },
    'HalfLife':     {'min': 2,     'max': HEDGE_LOOKBACK*INTERVAL,  'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': ENTRY, 'max': Z_STOP-ENTRY,             'spreads': 1,  'run': 0 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 0 },
    'ADFPrices':    {'min': 0.10,  'max': 1.00,                     'spreads': 0,  'run': 1 }
}

LOOSE_PARAMS              = {
    'Correlation':  {'min': -1.00, 'max': 1.00,                     'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 0,     'max': HEDGE_LOOKBACK*INTERVAL,  'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,   'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.05,  'max': 1.00,                     'spreads': 0,  'run': 0 }
}