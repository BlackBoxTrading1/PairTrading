### <summary>
### Parameters Class
###
### Parameters used in Pair Trading Algorithm.
### </summary>

# UNIVERSE PARAMS
RUN_TEST_STOCKS     = False
TEST_STOCKS         = {123: ['F', 'GM', 'FB', 'TWTR', 'KO', 'PEP']}
COARSE_LIMIT        = 10000
FINE_LIMIT          = 1000

# BACKTEST PARAMS
ST_M, ST_D, ST_Y    = 1, 1, 2013
END_M, END_D, END_Y = 6, 28, 2013

# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 1e4
LEVERAGE               = 1.0
INTERVAL               = 1     #caldeira: 4 months w/reversals
DESIRED_PAIRS          = 10    #caldeira: 20
LOOKBACK               = 365*5 #quantopian: 5 years
HEDGE_LOOKBACK         = 21*3  #pairtradinglab: 15-300, quantconnect: 3 mo
ENTRY                  = 1.50  #pairtradinglab: 1.5, quantconnect: 2.23, caldeira: 2.0
EXIT                   = 0.50  #pairtradinglab: 0.0, quantconnect: 0.5, caldeira: 0.5
Z_STOP                 = 4.00  #pairtradinglab >4.0, quantconnect: 4.5
STOPLOSS               = 0.07  #caldeira: 7%
MIN_SHARE              = 5.00
MIN_AGE                = LOOKBACK*1.5
MIN_WEIGHT             = 0.40
MAX_SHARE              = MIN_WEIGHT*(INITIAL_PORTFOLIO_VALUE/DESIRED_PAIRS)
MAX_PAIR_WEIGHT        = 0.20
MIN_VOLUME             = 1e4
MKTCAP_MIN             = 1e8
MKTCAP_MAX             = 1e11

EQUAL_WEIGHTS          = True
SIMPLE_SPREADS      = True

# TESTING PARAMS
RANK_BY                   = 'Zscore' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = False
PVALUE                    = 0.05

TEST_PARAMS               = {
    'Correlation':  {'min': 0.90,  'max': 1.00,                     'spreads': 0,  'run': 1 }, #quantconnect: min = 0.9
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 }, #wikipedia: 0-0.49
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 3,     'max': 50,                       'spreads': 1,  'run': 0 }, #caldeira: max=50 
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': ENTRY, 'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 }
}

LOOSE_PARAMS              = {
    'Correlation':  {'min': -1.00, 'max': 1.00,                     'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 0,     'max': 42,                       'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,   'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 1 },
    'ADFPrices':    {'min': 0.05,  'max': 1.00,                     'spreads': 0,  'run': 0 }
}