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
ST_M, ST_D, ST_Y    = 11, 1, 2016
END_M, END_D, END_Y = 12, 28, 2016

# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 1e4
LEVERAGE               = 1.0
INTERVAL               = 1     #caldeira: 4 months w/reversals
DESIRED_PAIRS          = 20    #caldeira: 20
MAX_ACTIVE_PAIRS       = 10
LOOKBACK               = 365*1 #quantopian: 5 years
HEDGE_LOOKBACK         = 21*3  #pairtradinglab: 15-300, quantconnect: 3 mo, quantopian: 20
ENTRY                  = 2.00  #pairtradinglab: 1.5, quantconnect: 2.23, caldeira: 2.0, quantopian: 1
EXIT                   = 0.50  #pairtradinglab/quantopian: 0.0, quantconnect: 0.5, caldeira: 0.5
Z_STOP                 = 4.50  #pairtradinglab >4.0, quantconnect: 4.5
STOPLOSS               = 0.15  #caldeira: 7%
MIN_SHARE              = 5.00
MIN_AGE                = LOOKBACK*1.5
MIN_WEIGHT             = 0.40
MAX_SHARE              = 500
MAX_PAIR_WEIGHT        = 0.20
MIN_VOLUME             = 1e4
MKTCAP_MIN             = 1e8
MKTCAP_MAX             = 1e11

EQUAL_WEIGHTS          = True
SIMPLE_SPREADS         = True
CHECK_DOWNTICK         = True

# TESTING PARAMS
RANK_BY                   = 'Zscore' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING           = True
PVALUE                    = 0.05

TEST_PARAMS               = {
    'Correlation':  {'min': 0.80,  'max': 1.00,                     'spreads': 0,  'run': 1 },  #quantconnect: min = 0.9
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 1 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },  #wikipedia: 0-0.49
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 1 },
    'HalfLife':     {'min': 0,     'max': 21,                       'spreads': 1,  'run': 0 },  #caldeira: max=50 
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.00,  'max': 2.50,                     'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 0 },
    'ADFPrices':    {'min': 0.10,  'max': 1.00,                     'spreads': 0,  'run': 1 }
}

LOOSE_PARAMS              = {
    'Correlation':  {'min': -1.00, 'max': 1.00,                     'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 0,     'max': 42,                       'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,   'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 0 },
    'ADFPrices':    {'min': 0.05,  'max': 1.00,                     'spreads': 0,  'run': 0 }
}