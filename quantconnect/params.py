### <summary>
### Parameters Class
###
### Parameters used in Pair Trading Algorithm.
### </summary>

# UNIVERSE PARAMS
RUN_TEST_STOCKS        = False
TEST_STOCKS            = {123: ['F', 'GM', 'FB', 'TWTR', 'KO', 'PEP']}
COARSE_LIMIT           = 10000
FINE_LIMIT             = 500    # <COARSE_LIMIT

# BACKTEST PARAMS
ST_M, ST_D, ST_Y       = 1, 1, 2016
END_M, END_D, END_Y    = 12, 28, 2016
# TRADING PARAMS
INITIAL_PORTFOLIO_VALUE= 5e3
LEVERAGE               = 1.0
INTERVAL               = 1     #caldeira: 4 months w/reversals
DESIRED_PAIRS          = FINE_LIMIT    
MAX_ACTIVE_PAIRS       = 5     #caldeira: 20
HEDGE_LOOKBACK         = 15*1  #pairtradinglab: 15-300, quantconnect: 3 mo, quantopian: 20
RSI_LOOKBACK           = 14    #default = 14
LOOKBACK               = 365*2 #quantopian: 5 years
ENTRY                  = 2.00  #pairtradinglab: 1.5, quantconnect: 2.23, caldeira: 2.0, quantopian: 1
EXIT                   = 0.50  #pairtradinglab/quantopian: 0.0, quantconnect: 0.5, caldeira: 0.5
DOWNTICK               = 1.00  #pairtradinglab: 0-1
Z_STOP                 = 4.50  #pairtradinglab >4.0, quantconnect: 4.5
RSI_THRESHOLD          = 30    # default = 20
STOPLOSS               = 0.07  #caldeira: 7%
MIN_SHARE              = 5.00
MIN_AGE                = LOOKBACK*2
MIN_WEIGHT             = 0.40
MAX_SHARE              = 250
MAX_PAIR_WEIGHT        = 0.20
MIN_VOLUME             = 1e4
MKTCAP_MIN             = 1e8
MKTCAP_MAX             = 1e11

EQUAL_WEIGHTS          = True
SIMPLE_SPREADS         = True
CHECK_DOWNTICK         = True
EWA                    = False
RSI                    = True

# TESTING PARAMS
RANK_BY                = 'Hurst' # Ranking metric: select key from TEST_PARAMS
RANK_DESCENDING        = False
PVALUE                 = 0.05

TEST_PARAMS            = {
    'Correlation':  {'min': 0.75,  'max': 1.00,                     'spreads': 0,  'run': 1 },  #quantconnect: min = 0.9
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 1 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 1 },  #wikipedia: 0-0.49
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 1 },
    'HalfLife':     {'min': 3,     'max': 21,                       'spreads': 1,  'run': 1 },  #caldeira: max=50 
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 1 },
    'Zscore':       {'min': 0.00,  'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 0 },
    'ADFPrices':    {'min': 0.10,  'max': 1.00,                     'spreads': 0,  'run': 1 }
}

LOOSE_PARAMS           = {
    'Correlation':  {'min': -1.00, 'max': 1.00,                     'spreads': 0,  'run': 0 },
    'Cointegration':{'min': 0.00,  'max': PVALUE,                   'spreads': 0,  'run': 0 },
    'Hurst':        {'min': 0.00,  'max': 0.49,                     'spreads': 1,  'run': 0 },
    'ADFuller':     {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'HalfLife':     {'min': 0,     'max': 50,                       'spreads': 1,  'run': 0 },
    'ShapiroWilke': {'min': 0.00,  'max': PVALUE,                   'spreads': 1,  'run': 0 },
    'Zscore':       {'min': 0.0,   'max': Z_STOP,                   'spreads': 1,  'run': 1 },
    'Alpha':        {'min': 1e-1,  'max': 1e1,                      'spreads': 0,  'run': 0 },
    'ADFPrices':    {'min': 0.05,  'max': 1.00,                     'spreads': 0,  'run': 0 }
}