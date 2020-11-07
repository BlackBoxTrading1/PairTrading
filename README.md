# Pair Trading Algorithm
Market neutral algorithm to trade pairs of stocks based on historical correlation.

### Organization
- quantopian: Initial development code that was written thru the [Quantopian](https://www.quantopian.com/) API and IDE. Purely used for backtesting.
- deployment: All files associated with pulling stock data from [Polygon.io](https://polygon.io/) and backtesting using Zipline and Alpaca.
- backtesting: All files associated with cloud deployment thru Heroku.
- - - -
# API Documentation & Algorithm Logic
## Table of Contents
- [Class Definitions](#class-definitions)
  - [Stock](#stock)
  - [Pair](#pair)
- [Testing Pairs](#testing-pairs)
  - [Testing Parameters](#testing-parameters)
  - [Tests](#tests)
    - [Price Tests](#price-tests)
    - [Spread Tests](#spread-tests)
  - [Testing Logic](#testing-logic)
- [Scheduling](#scheduling)
- [Logging](#logging)

## CLASS DEFINITIONS
There are two kinds of objects in this algorithm: Stocks and Pairs.

### Stock
A Stock contains the following information:
```
Attributes

equity            symbol obect of a string ticker
name              string of stock sid and symbol
price_history     list of historical price data for stock
purchase_price    dictionary that holds price of stock when purchased as well as if the position was long or short
```

#### update_purchase_price(price, isLong)
> updates purchase_price attribute

#### test_stoploss(data)
> returns true if stock price is within STOPLOSS % of initial purchase price. Direction depends on long vs. short position

### Pair
A Pair contains the following information:
```
Attributes        

left                  the 'left' Stock in the pair. Always passed first in statistical tests
right                 the 'right' Stock in the pair. Always passed second in statistical tests
to_string             string representation of Pair for logging purposes
industry              string industry name
spreads               list of historical zscores for Pair
unfiltered_spreads    list of historical zscores for Pair without running Kalman filter on price histories for each Stock
latest_test_results   dictionary mapping each test a Pair has gone thru to the numerical value output by the test
failed_test           string name of latest test the Pair failed
currently_long        true if Pair's spreads are currently being long-ed by algo
currently_short       true if Pair's spreads are currently being shorted by algo
```

#### is_tradable(data)
> returns (True, []) if Pair's stocks can be traded. Returns (False, list of untradable stocks) if any stock in the Pair cannot be traded

#### test(context, data, loose_screens=False, test_type="spread")
> returns True if pair passes all tests, False otherwise. loose_screens parameter determines whether loose or strict parameters should be used. test_type indicates if 'price' tests should be run or 'spread' tests. More information on this function's logic in [Testing Logic](#testing-logic).


## TESTING PAIRS
### Testing Parameters
The following constants are used to search for and select pairs:
Parameter       | Description
----------------| -------------
DESIRED_PAIRS   | Max # pairs to search for in each interval
Z_STOP          | Max Z-Score allowed for pair
RANK_BY         | String to indicate which test result to rank passing pairs by
RANK_DESCENDING | True if ranking metric assigns larger values to better pairs
LOOKBACK        | How many days of price history to factor in to each pair's testing
DESIRED_PVALUE  | Max p-value for statistical tests
LOOSE_PVALUE    | Max p-value for pair's tests each day after selection
TEST_ORDER      | List of strings to indicate which order to run tests

Each test's minimum and maximum acceptable values are stored in a dictionary called TEST_PARAMS. Each entry in the dictionary has the following structure:
```python
'Hurst': {'min': 0.00, 'max': 0.49, 'type': 'spread', 'run': True }
```
The 'type' field indicates if the test is a spread test or price test. More on this in [Tests](#tests)

### Tests
#### Price Tests
Here are the 'price' tests that are run on every single pair:
Test            | Function                  | Description
----------------| --------------------------| --------------
Correlation     | correlation(s1, s2)       | 
Cointegration   | cointegration(s1, s2)     |
AD Fuller       | adf_prices(s1, s2)        |
Alpha           | alpha(s1, s2, s1_p, s2_p) |

Note: Alpha test also takes in the current price of s1 and s2 along with their filtered price histories

#### Spread Tests
Here are the 'spread' tests that are run on every pair that passes the price tests (run on pair's zscores):
Test            | Function                  | Description
----------------| --------------------------| --------------
AD Fuller       | adf_pvalue(zscores)       |
Hurst Exponent  | hurst(zscores)            |
Half-life       | half_life(zscores)        |
Shapiro-Wilke   | shapiro_pvalue(zscores)   |
Z-Score         | zscore (zscores)          |

### Testing Logic
#### get_test_by_name(test)
> returns appropriate test function from [list](#tests) given string name

#### Pair.test(context, data, loose_screens=False, test_type="spread")
Pseudo code for the pair testing algorithm:
```
FOR each active test in test order
  SET result = invalid
  GET current test function by name

  IF test type = 'price'
    SET result = current test run on price histories
  IF test type = 'spread'
    SET result = current test run on zscores

  IF result = invalid
    RETURN failure
  IF result not inside bounds specified for test by TEST_PARAMS
    RETURN failure
  IF test = ranking metric and result is worse than all other results for current industry
    RETURN failure

IF test type = 'spread'
  UPDATE current industry's list of top pairs without allowing repeated stocks

RETURN success
```
The final two conditionals do not take place if loose_screens is set to True; loose screening only takes place after all pairs are chosen to make sure their quality does not significantly drop.

## SCHEDULING
There are two functions in this algorithm that are scheduled: set_universe and check_pair_status. The universe must be set on the first trading day that is not after the 19th trading day of the month every single month. The status of the chosen pairs must be checked every day after that. The code for this schedule is below:
```python
day = get_datetime().day - (int)(2*get_datetime().day/7) - 3

schedule_function(set_universe, date_rules.month_start(day * (not (day < 0 or day > 19))), time_rules.market_open(hours=0, minutes=1))
schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(hours = 1))
```

## LOGGING
#### write_to_file(filename, data)
> Opens file with specified name and writes data to it.
#### log (message)
> Writes message to file with name "logs.txt". Target file should exist first.
