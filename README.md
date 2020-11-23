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
- [Collecting Ticker Data](#collecting-ticker-data)
- [Selecting Pairs](#selecting-pairs)
- [Trading Pairs](#trading-pairs)
  - [check_pair_status()](#check_pair_status)
  - [Helper Functions](#helper-functions-for-trading-logic)
- [Logging](#logging)

## CLASS DEFINITIONS
There are two kinds of objects in this algorithm: Stocks and Pairs.

### Stock
A Stock contains the following information:
```
Attributes

equity            symbol object of a string ticker
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
Correlation     | correlation(s1, s2)       | Returns correlation coefficient.
Cointegration   | cointegration(s1, s2)     | Returns cointegration p-value.
AD Fuller       | adf_prices(s1, s2)        | Unit root test for stationarity. Returns p-value.
Alpha           | alpha(s1, s2, s1_p, s2_p) | Returns slope of regression if projected target weights are greater than minimum.

Note: Alpha test also takes in the current price of s1 and s2 along with their filtered price histories

#### Spread Tests
Here are the 'spread' tests that are run on every pair that passes the price tests (run on pair's zscores):
Test            | Function                  | Description
----------------| --------------------------| --------------
AD Fuller       | adf_pvalue(zscores)       | Unit root test for stationarity. Returns p-value.
Hurst Exponent  | hurst(zscores)            | Measures autocorrelation of Z-Scores.
Half-life       | half_life(zscores)        | Returns half-life of spreads converging.
Shapiro-Wilke   | shapiro_pvalue(zscores)   | Tests normality of Z-Scores. Returns p-value.
Z-Score         | zscore (zscores)          | Tests if latest Z-Score is within acceptable range.

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

## COLLECTING TICKER DATA
#### collect_polygon_tickers(base_url, key_id, secret_key)
> returns dictionary mapping all tradable Polygon tickers to an industry given Alpaca authentication

#### context.industries
> dictionary mapping industry to list of Stock objects, number of Stocks, and list of top pairs

#### context.universe_set
> True if there exists an industry with more than 1 stock. Pairs will not be chosen if False

#### context.target_weights
> dictionary mapping each ticker to its current position weight in portfolio

#### context.remaining_codes
> sorted list of industries from largest to smallest

#### run_kalman(price_history)
> returns kalman filtered price history; every ticker's price history gets filtered before pair selection process

## SELECTING PAIRS
Every possible pair within each industry is generated and tested. The pairs that pass the price tests move on to the spread tests. The pairs that pass the spread tests move on to the ranking. The top pairs as determined by the ranking metric are then selected to be traded. The ranking metric and number of desired top pairs are specified as [testing parameters](#testing-parameters).

#### context.all_pairs
> dictionary mapping industry to Pairs within the idustry that are still 'alive' in testing

#### context.pairs
> master list of Pair objects that have been selected to trade

## TRADING PAIRS
### check_pair_status
Pseudo code of trading logic:
```
FOR each selected pair
  IF pair fails stoploss test
    REMOVE pair
  IF pair contains untradable stock
    REMOVE pair
  IF pair fails loose parameter testing
    REMOVE pair

  IF pair is being shorted and zscore too low or pair is being longed and zscore too high
    SELL pair
  IF pair is not being longed and zscore in long entry window
    BUY pair
  IF pair is not being shorted and zscore in short entry window
    BUY pair
```

### Helper functions for trading logic
#### sell_pair(context, data, pair)
> empties position on pair and scales weights of other trading pairs appropriately

#### buy_pair(context, data, pair, y_target_shares, X_target_shares, s1, s2, new_pair=True)
> calculates target percentages based on target shares of pair (s1, s2) and scales other pairs appropriately; set new_pair to True if pair had no position before this call

#### calculate_target_pcts(y_target_shares, X_target_shares, s1_price, s2_price)
> determines target weight percentages for pair given desired target shares

#### allocate(context, data)
> goes through context.target_weights after trading logic executes to officially place the orders

#### num_allocated_stocks(context)
> returns number of stocks that currently have non-zero weight

#### scale_stocks(context, factor)
> scales weight of all stocks in portfolio by factor

#### scale_pair_pct(context, factor)
> scales weight of every selected pair by factor

#### update_target_weight(context, data, stock, new_weight)
> updates context.target_weights for given stock

#### remove_pair(context, pair, index)
> removes pair from context.pairs and increments number of desired pairs

#### get_spreads(data, s1_price, s2_price, length)
> returns (historical zscores, final HEDGE_LOOKBACK residuals) for given pair of price histories


## LOGGING
#### write_to_file(filename, data)
> Opens file with specified name and writes data to it.
#### log (message)
> Writes message to file with name "logs.txt". Target file should exist first.
