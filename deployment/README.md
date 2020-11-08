# Pair Trading Deployment Code
The files in this folder are used for local backtesting thru the Zipline API and Alpaca

### Run Algorithm Locally on Zipline 
Run the following command. Replace START and END with format YYYY-DD-MM to indicate start and end dates. Replace NAME with a unique name for the backtest. The logs and output of the backtest will be stored in a folder with this name:
```
make algo start=START end=END name=NAME
```

### View Returns of Algorithm 
Run the following command AFTER running the trading algorithm to view a graph of returns. Replace NAME with the desired backtest:
```
make returns name=NAME
```

### View All Polygon Tickers by Industry
Run this command to view a map of all Polygon industries to their tickers:
```
make industries
```