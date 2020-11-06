# Pair Trading Deployment Code
The files in this folder are used for local backtesting thru the Zipline API and Alpaca

### Run Algorithm Locally on Zipline 
Run the following command, replacing START and END with format YYYY-DD-MM to indicate start and end dates:
```
make algo start=START end=END
```

### View Returns of Algorithm 
Run the following command AFTER running the trading algorithm to view a graph of returns:
```
make returns
```

### View All Polygon Tickers by Industry
Run this command to view a map of all Polygon industries to their tickers:
```
make industries
```

### View Backtest Logs
The logs of a backtest will be stored in a new file named "logs.txt" once a backtest is run
