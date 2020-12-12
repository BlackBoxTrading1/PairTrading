# Pair Trading Backtesting Code
The files in this folder are used for local backtesting thru the Zipline API and Alpaca

## Local Backtesting

### Run Algorithm Locally on Zipline 
Run the following command. Replace START and END with format YYYY-DD-MM to indicate start and end dates. Replace NAME with a unique name for the backtest. The logs and output of the backtest will be stored in a folder with this name:
```
make algo start=START end=END
```

### View Returns of Algorithm 
Run the following command AFTER running the trading algorithm to view a graph of returns. Replace NAME with the desired backtest:
```
make returns
```

### View All Tickers by Industry
Run one of the following two command to view a map of tickers to their industries from either Polygon or Yahoo Finance:
```
make industries-polygon
make industries-yahoo
```

- - - -

## Heroku Backtesting

### Install Heroku CLI
[Click here to download appropriate installer](https://devcenter.heroku.com/articles/heroku-cli)

### Setup
Initalize a new repository:
```
git init
```

Create a new empty Heroku application:
```
heroku create
```

Set the stack of the app to conatiner:
```
heroku stack:set container
```

Add the Heroku app as a Git remote:
```
heroku git:remote -a yourapp
```

Install the Papertrail add-on to your app:
```
heroku addons:create papertrail
```

### Running Backtests
Specify start and end dates alongside capital base in commands.sh

Download logs.txt and output.pickle using link from Papertrail logs:
```
curl https://file.io/your_key
```

### Push to Heroku
Add the changes in the working directory to the staging area:
```
git add .
```

Locally record changes to the repository:
```
git commit -m "initial commit"
```

Deploy the app to Heroku:
```
git push heroku master
```

Activate dyno
```
heroku ps:scale worker=1
```