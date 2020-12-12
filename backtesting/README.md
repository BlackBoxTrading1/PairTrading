# Pair Trading Backtesting Code
The files in this folder are used for local backtesting thru the Zipline API and Alpaca

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