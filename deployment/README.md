# Pair Trading Deployment Code
The files in this folder are used for deployment thru the Pylivetrader API and Alpaca

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

### Set environmental variables for Heroku
```
heroku config:set APCA_API_KEY_ID=<ReplaceWithSuppliedKey>
heroku config:set APCA_API_SECRET_KEY=<ReplaceWithSuppliedSecKey>
heroku config:set APCA_API_base_url=https://paper-api.alpaca.markets
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