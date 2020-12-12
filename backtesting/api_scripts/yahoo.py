import yahoo_fin.stock_info as si
import yfinance as yf
import numpy as np
import progressbar
from yahoo_finance import Share

ONE_BILLION = 10 ** 9
TEN_BILLION = 10 ** 10

all_tickers = np.unique(si.tickers_nasdaq() + si.tickers_other())
all_tickers = all_tickers[0:100]

num_tickers = len(all_tickers)
industries = {}
print("Pulling all Yahoo Finance tickers")
bar = progressbar.ProgressBar(maxval=len(all_tickers), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(num_tickers):
    bar.update(i+1)
    try:
        if (len(all_tickers[i]) > 0):
            response = yf.Ticker(all_tickers[i])
            info = response.info
            bucket = info['industry']
            if info['marketCap'] < ONE_BILLION:
                bucket += " <> LOW MC"
            elif info ['marketCap'] > TEN_BILLION:
                bucket += " <> HIGH MC"
            else:
                bucket += " <> MID MC"
            industries.setdefault(bucket, []).append(all_tickers[i])
    except:
        pass

bar.finish()
print(industries)