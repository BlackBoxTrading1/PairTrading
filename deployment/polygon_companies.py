import requests
import json
import alpaca_trade_api as tradeapi
import progressbar

MAX_COMPANY = 50
EXCHANGES = ['New York Stock Exchange', 'Nasdaq Global Select', 'NYSE American', 'NASDAQ Global Market', 'NASDAQ Capital Market']

API = tradeapi.REST(
    base_url="https://paper-api.alpaca.markets",
    key_id="PKSL6HFOBBRWI3ZYB3CE",
    secret_key="oFil1E/0DN1WTatQMGoo6YahQXudVRED9t6dBNbV"
)
assets = API.list_assets()
symbols = [asset.symbol for asset in assets if asset.tradable]

num_requests = int(len(symbols)/MAX_COMPANY)+(1 if len(symbols) % MAX_COMPANY > 0 else 0)

print("Pulling all Polygon companies")
bar = progressbar.ProgressBar(maxval=num_requests, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
companies = {}
for r in range(num_requests):
    new_companies = {}
    if r == num_requests - 1:
        new_companies = API.polygon.company(symbols[r*MAX_COMPANY:])
    else:
        new_companies = API.polygon.company(symbols[r*MAX_COMPANY:(r+1)*MAX_COMPANY])
    companies.update(new_companies)
    bar.update(r+1)
bar.finish()

industries = {}
for ticker, company in companies.items():
    if not (company.exchange in EXCHANGES):
        continue
    if not hasattr(company, 'type') or company.type != 'CS':
        continue
    if ('Trust' in company.description.split()) or ('Fund' in company.description.split()):
        continue

    if not (company.industry in industries):
        industries[company.industry] = []
    industries[company.industry].append(ticker)

delete = [key for key in industries if len(industries[key]) < 2]
for key in delete:
    del industries[key]

print(sum(len(industries[key]) for key in industries))
