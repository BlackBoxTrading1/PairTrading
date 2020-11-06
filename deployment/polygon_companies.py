import requests
import json
import alpaca_trade_api as tradeapi
import progressbar
import env_constants

MAX_COMPANY = 50
EXCHANGES = ['New York Stock Exchange', 'Nasdaq Global Select', 'NYSE American', 'NASDAQ Global Market', 'NASDAQ Capital Market']

API = tradeapi.REST(
    base_url=env_constants.APCA_BASE_URL,
    key_id=env_constants.APCA_KEY_ID,
    secret_key=env_constants.APCA_SECRET_KEY
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

print(industries)
