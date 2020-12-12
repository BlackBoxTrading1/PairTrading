FROM python:3.6

WORKDIR /root/
ADD . /root/

RUN pip install zipline==1.3.0

RUN pip install asyncio-nats-client \
				pyyaml \
				websocket-client \
				websockets \
				pykalman \
				alpaca_trade_api_fixed \
				progressbar;  zipline ingest -b quantopian-quandl

RUN mv SPY_benchmark.csv /root/.zipline/data; mv treasury_curves.csv /root/.zipline/data
CMD bash /root/commands.sh