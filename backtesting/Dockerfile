FROM python:3.6

WORKDIR /root/
ADD . /root/

RUN pip install zipline==1.3.0 

RUN pip install alpaca-trade-api==0.38 \
				appnope==0.1.0 \
				astor==0.8.0 \
				asyncio-nats-client==0.9.2 \
				backcall==0.1.0 \
				jedi==0.15.1 \
				mock==3.0.5 \
				parso==0.5.1 \
				pexpect==4.7.0 \
				pickleshare==0.7.5 \
				pipeline-live==0.1.9 \
				prompt-toolkit==2.0.9 \
				ptyprocess==0.6.0 \
				pygments==2.4.2 \
				pylivetrader==0.1.1 \
				pyyaml==5.1.2 \
				traitlets==4.3.2 \
				wcwidth==0.1.7 \
				websocket-client==0.56.0 \
				websockets==8.0.2 \
				wrapt==1.11.2 \
				pymarketstore \
				ConfigParser \
				matplotlib \
				pykalman \
				alpaca_trade_api_fixed \
				progressbar;  zipline ingest -b quantopian-quandl

RUN mv SPY_benchmark.csv /root/.zipline/data; mv treasury_curves.csv /root/.zipline/data
CMD bash /root/commands.sh