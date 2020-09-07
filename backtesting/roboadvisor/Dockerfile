FROM python:3.6

WORKDIR /home/robo-advisor
ADD . /home/robo-advisor

RUN apt-get update && apt-get install -y \
	libatlas-base-dev \
	python-dev \
	gfortran \
	pkg-config \
	vim \
	libfreetype6-dev

RUN pip install alembic==1.1.0 \
				alpaca-trade-api==0.38 \
				appnope==0.1.0 \
				astor==0.8.0 \
				asyncio-nats-client==0.9.2 \
				backcall==0.1.0 \
				bcolz==0.12.1 \
				certifi==2020.06.20 \
				chardet==3.0.4 \
				click==7.0 \
				contextlib2==0.5.5 \
				cyordereddict==1.0.0 \
				cython==0.29.13 \
				decorator==4.4.0 \
				empyrical==0.5.3 \
				idna==2.8 \
				iexfinance==0.4.3 \
				intervaltree==3.0.2 \
				ipython==7.8.0 \
				ipython-genutils==0.2.0 \
				jedi==0.15.1 \
				logbook==1.5.2 \
				lru-dict==1.1.6 \
				lxml==4.4.1 \
				mako==1.1.0 \
				markupsafe==1.1.1 \
				mock==3.0.5 \
				multipledispatch==0.6.0 \
				networkx==1.11 \
				numexpr==2.7.0 \
				numpy==1.18.1 \
				bottleneck==1.0.0 \
				pandas==0.22.0 \
				pandas-datareader==0.7.4 \
				parso==0.5.1 \
				patsy==0.5.1 \
				pexpect==4.7.0 \
				pickleshare==0.7.5 \
				pipeline-live==0.1.9 \
				prompt-toolkit==2.0.9 \
				ptyprocess==0.6.0 \
				pygments==2.4.2 \
				pylivetrader==0.1.1 \
				python-dateutil==2.8.0 \
				python-editor==1.0.4 \
				pytz==2019.2 \
				pyyaml==5.1.2 \
				requests==2.22.0 \
				requests-file==1.4.3 \
				scipy==1.3.1 \
				six==1.12.0 \
				sortedcontainers==2.1.0 \
				sqlalchemy==1.3.8 \
				statsmodels==0.10.1 \
				tables==3.5.2 \
				toolz==0.10.0 \
				trading-calendars==1.8.1 \
				traitlets==4.3.2 \
				urllib3==1.24.3 \
				wcwidth==0.1.7 \
				websocket-client==0.56.0 \
				websockets==8.0.2 \
				wrapt==1.11.2 \
				zipline==1.3.0 \
				pymarketstore \
				ConfigParser \
				matplotlib \
				pykalman \
				alpaca_trade_api_fixed \
				progressbar \
				pip
				
RUN zipline ingest -b quantopian-quandl
RUN mv setup/dotzipline.tgz /root/.zipline/
RUN rm /root/.zipline/extension.py
RUN tar -xvzf /root/.zipline/dotzipline.tgz -C /root/.zipline/
RUN mkdir /root/.zipline/data/alpaca/2018-06-11T20\;08\;42.452595/minute_equities.bcolz
RUN cp `find /root/.zipline/ -name "metadata.json"` /root/.zipline/data/alpaca/2018-06-11T20\;08\;42.452595/minute_equities.bcolz/
RUN cp $(find /root/.zipline/data/quantopian-quandl -name "adjustments.sqlite") /root/.zipline/data/alpaca/2018-06-11T20\;08\;42.452595/
RUN zipline run -f algo.py -b alpaca --start 2018-01-01 --end 2018-06-01 -o output.pickle
RUN curl --upload-file output.pickle https://transfer.sh/output.pickle