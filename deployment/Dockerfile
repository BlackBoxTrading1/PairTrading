FROM alpacamarkets/pylivetrader

ARG APCA_API_SECRET_KEY
ARG APCA_API_KEY_ID
ARG APCA_API_BASE_URL

ENV APCA_API_SECRET_KEY=oFil1E/0DN1WTatQMGoo6YahQXudVRED9t6dBNbV
ENV APCA_API_KEY_ID=PKSL6HFOBBRWI3ZYB3CE
ENV APCA_API_BASE_URL=https://paper-api.alpaca.markets

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN pip install pylivetrader \
				numpy \
				pandas \
				statsmodels \
				scipy \
				pykalman \
				requests \
				alpaca_trade_api_fixed \
				progressbar

CMD pylivetrader run -f algo.py