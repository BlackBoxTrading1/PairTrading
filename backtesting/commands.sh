zipline run -f algo.py -b quantopian-quandl --start 2015-01-01 --end 2017-01-01 -o output.pickle
curl -F "file=@output.pickle" https://file.io 
curl -F "file=@logs.txt" https://file.io
sleep 24h