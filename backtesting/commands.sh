zipline run -f algo.py -b quantopian-quandl --start 2012-01-01 --end 2014-01-01 -o output.pickle --capital-base 10000.0 
curl -F "file=@output.pickle" https://file.io 
curl -F "file=@logs.txt" https://file.io
sleep 120h