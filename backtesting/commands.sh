zipline run -f algo.py -b quantopian-quandl --start 2011-01-01 --end 2011-02-01 -o out.pickle --capital-base 10000.0
curl -F "file=@out.pickle" https://file.io 
curl -F "file=@logs.txt" https://file.io
sleep 168h