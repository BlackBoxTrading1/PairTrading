algo:
	zipline run -f algo.py --start $(start) --end $(end) -o out.pickle --no-benchmark
	
industries-polygon:
	python3 api_scripts/polygon.py

industries-yahoo:
	python3 api_scripts/yahoo.py

returns:
	python unpickle.py -f out.pickle