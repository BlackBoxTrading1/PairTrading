algo:
	@if [ ! -d "backtests" ]; then \
		echo "Creating backtests folder"; \
		mkdir "backtests"; \
	fi

	@if [ -d "backtests/$(name)" ]; then \
		echo "Provide unique backtest name"; \
	else \
		echo "Running algorithm"; \
		zipline run -f algo.py --start $(start) --end $(end) -o out.pickle; \
		cd "backtests"; \
		mkdir "$(name)"; \
		mv ../logs.txt ../out.pickle $(name); \
	fi
	
industries:
	python3 polygon_companies.py

returns:
	python unpickle.py -f backtests/$(name)/out.pickle