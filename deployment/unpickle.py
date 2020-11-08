import os.path
import sys, getopt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import env_constants

def main(argv):
    folder = ""
    try:
        opts, args = getopt.getopt(argv,"n:",["name="])
    except getopt.GetoptError:
        print("Usage: python unpickly.py -n <backtest name>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-n", "--name"):
            folder = arg

    if os.path.isfile("backtests/{0}/{1}".format(folder,env_constants.OUTPUT_FILE)):
        style.use('ggplot')
        backtest_df = pd.read_pickle("backtests/{0}/{1}".format(folder,env_constants.OUTPUT_FILE))
        backtest_df.algorithm_period_return.plot()
        plt.show()
    else:
        print ("Error: No output file found in folder with name {0}.".format(folder))

if __name__ == "__main__":
   main(sys.argv[1:])