import os.path
import sys, getopt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

def main(argv):
    filename = ""
    try:
        opts, args = getopt.getopt(argv,"f:",["file="])
    except getopt.GetoptError:
        print("Usage: python unpickly.py -f <backtest name>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            filename = arg

    if os.path.isfile(filename):
        style.use('ggplot')
        backtest_df = pd.read_pickle(filename)
        ax1 = plt.subplot(111)
        backtest_df.algorithm_period_return.plot(ax=ax1)
        ax1.set_ylabel('Algorithm Period Return')
        plt.show()
    else:
        print ("Error: No output file found with name {0}.".format(filename))

if __name__ == "__main__":
   main(sys.argv[1:])