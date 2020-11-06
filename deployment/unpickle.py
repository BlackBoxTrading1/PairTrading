import os.path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import env_constants

if os.path.isfile(env_constants.OUTPUT_FILE):
    style.use('ggplot')
    backtest_df = pd.read_pickle(env_constants.OUTPUT_FILE)
    backtest_df.algorithm_period_return.plot()
    plt.show()
else:
    print ("No output file found. Run backtest first.")
