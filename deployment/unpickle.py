import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
backtest_df = pd.read_pickle("out.pickle")
backtest_df.algorithm_period_return.plot()
plt.show()
