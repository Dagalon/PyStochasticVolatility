import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'HistoricalStoxx50E.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["date", "price"], sep=";")
no_dates = len(historical_data['date'])
historical_data['price'][1:] = historical_data['price'][1:].apply(lambda x: float(x))
log_increments = np.diff(np.log(list(historical_data['price'][1:])))


plt.hist(log_increments, bins=20, range=[-0.1, 0.1], histtype='step', edgecolor='b', linewidth=2, density=True)
plt.title("Empirical log-return Vs Black-Scholes")
plt.show()