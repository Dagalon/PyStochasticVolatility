import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sbn

from pathlib import Path
from scipy import stats

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'HistoricalStoxx50E.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["date", "price"], sep=";")
no_dates = len(historical_data['date'])
historical_data['price'][1:] = historical_data['price'][1:].apply(lambda x: float(x))
log_increments = np.diff(np.log(list(historical_data['price'][1:])))

mean_log_increments = log_increments.mean()
std_log_increments = log_increments.std()

no_bins = 20
bins = np.linspace(-0.1, 0.1, no_bins)
cdf_normal = np.zeros(no_bins)
for i in range(0, no_bins):
    cdf_normal[i] = stats.norm.cdf((bins[i] - mean_log_increments) / std_log_increments)


sbn.distplot(log_increments, kde=True, bins=bins, hist=True, norm_hist=True,
             kde_kws={"color": "black", "lw": 1, "label": "Kernel distribution", 'linestyle': '--', 'cumulative': True},
             hist_kws={"histtype": "bar", "linewidth": 2, "alpha": 1, "color": "grey", 'cumulative': True})

plt.plot(bins, cdf_normal, color='black', label='Normal distribution')

plt.legend()
plt.show()
