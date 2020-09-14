import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
# import seaborn as sbn
from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'HistoricalStoxx50E.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["date", "price"], sep=";")
no_dates = len(historical_data['date'])
historical_data['price'][1:] = historical_data['price'][1:].apply(lambda x: float(x))
log_increments = np.diff(np.log(list(historical_data['price'][1:])))

mean_log_increments = log_increments.mean()
std_log_increments = log_increments.std()
log_increments_normalized = (log_increments - mean_log_increments) / std_log_increments

no_bins = 25
bins = np.linspace(log_increments_normalized.min(), log_increments_normalized.max(), no_bins)
# cdf_normal = np.zeros(no_bins)
# for i in range(0, no_bins):
#     cdf_normal[i] = stats.norm.cdf((bins[i] - mean_log_increments) / std_log_increments)


# sbn.distplot(log_increments_normalized, kde=True, bins=bins, fit=stats.norm, hist=True, norm_hist=True,
#              kde_kws={"color": "black", "lw": 1, "label": "Market density", 'linestyle': '--'},
#              fit_kws={"label": "Normal density"},
#              hist_kws={"histtype": "bar", "linewidth": 2, "alpha": 1, "color": "grey"})

# plt.plot(bins, cdf_normal, color='black', label='Normal distribution')
plt.figure(figsize=(20, 10))
pp = sm.ProbPlot(log_increments_normalized, fit=True)
qq = pp.qqplot(marker='.', markerfacecolor='k', markeredgecolor='k', alpha=0.3)
sm.qqline(qq.axes[0], line='45', fmt='k--')

# plt.legend()
plt.show()
