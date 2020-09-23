import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'SPX_sample.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["DateTime", "Open", "High", "Low", "Close"],
                              sep=",")

sampling_dates = list(set(historical_data["DateTime"].apply(lambda x: x[0:10])))

rv_i_s = []
spot_vol = [0]
alpha_t_min = 24.0 * 60.0 * 253
sum_rv_i_1 = 0.0
for date in sampling_dates:
    log_price_intraday = np.log(np.array(historical_data["Close"][historical_data["DateTime"].apply(lambda x: x[0:10] == date)]))
    rv_i_s += list(np.cumsum(np.power(np.diff(log_price_intraday), 2.0)) + sum_rv_i_1)
    spot_vol += [spot_vol[-1]] + list(np.diff(np.cumsum(np.power(np.diff(log_price_intraday), 2.0))) * alpha_t_min)
    sum_rv_i_1 = rv_i_s[-1]

fig, axs = plt.subplots(2, 1)
plt.figure(figsize=(30, 15))

min_time_vol = np.arange(0, len(rv_i_s), 1)
axs[0].plot(min_time_vol, rv_i_s, label="integrated_volatility", color="black")
axs[0].set_title("Integrated_volatility")
axs[0].set(xlabel='t (minutes)')
axs[1].plot(min_time_vol, np.sqrt(spot_vol[1:]), color="black")
axs[1].set(xlabel='t (minutes)')
axs[1].set_title("Spot volatility")

plt.show()
