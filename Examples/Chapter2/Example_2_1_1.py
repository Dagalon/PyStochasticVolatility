import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path
from AnalyticEngines.VolatilityTools import VolatilityEstimators

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
historical_data_path = os.path.join(folder_directory, 'Data', 'SPX_sample.txt')

historical_data = pd.read_csv(historical_data_path, header=None, names=["DateTime", "Open", "High", "Low", "Close"],
                              sep=",")

sampling_dates = sorted(list(set(historical_data["DateTime"].apply(lambda x: x[0:10]))))

rv_i_s = []
end_date_rv_i_s = []
end_date_spot_i_s = []
rv_i_s_fourier = []
spot_vol_i_s_fourier = []

spot_vol = [0]
alpha_t_min = 24.0 * 60.0 * 253
sum_rv_i_1 = 0.0
sum_rv_i_1_fourier = 0.0
i_cont = 0
for date in sampling_dates:
    log_price_intraday = np.log(np.array(historical_data["Close"][historical_data["DateTime"].apply(lambda x: x[0:10] == date)]))

    # fourier estimator
    no_time_steps = len(log_price_intraday)
    delta_time = no_time_steps/(60 * 24)
    log_paths = np.zeros(shape=(1, no_time_steps))
    log_paths[0] = log_price_intraday
    frequency = 1
    t_k = np.linspace(i_cont * delta_time, (i_cont + 1) * delta_time, no_time_steps)
    rv_i_s_fourier.append(VolatilityEstimators.get_integrated_variance_fourier(log_paths, t_k, frequency, 1)[0] +
                          sum_rv_i_1_fourier)

    spot_vol_i_s_fourier.append(VolatilityEstimators.get_spot_variance_fourier(log_paths, t_k, 1, t_k[-1])[0] * 253.0)

    aux_rv = list(np.cumsum(np.power(np.diff(log_price_intraday), 2.0)) + sum_rv_i_1)
    rv_i_s += aux_rv
    end_date_rv_i_s.append(rv_i_s[-1])
    spot_vol += [spot_vol[-1]] + list(np.diff(np.cumsum(np.power(np.diff(log_price_intraday), 2.0))) * alpha_t_min)
    end_date_spot_i_s.append((rv_i_s[-1] - sum_rv_i_1) * 253.0)
    sum_rv_i_1 = rv_i_s[-1]
    sum_rv_i_1_fourier = rv_i_s_fourier[-1]
    i_cont += 1
days = np.arange(0, len(sampling_dates))
# rv estimator
plt.plot(days, end_date_rv_i_s, label="rv_estimator_integrated_variance", color="black")
plt.plot(days, rv_i_s_fourier, label="fourier_estimator_integrated_variance", color="black", linestyle="dashed")

# spot vol estimator
# plt.plot(days, end_date_spot_i_s, label="rv_estimator_spot_vol", color="black")
# plt.plot(days, spot_vol_i_s_fourier, label="fourier_estimator_spot_vol", color="black", linestyle="dashed")

plt.xlabel('t (traiding days)')
plt.legend()
plt.show()

