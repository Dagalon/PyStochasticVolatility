import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import QuantLib as ql

from pathlib import Path
from Tools import SABRTool

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrSurfaceParameter.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["value_date", "date", "alpha", "rho", "nu"], sep=";")
no_dates = len(parameters['date'])

no_z_i = 100
z_i = np.linspace(-0.5, 0.5, no_z_i)

sabr_iv_map = {}
for i in range(1, no_dates):
    alpha_i = float(parameters['alpha'][i])
    rho_i = float(parameters['rho'][i])
    nu_i = float(parameters['nu'][i])
    dti = (float(parameters['date'][i]) - float(parameters['value_date'][i])) / 365.0
    iv = []
    for j in range(0, no_z_i):
        iv.append(SABRTool.ln_hagan_vol(alpha_i, rho_i, nu_i, z_i[j], dti))
    sabr_iv_map[int(parameters['date'][i])] = iv

nu_param = []
rho_param = []
delta_time = []
for i in range(1, no_dates):
    delta_time.append((float(parameters['date'][i]) - float(parameters['value_date'][i])) / 365.0)
    nu_param.append(float(parameters['nu'][i]))
    rho_param.append(float(parameters['rho'][i]))

# plt.plot(delta_time, nu_param, label='vol-of-vol parameter', linestyle='dashed', color='black')
# plt.xlabel('t')
# # plt.plot(delta_time, rho_param, label='correlation parameter')
#
# plt.legend()
# plt.show()


fig, axs = plt.subplots(2, 3)
index = [0, 3, 6, 8, 12, 18]
for i in range(0, 3):
    date_str = str(ql.Date(int(parameters['date'][index[i] + 1])))
    axs[0, i].set(xlabel='z', ylabel='iv')
    axs[0, i].plot(z_i, sabr_iv_map[int(parameters['date'][index[i] + 1])], label=parameters['date'][index[i] + 1],
                   linestyle='dashed', color='black')
    axs[0, i].set_title(date_str)

for i in range(0, 3):
    date_str = str(ql.Date(int(parameters['date'][index[i + 3] + 1])))
    axs[1, i].set(xlabel='z', ylabel='iv')
    axs[1, i].plot(z_i, sabr_iv_map[int(parameters['date'][index[i + 3] + 1])],
                   label=parameters['date'][index[i + 3] + 1],  linestyle='dashed', color='black')
    axs[1, i].set_title(date_str)

plt.show()