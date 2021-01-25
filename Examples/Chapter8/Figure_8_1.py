import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import QuantLib as ql

from pathlib import Path
from VolatilitySurface.Tools import SABRTools

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrSurfaceParameter.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["value_date", "date", "alpha", "rho", "nu"], sep=";")
no_dates = len(parameters['date'])

no_z_i = 25
z_i = np.linspace(-0.5, 0.5, no_z_i)

sabr_iv_map = {}
for i in range(1, no_dates):
    alpha_i = float(parameters['alpha'][i])
    rho_i = float(parameters['rho'][i])
    nu_i = float(parameters['nu'][i])
    dti = (float(parameters['date'][i]) - float(parameters['value_date'][i])) / 365.0
    iv = []
    for j in range(0, no_z_i):
        vol_aux = SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z_i[j], dti)
        iv.append(SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z_i[j], dti))
    sabr_iv_map[int(parameters['date'][i])] = iv

nu_param = []
rho_param = []
delta_time = []
for i in range(1, no_dates):
    delta_time.append((float(parameters['date'][i]) - float(parameters['value_date'][i])) / 365.0)
    nu_param.append(float(parameters['nu'][i]))
    rho_param.append(float(parameters['rho'][i]))


index = np.arange(0, no_dates)

maturities = [44183, 44365, 44547, 44729, 44911, 45275, 46010, 46738, 47473]
no_maturities = len(maturities)

fig, axs = plt.subplots(3, 3, figsize=(20, 20))

for i in range(0, 3):
    date_str = str(ql.Date(maturities[i]))
    axs[0, i].plot(z_i, sabr_iv_map[maturities[i]],
                   linestyle='dashed', color='black', marker='.')

    axs[0, i].set_title(date_str, fontsize=18)
    axs[0, i].set_xlabel('ln(f/k)')
    axs[0, i].set_ylabel('iv')
    axs[0, i].set_ylim([0.0, 0.6])
    axs[0, i].axes.xaxis.label.set_size(20)
    axs[0, i].axes.yaxis.label.set_size(20)

for i in range(0, 3):
    date_str = str(ql.Date(maturities[i + 3]))
    axs[1, i].plot(z_i, sabr_iv_map[maturities[i + 3]],
                   linestyle='dashed', color='black', marker='.')

    axs[1, i].set_title(date_str, fontsize=18)
    axs[1, i].set_xlabel('ln(f/k)')
    axs[1, i].set_ylabel('iv')
    axs[1, i].set_ylim([0.0, 0.6])
    axs[1, i].axes.xaxis.label.set_size(20)
    axs[1, i].axes.yaxis.label.set_size(20)

for i in range(0, 3):
    date_str = str(ql.Date(maturities[i + 6]))
    axs[2, i].plot(z_i, sabr_iv_map[maturities[i + 6]],
                   linestyle='dashed', color='black', marker='.')

    axs[2, i].set_title(date_str, fontsize=18)
    axs[2, i].set_xlabel('ln(f/k)')
    axs[2, i].set_ylabel('iv')
    axs[2, i].set_ylim([0.0, 0.6])
    axs[2, i].axes.xaxis.label.set_size(20)
    axs[2, i].axes.yaxis.label.set_size(20)





# for i in range(0, len(index)):
#     date_str = str(ql.Date(int(parameters['date'][index[i] + 1])))
#     plt.plot(z_i, sabr_iv_map[int(parameters['date'][index[i] + 1])], label=parameters['date'][index[i] + 1],
#              linestyle='dashed', color='black')
#
#     plt.title(date_str)
#     plt.xlabel('ln(f/k)')
#     plt.ylabel('iv')
#     id_file = "plot_" + parameters['date'][index[i] + 1] + '.png'
#     # plt.savefig(os.path.join('D://GitRepository//Python//SV_Engines//Examples//Chapter8//',id_file))

plt.savefig(os.path.join('D://GitHubRepository//Python//PyStochasticVolatility//Examples//Chapter8//', 'cover.jpeg'),
            format='jpeg', dpi=1200)
plt.show()
