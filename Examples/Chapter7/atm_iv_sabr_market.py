import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path
from VolatilitySurface.Tools import SABRTools
from scipy.optimize import curve_fit

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
        iv.append(SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z_i[j], dti))
    sabr_iv_map[int(parameters['date'][i])] = iv

nu_param = []
rho_param = []
delta_time = []
for i in range(1, no_dates):
    delta_time.append((float(parameters['date'][i]) - float(parameters['value_date'][i])) / 365.0)
    nu_param.append(float(parameters['nu'][i]))
    rho_param.append(float(parameters['rho'][i]))

# To plot the skew for diferent maturities
plt.plot(delta_time, nu_param, label="atm implied volatility", color="black", linestyle="dashed")


def f_law(x, a, b):
    return a * np.power(x, -b)


popt, pcov = curve_fit(f_law, delta_time, nu_param)
y_fit_values = f_law(delta_time, *popt)

plt.plot(delta_time, y_fit_values, label="%s * t^-%s)" % (round(popt[0], 5), round(popt[1], 5)), color="black", linestyle="dashed",
         marker='.')

plt.plot("T")
plt.legend()
plt.show()

