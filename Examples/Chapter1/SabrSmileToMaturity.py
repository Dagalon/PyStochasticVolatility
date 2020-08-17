import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path
from Tools import SABRTool

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrSurfaceParameter.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["value_date", "date", "alpha", "rho", "nu"], sep=";")
no_dates = len(parameters['date'])

no_z_i = 100
z_i = np.linspace(-2.0, 2.0, no_z_i)

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

for i in range(1, no_dates):
    plt.plot(z_i, sabr_iv_map[int(parameters['date'][i])], label=parameters['date'][i])

plt.legend()
plt.title('Sabr smile evolution with calibration to market')
plt.show()