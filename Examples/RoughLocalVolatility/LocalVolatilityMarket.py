import os
import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib as plt

from pathlib import Path
from VolatilitySurface import TermStructureVolatility
from VolatilitySurface.Tools import SABRTools, ParameterTools
from AnalyticEngines.LocalVolatility.Dupire import NonParametricLV

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
except ImportError:
    print('The import of matplotlib is not working.')

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrParametersStoxx50e.txt')
vol_atm_path = os.path.join(folder_directory, 'Data', 'VolAtmSabrStoxx50e.txt')
forward_path = os.path.join(folder_directory, 'Data', 'ForwardsStoxx50e.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["parameter", "rho", "nu"], sep=";")
vol_atm = pd.read_csv(vol_atm_path, header=None, names=["value_date", "date", "vol"], sep=";")
forwards = pd.read_csv(forward_path, header=None, names=["value_date", "date", "forward"], sep=";")

no_dates = len(vol_atm['date'])

# We must build the term structure volatility
rho_parameter = list(parameters["rho"])
rho_parameter.remove("rho")
rho_parameter = list(map(lambda x: float(x), rho_parameter))
nu_parameter = list(parameters["nu"])
nu_parameter.remove("nu")
nu_parameter = list(map(lambda x: float(x), nu_parameter))

p_rho = np.array(rho_parameter)
p_nu = np.array(nu_parameter)

vol_atm_list = list(vol_atm["vol"])
forward_list = list(forwards["forward"])
dates_list = list(vol_atm["date"])

vol_atm_map = dict([(int(dates_list[i]), float(vol_atm_list[i])) for i in range(0, no_dates)])
forward_map = dict([(int(dates_list[i]), float(forward_list[i])) for i in range(0, no_dates)])
value_date = list(vol_atm["value_date"])[0]
ql_date = ql.Date(int(value_date))
day_counter = ql.Actual365Fixed()

sabr_term_structure_vol = TermStructureVolatility.SABRImpliedVolatilitySurface(ql_date, day_counter, rho_parameter,
                                                                               nu_parameter, vol_atm_map)
loc_vol = NonParametricLV.SABRLocalVol(sabr_term_structure_vol)

no_z_i_s = 50
z_i_s = np.linspace(-1.0, 1.0, no_z_i_s)

m_iv = np.zeros(shape=(no_dates, no_z_i_s))
m_loc_vol = np.zeros(shape=(no_dates, no_z_i_s))
delta_time = np.zeros(no_dates)

for i in range(0, no_dates):
    delta_time[i] = day_counter.yearFraction(ql_date, ql.Date(int(dates_list[i])))
    rho_i = ParameterTools.rho_sabr(p_rho, delta_time[i])
    nu_i = ParameterTools.nu_sabr(p_nu, delta_time[i])
    atm_vol_i = sabr_term_structure_vol.get_atm_volatility(ql.Date(int(dates_list[i])))
    alpha_i = ParameterTools.alpha_atm_sabr(rho_i, nu_i, atm_vol_i[0], delta_time[i])
    x_i_s = np.log(forward_map[int(dates_list[i])]) - z_i_s
    m_loc_vol[i, :] = loc_vol.get_vol(int(dates_list[i]), x_i_s, forward_map[int(dates_list[i])])
    for j in range(0, no_z_i_s):
        m_iv[i, j] = SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z_i_s[j], delta_time[i])

fig_surface = plt.figure(figsize=(13, 5))
ax = fig_surface.add_subplot(121, projection='3d')

t, z = np.meshgrid(delta_time, z_i_s)

surf = ax.plot_surface(t,
                       z,
                       m_iv.transpose(),
                       rstride=1,
                       cstride=1,
                       cmap=cm.gray,
                       linewidth=0,
                       antialiased=False)

ax.set_zlim(0.0, 3.0)
ax.set_xlabel('t(years)')
ax.set_ylabel('ln(F/K)')
ax.set_zlabel('volatility')
ax.set_title('STOX50E implied volatility surface')

ax = fig_surface.add_subplot(122, projection='3d')

surf_local_vol = ax.plot_surface(t,
                                 z,
                                 m_loc_vol.transpose(),
                                 rstride=1,
                                 cstride=1,
                                 cmap=cm.gray,
                                 linewidth=0,
                                 antialiased=False)
ax.set_zlim(0.0, 3.0)
ax.set_xlabel('t(years)')
ax.set_ylabel('ln(F/K)')
ax.set_zlabel('volatility')
ax.set_title('STOX50E local volatility surface')

plt.show()
