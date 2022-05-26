import os
import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib as plt

from pathlib import Path

from scipy.optimize import curve_fit

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
value_date = list(vol_atm["value_date"])[0]
ql_date = ql.Date(int(value_date))
day_counter = ql.Actual365Fixed()
vol_atm_map = dict([(int(dates_list[i]), float(vol_atm_list[i])) for i in range(0, no_dates)])

# forward interpolator
forward_map = dict([(int(dates_list[i]), float(forward_list[i])) for i in range(0, no_dates)])
t_forward = np.zeros(len(forward_map))
f_values = np.zeros(len(forward_map))

for i in range(0, len(forward_map)):
    t_forward[i] = day_counter.yearFraction(ql_date, ql.Date(dates_list[i]))
    f_values[i] = forward_map[dates_list[i]]


sabr_term_structure_vol = TermStructureVolatility.SABRImpliedVolatilitySurface(ql_date, day_counter, rho_parameter,
                                                                               nu_parameter, vol_atm_map)
loc_vol = NonParametricLV.SABRLocalVol(sabr_term_structure_vol)

no_z_i_s = 50
z_i_s = np.linspace(-1.0, 1.0, no_z_i_s)

jumps = [(dates_list[i] - ql_date.serialNumber()) for i in range(0, len(dates_list))]

jumps = np.arange(7, 180, 7)
no_jumps = len(jumps)
# no_jumps = 12
delta_time = np.zeros(no_jumps)

atm_iv = np.zeros(no_jumps)
atm_loc_vol = np.zeros(no_jumps)
skew_atm_iv = np.zeros(no_jumps)
skew_atm_lov_vol = np.zeros(no_jumps)

for i in range(0, no_jumps):
    end_date = ql.Date(int(ql_date.serialNumber() + jumps[i]))
    delta_time[i] = day_counter.yearFraction(ql_date, end_date)
    rho_i = ParameterTools.rho_sabr(p_rho, delta_time[i])
    nu_i = ParameterTools.nu_sabr(p_nu, delta_time[i])
    atm_vol_i = sabr_term_structure_vol.get_atm_volatility(end_date)
    alpha_i = ParameterTools.alpha_atm_sabr(rho_i, nu_i, atm_vol_i[0], delta_time[i])

    fwd_interp = np.interp(delta_time[i], t_forward, f_values)
    fwd_interp_shift_right = 1.01 * fwd_interp
    fwd_interp_shift_left = 0.99 * fwd_interp
    x_i_s = np.asfortranarray(np.log(fwd_interp))
    atm_loc_vol[i] = loc_vol.get_vol(end_date.serialNumber(), x_i_s, fwd_interp)

    # skew lv
    x_t_right = np.asfortranarray(np.log(fwd_interp_shift_right))
    x_t_left = np.asfortranarray(np.log(fwd_interp_shift_left))
    atm_loc_vol_right = SABRTools.get_sabr_loc_vol(p_nu, p_rho, atm_vol_i[0], atm_vol_i[1], delta_time[i], fwd_interp, x_t_right)[0]
    atm_loc_vol_left = SABRTools.get_sabr_loc_vol(p_nu, p_rho, atm_vol_i[0], atm_vol_i[1], delta_time[i], fwd_interp, x_t_left)[0]
    skew_atm_lov_vol[i] = (atm_loc_vol_right - atm_loc_vol_left) / (fwd_interp_shift_right - fwd_interp_shift_left)

    # skew iv
    # iv_right = sabr_term_structure_vol.get_impl_volatility(fwd_interp, fwd_interp_shift_right, end_date)
    # iv_left = sabr_term_structure_vol.get_impl_volatility(fwd_interp, fwd_interp_shift_left, end_date)

    epsilon = 0.001
    iv_right = SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, epsilon, delta_time[i])
    iv_left = SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, -epsilon, delta_time[i])
    # skew_atm_iv[i] = -(iv_right - iv_left) / (fwd_interp_shift_right - fwd_interp_shift_left)
    skew_atm_iv[i] = - 0.5 * (iv_right - iv_left) / (fwd_interp * epsilon)
    # atm_iv[i] = atm_vol_i[0]


def f_law(x, b, c):
    return b * np.power(x, c)


popt_loc_vol, pcov_loc_vol = curve_fit(f_law, delta_time, skew_atm_iv / skew_atm_lov_vol)
y_fit_loc_vol = f_law(delta_time, *popt_loc_vol)

plt.plot(delta_time,  skew_atm_iv / skew_atm_lov_vol, label='skew_atm_iv / skew_atm_lv', linestyle='--', color='green')

plt.plot(delta_time, y_fit_loc_vol, label=" %s * T^(%s)" % (round(popt_loc_vol[0], 5),
                                                            round(popt_loc_vol[1], 5)), color='red', linestyle="dotted")

plt.xlabel("T")
plt.legend()
plt.show()
