import numpy as np
import matplotlib.pylab as plt
import os
import pandas as pd
import QuantLib as ql

from pathlib import Path

from scipy.optimize import curve_fit

from VolatilitySurface import TermStructureVolatility
from VolatilitySurface.Tools import SABRTools, ParameterTools
from VolatilitySurface.Tools import SABRTools

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
sabr_parameter_paths = os.path.join(folder_directory, 'Data', 'SabrParametersStoxx50e.txt')
vol_atm_path = os.path.join(folder_directory, 'Data', 'VolAtmSabrStoxx50e.txt')
forward_path = os.path.join(folder_directory, 'Data', 'ForwardsStoxx50e.txt')

parameters = pd.read_csv(sabr_parameter_paths, header=None, names=["parameter", "rho", "nu"], sep=";")
vol_atm = pd.read_csv(vol_atm_path, header=None, names=["value_date", "date", "vol"], sep=";")
forwards = pd.read_csv(forward_path, header=None, names=["value_date", "date", "forward"], sep=";")

# build the parameter from the functional form
rho_parameter = list(parameters["rho"])
rho_parameter.remove("rho")
rho_parameter = list(map(lambda x: float(x), rho_parameter))
nu_parameter = list(parameters["nu"])
nu_parameter.remove("nu")
nu_parameter = list(map(lambda x: float(x), nu_parameter))

p_rho = np.array(rho_parameter)
p_nu = np.array(nu_parameter)

no_dates = len(vol_atm['date'])

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

# skew local vol and implied vol
delta_time = []
skew_atm = []
skew_lv = []
skew_iv = []
ratio = []
epsilon = 0.5
spreads = np.linspace(-epsilon, epsilon, 3)

for i in range(0, no_dates):
    delta_time.append(day_counter.yearFraction(ql_date, ql.Date(int(dates_list[i]))))
    rho_i = ParameterTools.rho_sabr(p_rho, delta_time[i])
    nu_i = ParameterTools.nu_sabr(p_nu, delta_time[i])
    atm_vol_i = sabr_term_structure_vol.get_atm_volatility(ql.Date(int(dates_list[i])))
    alpha_i = ParameterTools.alpha_atm_sabr(rho_i, nu_i, atm_vol_i[0], delta_time[i])
    f0 = forwards["forward"][i]
    strikes = [f0 + s for s in spreads]
    zi = np.log(np.divide(f0, strikes))

    # skew lv
    skew_lv.append(rho_i * nu_i / f0)

    # skew iv
    iv_hagan = [SABRTools.sabr_vol_jit(alpha_i, rho_i, nu_i, z, delta_time[i]) for z in zi]
    skew_atm.append(nu_i)
    skew_iv.append(0.5 * (iv_hagan[2] - iv_hagan[0]) / epsilon)

    ratio.append(skew_iv[i] / skew_lv[i])


def f_law(x, a, b):
    return a * np.power(x, b)


popt, pcov = curve_fit(f_law, delta_time, ratio)
ratio_fit = f_law(delta_time, *popt)
plt.plot(delta_time, ratio_fit, label="%st^%s" % (round(popt[0], 5), round(popt[1], 5)), color="green",
         linestyle="dashdot", marker=".")

plt.scatter(delta_time, ratio, label="skew_iv / skew_lv", color="blue", linestyle="dotted", marker="o")

plt.ylim([0.45, 0.51])
plt.legend()
plt.show()

