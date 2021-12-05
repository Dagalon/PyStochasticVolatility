import matplotlib.pylab as plt
import numpy as np
import csv

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools

dt = np.arange(0.1, 182, 2) * 1.0 / 365.0
no_dt_s = len(dt)

# simulation info
h = 0.4
nu = 0.5
rho = 0.0
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

seed = 123
# no_paths = 10

# delta_time = 1.0 / 365.0
# no_time_steps = 75

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
options = []
implied_vol_atm = []
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
variance_swap = []
vol_swap_approx = []
vol_swap_mc = []
implied_vol_atm = []
implied_vol_approx = []
output_vol_swap = []
output_variance_swap = []
aux_output = []
variance_swap_mc = []
diff_vol_swap_var_swap = []

for i in range(0, no_dt_s):
    if dt[i] < 0.1:
        no_time_steps = 100
        no_paths = 1000000
    else:
        no_time_steps = 100
        no_paths = 1000000

    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, sigma_0, no_paths,
                                                     no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                     rnd_generator)

    mc_option_price = options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_atm.append(implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c'))
    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    implied_vol_approx.append(
        ExpansionTools.get_iv_atm_rbergomi_approximation(parameters, vol_swap_mc[i], sigma_0, dt[i], 'var_swap'))
    error_mc_vol_swap = np.std(
        np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])) / np.sqrt(no_paths)
    vol_swap_approx.append(ExpansionTools.get_vol_swap_rbergomi(parameters, sigma_0, dt[i]))
    variance_swap.append(ExpansionTools.get_variance_swap_rbergomi(parameters, sigma_0, dt[i]))
    variance_swap_mc.append(np.sqrt(np.mean(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    output_vol_swap.append(implied_vol_atm[i] - vol_swap_mc[i])
    output_variance_swap.append(implied_vol_atm[i] - variance_swap[i])
    diff_vol_swap_var_swap.append(variance_swap[i] - vol_swap_mc[i])

# csv parser
headers = ["time", "iv_atm", "iv_atm_approx", "vol_swap_mc", "vol_swap_approx", "variance_swap", "out_variance_swap",
           "out_vol_swap"]
rows = []
for i in range(0, no_dt_s):
    rows.append({"time": str(dt[i]), "iv_atm": str(implied_vol_atm[i]),
                 "iv_atm_approx": str(implied_vol_approx[i]), "vol_swap_mc": str(vol_swap_mc[i]),
                 "vol_swap_approx": str(vol_swap_approx[i]), "variance_swap": str(variance_swap[i]),
                 "out_variance_swap": str(output_variance_swap[i]), "out_vol_swap": str(output_vol_swap[i])})
#
# file = open('D://GitRepository//Python//SV_Engines//Examples//Chapter6//output_rbergomi_h_05_rho_06.csv', 'w')
# csv_writer = csv.DictWriter(file, fieldnames=headers, lineterminator='\n')
# csv_writer.writeheader()
# csv_writer.writerows(rows)
# file.close()


# curve fit


def f_law(x, b, c):
    return b * np.power(x, c)


popt_diff_vol_swap, pcov_diff_vols_swap = curve_fit(f_law, dt, output_variance_swap)
y_fit_diff_values_vol_swap_var_swap = f_law(dt, *popt_diff_vol_swap)

# popt_vol_swap, pcov_vols_swap = curve_fit(f_law, dt, output_vol_swap)
# y_fit_values_vol_swap = f_law(dt, *popt_vol_swap)

# popt_variance_swap, pcov_variance_swap = curve_fit(f_law, dt, output_variance_swap)
# y_fit_values_variance_swap = f_law(dt, *popt_variance_swap)

#
# plt.plot(dt, output_vol_swap, label='I(t,f0) - E(v_t)', linestyle='--', color='black')
# plt.plot(dt, diff_vol_swap_var_swap, label='sqrt(var_swap_t) - E(v_t)', linestyle='--', color='black')
plt.plot(dt, output_variance_swap, label='I(0,f0) - sqrt(var_swap_0)', linestyle='--', color='black')
# plt.plot(dt, vol_swap_mc, label='E(v_t)', linestyle='--', marker='.')
# plt.plot(dt, implied_vol_atm, label='implied volatility atm', linestyle='--', marker='x')
# plt.plot(dt, y_fit_values_vol_swap, label="%s * t^%s" % (round(popt_vol_swap[0], 5), round(popt_vol_swap[1], 5)),
#          marker='.', linestyle='--', color='black')

# plt.plot(dt, y_fit_values_variance_swap, label=" %s * t^(%s)" % (round(popt_variance_swap[0], 5),
#          round(popt_variance_swap[1], 5)), marker='+', linestyle='--', color='black')

plt.plot(dt, y_fit_diff_values_vol_swap_var_swap, label=" %s * T^(%s)" % (round(popt_diff_vol_swap[0], 5),
         round(popt_diff_vol_swap[1], 5)), marker='+', linestyle='--', color='black')

# plt.plot(dt, implied_vol_approx, label='approximation iv', linestyle='--')
# plt.plot(dt, implied_vol_atm, label='mc iv', linestyle='--')

plt.xlabel('T')
plt.legend()
plt.show()
