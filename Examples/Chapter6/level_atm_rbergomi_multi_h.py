import numpy as np
import csv
import multiprocessing as mp

from functools import partial
from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools

dt = np.arange(7, 180, 1) * 2.0 / 365.0
no_dt_s = len(dt)

# simulation info
#h_s = [0.1]
h_s = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
no_h_s = len(h_s)
nu = 0.5
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

no_paths = 500000
delta_time = 1.0 / 365.0
no_time_steps = 75


# option information
f0 = 100.0
options = []
implied_vol_atm = []
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))


# power law to fit
def f_law(x, b, c):
    return b * np.power(x, c)


def f_linear(x, b, c):
    return b + c * x


# outputs
variance_swap = []
vol_swap_approx = []
vol_swap_mc = []
implied_vol_atm = []
implied_vol_approx = []
output_vol_swap = []
output_variance_swap = []

h_output_log_estimated = []
h_output_power_law_estimated = []


def estimated_set_h_level_iv(hurst_parameter: float):
    # random number generator
    seed = 123456789
    rnd_generator = RNG.RndGenerator(seed)

    parameters = [nu, rho, hurst_parameter]

    # outputs for each h
    variance_swap = []
    vol_swap_approx = []
    vol_swap_mc = []
    implied_vol_atm = []
    implied_vol_approx = []
    output_vol_swap = []
    output_variance_swap = []

    for i in range(0, no_dt_s):
        map_output = RBergomi_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                         no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                         rnd_generator)

        mc_option_price = options[i].get_price(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])

        implied_vol_atm.append(implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c'))
        vol_swap_mc.append(
            np.mean(np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
        error_mc_vol_swap = np.std(
            np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])) / np.sqrt(no_paths)
        vol_swap_approx.append(ExpansionTools.get_vol_swap_rbergomi(parameters, sigma_0, dt[i]))
        implied_vol_approx.append(
            ExpansionTools.get_iv_atm_rbergomi_approximation(parameters, vol_swap_mc[i], sigma_0, dt[i], 'vol_swap'))
        variance_swap.append(ExpansionTools.get_variance_swap_rbergomi(parameters, sigma_0, dt[i]))
        output_vol_swap.append(implied_vol_atm[i] - vol_swap_mc[i])
        output_variance_swap.append(implied_vol_atm[i] - variance_swap[i])

    # csv parser
    headers = ["time", "iv_atm", "iv_atm_approx", "vol_swap_mc", "vol_swap_approx", "variance_swap",
               "out_variance_swap", "out_vol_swap"]
    rows = []
    for k in range(0, no_dt_s):
        rows.append({"time": str(dt[k]), "iv_atm": str(implied_vol_atm[k]),
                     "iv_atm_approx": str(implied_vol_approx[k]), "vol_swap_mc": str(vol_swap_mc[k]),
                     "vol_swap_approx": str(vol_swap_approx[k]), "variance_swap": str(variance_swap[k]),
                     "out_variance_swap": str(output_variance_swap[k]), "out_vol_swap": str(output_vol_swap[k])})

    id_file = "output_rbergomi_h_" + "".join(str(hurst_parameter).split('.')) + ".csv"
    file = open("D:/GitRepository/Python/SV_Engines/Examples/Chapter6%s" % id_file, 'w')
    csv_writer = csv.DictWriter(file, fieldnames=headers, lineterminator='\n', delimiter=';')
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    file.close()

    # curve fit
    # popt_linear_var_swap, pcov_linear_var_swap = curve_fit(f_law, np.log(dt), np.log(np.abs(output_variance_swap)))
    # y_fit_values_linear_var_swap = f_law(dt, *popt_linear_var_swap)
    #
    # if popt_linear_var_swap[1] > 1.0:
    #     h_output_log_estimated = popt_linear_var_swap[1] - 0.5
    # else:
    #     h_output_log_estimated = popt_linear_var_swap[1]
    #
    # popt_variance_swap, pcov_variance_swap = curve_fit(f_law, dt, output_variance_swap)
    # y_fit_values_variance_swap = f_law(dt, *popt_variance_swap)
    # h_output_power_law_estimated = popt_variance_swap[1]
    #
    # return hurst_parameter, h_output_log_estimated, h_output_power_law_estimated
    return hurst_parameter, id_file


if __name__ == '__main__':
    # parallel computing
    no_cores = mp.cpu_count()
    effective_no_cores = int(no_cores / 2.0) - 2
    pool = mp.Pool(effective_no_cores)
    outputs = pool.map(estimated_set_h_level_iv, h_s)
    pool.close()
    pool.join()

    # map_output = dict((out[0], [out[1], out[2]]) for out in outputs)
    #
    # id_file = "table_h_s.csv"
    # file_for_h_s = open("D:/GitHubRepository/Python/SV_Engines/Examples/Chapter6/%s" % id_file, 'w')
    # headers = ["real_value_h", "linear_h_estimated", "power_law_h_estimated"]
    #
    # rows = []
    # for k in range(0, no_h_s):
    #     rows.append({"real_value_h": h_s[k],
    #                  "linear_h_estimated": str(map_output[h_s[k]][0]),
    #                  "power_law_h_estimated": str(str(map_output[h_s[k]][1]))})
    #
    # csv_writer = csv.DictWriter(file_for_h_s, fieldnames=headers, lineterminator='\n', delimiter=';')
    # csv_writer.writeheader()
    # csv_writer.writerows(rows)
    # file_for_h_s.close()
