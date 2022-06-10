import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import Types
from Tools import RNG
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from MC_Engines.MC_RBergomi import LocalVolRBegomi
from py_vollib.black_scholes import implied_volatility

# simulation info
hurst = 0.4999
nu = 0.5
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, hurst]

f0 = 100
T = np.arange(7, 120, 2) * 1.0 / 365.0

seed = 123456789

no_time_steps = 100
no_paths = 100000

atm_lv = []
atm_lv_skew = []
atm_iv_skew = []
atm_lv_skew_derive_estimator = []
var_swap = []
ratio = []

rnd_generator = RNG.RndGenerator(seed)

for t_i in T:
    rnd_generator.set_seed(seed)
    # Julien Guyon paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032&download=yes
    h = 1.5 * np.sqrt(t_i) * np.power(no_paths, - 0.2)
    delta = h / 10.0

    bump = 0.001
    f_left = (1.0 - bump) * f0
    f_right = (1.0 + bump) * f0

    # Options to compute the skew
    option_left = EuropeanOption(f_left, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option_option = EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option_right = EuropeanOption(f_right, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)

    # Rbergomi paths simulation
    map_bergomi_output = RBergomi_Engine.get_path_multi_step(0.0, t_i, parameters, f0, v0, no_paths,
                                                             no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                             rnd_generator)

    # option prices
    left_price = option_left.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                       map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    right_price = option_left.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                        map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    price = option_left.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                  map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    iv_left = implied_volatility.implied_volatility(left_price[0], f0, f_left, t_i, 0.0, 'c')
    iv = implied_volatility.implied_volatility(price[0], f0, f_left, t_i, 0.0, 'c')
    iv_right = implied_volatility.implied_volatility(right_price[0], f0, f_right, t_i, 0.0, 'c')

    skew_iv_i = 0.5 * (iv_right - iv_left) / (f_right - f_left)

    atm_iv_skew.append(skew_iv_i)

    lv_left = LocalVolRBegomi.get_local_vol(t_i, f0, f_left, rho,
                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    lv_i = LocalVolRBegomi.get_local_vol(t_i, f0, f0, rho,
                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    lv_right = LocalVolRBegomi.get_local_vol(t_i, f0, f_right, rho,
                                            map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                            np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS],1),
                                            np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    var_swap.append(np.sqrt(np.mean(map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1])))

    skew = 0.5 * (lv_right - lv_left) / (f_right - f_left)

    skew_sv_mc = LocalVolRBegomi.get_skew_local_vol(t_i, f0, f0, rho,
                                                    map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                                    np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                                    np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))
    atm_lv_skew.append(skew_sv_mc)
    ratio.append(skew_iv_i / skew_sv_mc)


def f_law(x, b, c):
    return b * np.power(x, c)


popt_atm_lv_skew, pcov_diff_vols_swap = curve_fit(f_law, T, atm_lv_skew)
# popt_atm_lv_skew, pcov_diff_vols_swap = curve_fit(f_law, T, ratio)
y_fit_atm_lv_skew = f_law(T, *popt_atm_lv_skew)
skew_lv_rbergomi_fit = f_law(T, *popt_atm_lv_skew)

plt.plot(T, atm_lv_skew, label="skew LV rBergomi", color="blue", linestyle="dotted")
plt.plot(T, skew_lv_rbergomi_fit, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
         round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")

# plt.plot(T, ratio, label="skew ratio ", color="green", linestyle="dotted")
# plt.plot(T, y_fit_atm_lv_skew, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
#          round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")
# plt.plot(T, skew_ratio, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
#          round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")

plt.xlabel("T")
plt.ylabel("Skew")


plt.legend()
plt.show()
