import numpy as np
import matplotlib.pylab as plt

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import Types
from Tools import RNG
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from MC_Engines.MC_RBergomi import LocalVolRBegomi
from py_vollib.black_scholes import implied_volatility, black_scholes

# simulation info
hurst = 0.49999
nu = 0.4
rho = -0.4
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, hurst]

f0 = 100
T = np.arange(15, 180, 10) * 1.0 / 360

seed = 123456789

no_time_steps = 30
no_paths = 1000000

atm_lv = []
atm_lv_skew_fd = []
atm_lv_skew = []
atm_iv_skew = []
atm_lv_skew_derive_estimator = []
var_swap = []
ratio = []
target_skew = []

rnd_generator = RNG.RndGenerator(seed)

for t_i in T:
    rnd_generator.set_seed(seed)

    bump = 0.01
    f_left = (1.0 - bump) * f0
    f_right = (1.0 + bump) * f0

    # Options to compute the skew
    option_left = EuropeanOption(f_left, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option = EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option_right = EuropeanOption(f_right, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)

    # Rbergomi paths simulation
    map_bergomi_output = RBergomi_Engine.get_path_multi_step(0.0, t_i, parameters, f0, sigma_0, no_paths,
                                                             no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                             rnd_generator)

    # check simulation
    forward = np.mean(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
    vol_mean = np.mean(map_bergomi_output[Types.RBERGOMI_OUTPUT.SPOT_VOLATILITY_PATHS][:, -1])

    # option prices
    left_price = option_left.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                       map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    right_price = option_right.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    price = option.get_price_control_variate(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                             map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    # iv_left = implied_volatility.implied_volatility(left_price[0], f0, f_left, t_i, 0.0, 'c')
    iv = implied_volatility.implied_volatility(price[0], f0, f0, t_i, 0.0, 'c')
    option_bs = black_scholes('c', f0, f0, t_i, 0.0, iv)
    # iv_right = implied_volatility.implied_volatility(right_price[0], f0, f_right, t_i, 0.0, 'c')

    # new estimation skew iv
    d2 = - 0.5 * iv * np.sqrt(t_i)
    der_k_bs = - f0 * ndtr(d2)
    vega_bs = f0 * np.sqrt(t_i) * normal_pdf(d2, 1.0, 0.0)
    der_price = f0 * 0.5 * (right_price[0] - left_price[0]) / (f_right - f_left)
    # skew_iv_i = f0 * 0.25 *(iv_right - iv_left) / (f_right - f_left)
    skew_iv_i = (der_price - der_k_bs) / vega_bs

    atm_iv_skew.append(skew_iv_i)

    # new lv markovian projection

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
                                             np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                             np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    skew = f0 * (lv_right - lv_left) / (f_right - f_left)

    skew_sv_mc = f0 * LocalVolRBegomi.get_skew_local_vol(t_i, f0, f0, rho,
                                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))
    atm_lv_skew.append(skew_sv_mc)
    atm_lv_skew_fd.append(skew)
    ratio.append(skew_iv_i / skew_sv_mc)
    target_skew.append(1.0/(hurst + 1.5))


def f_law(x, b, c):
    return b * np.power(x, c)


# popt_atm_lv_skew, pcov_diff_vols_swap = curve_fit(f_law, T, atm_lv_skew)
# popt_atm_lv_skew, pcov_diff_vols_swap = curve_fit(f_law, T, ratio)
# y_fit_atm_lv_skew = f_law(T, *popt_atm_lv_skew)
# skew_lv_rbergomi_fit = f_law(T, *popt_atm_lv_skew)

# plt.plot(T, ratio, label="skew_iv / skew_lv", color="blue", linestyle="dotted", marker="x")
# plt.plot(T, target_skew, label="1/(H + 3/2)", color="red", linestyle="dotted",marker="x")

# plt.plot(T, atm_lv_skew, label="skew_lv", color="blue", linestyle="dotted", marker="x")
# plt.plot(T, atm_lv_skew_fd, label="skew_lv_fd", color="red", linestyle="dotted", marker="x")
plt.plot(T, atm_iv_skew, label="skew_iv", color="orange", linestyle="dashdot", marker="x")

# plt.plot(T, skew_lv_rbergomi_fit, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
#          round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")

# plt.plot(T, ratio, label="skew ratio ", color="green", linestyle="dotted")
# plt.plot(T, y_fit_atm_lv_skew, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
#          round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")
# plt.plot(T, skew_ratio, label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
#          round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")


# plt.ylim((0.4, 0.6))
plt.xlabel("T")
plt.ylabel("Skew")


plt.legend()
plt.show()
