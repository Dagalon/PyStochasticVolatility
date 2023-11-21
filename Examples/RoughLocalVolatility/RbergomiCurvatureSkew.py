import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from MC_Engines.MC_RBergomi import RBergomi_Variance_Engine
from Tools import Types
from Tools import RNG
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes import implied_volatility
from AnalyticEngines.VolatilityTools import RoughSabrAnalytic
from MC_Engines.MC_RBergomi import LocalVolRBegomi
from AnalyticEngines.MalliavinMethod import ExpansionTools

# simulation info
hurst = 0.499999
nu = 1.1
rho = -0.6
v0 = 0.09
sigma_0 = np.sqrt(v0)

epsilon = 0.5
spreads = np.linspace(- epsilon, epsilon, 3)

parameters = [nu, rho, hurst]

f0 = 100
K = 90.0
ts = np.linspace(0.01, 1.0 / 12.0, 30)
# ts = [0.1]

seed = 123

no_time_steps = 50
no_paths = 1000000

rnd_generator = RNG.RndGenerator(seed)
skew_iv = []
skew_iv_bs = []
curvature_iv = []
curvature_iv_bs = []
curvature_log_iv = []
# curvature_log_iv_bs = []
skew_lv = []
curvature_lv = []
curvature_log_lv = []
ratio_skew = []
ratio_curvature = []

for T in ts:
    # Options to compute the skew
    option_atm = EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    option_left = EuropeanOption(f0 + spreads[0], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    option_right = EuropeanOption(f0 + spreads[2], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

    rnd_generator = RNG.RndGenerator(seed)

    # Rbergomi paths simulation
    map_bergomi_output = RBergomi_Variance_Engine.get_path_multi_step(0.0, T, parameters, f0, sigma_0, no_paths,
                                                                      no_time_steps,
                                                                      Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                                      rnd_generator,
                                                                      True)

    # option prices
    price_atm = option_atm.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
    price_left = option_left.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
    price_right = option_right.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])

    vol_swap = np.sqrt(v0)
    iv_atm = implied_volatility.implied_volatility(price_atm[0], f0, f0, T, 0.0, 'c')
    iv_atm_approximation = ExpansionTools.get_iv_atm_rbergomi_approximation(parameters, vol_swap, sigma_0, T,
                                                                            'vol_swap')

    iv_left = implied_volatility.implied_volatility(price_left[0], f0, f0 + spreads[0], T, 0.0, 'c')
    iv_right = implied_volatility.implied_volatility(price_right[0], f0, f0 + spreads[2], T, 0.0, 'c')
    skew_iv_bs.append(0.5 * (iv_right - iv_left) / epsilon)

    # approximation of the implied volatility
    ks = np.array([f0 + s for s in spreads])
    iv_approximation = RoughSabrAnalytic.RBergomiImpliedVol(f0, ks, T, hurst, rho, nu, iv_atm, vol_swap)

    skew_iv.append(0.5 * (iv_approximation[2] - iv_approximation[0]) / epsilon)
    curvature_iv.append((iv_approximation[2] - 2.0 * iv_approximation[1] + iv_approximation[0]) / (epsilon * epsilon))
    curvature_iv_bs.append((iv_right - 2.0 * iv_atm + iv_left) / (epsilon * epsilon))
    curvature_log_iv.append(curvature_iv[-1] * f0 * f0 + skew_iv[-1] * f0)

    # approximation of the local vol
    lv_left = LocalVolRBegomi.get_local_vol(T, f0, f0 + spreads[0], rho,
                                            map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                            np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS],
                                                   1),
                                            np.sum(map_bergomi_output[
                                                       Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS],
                                                   1))

    lv_i = LocalVolRBegomi.get_local_vol(T, f0, f0, rho,
                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                         np.sum(map_bergomi_output[
                                                    Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    lv_right = LocalVolRBegomi.get_local_vol(T, f0, f0 + spreads[2], rho,
                                             map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                             np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS],
                                                    1),
                                             np.sum(map_bergomi_output[
                                                        Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS],
                                                    1))

    skew_lv_formula = LocalVolRBegomi.get_skew_local_vol(T, f0, f0, rho,
                                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS],1))

    skew_lv.append(skew_lv_formula)
    curvature_lv.append((lv_right - 2.0 * lv_i + lv_left) / (epsilon * epsilon))
    curvature_log_lv.append(curvature_lv[-1] * f0 * f0 + skew_lv[-1] * f0)

ratio_skew = [skew_iv_bs[i] / skew_lv[i] for i in range(0, len(skew_iv))]
ratio_curvature = [curvature_log_iv[i] / curvature_log_lv[i] for i in range(0, len(curvature_iv))]
target_skew = [1.0 / (hurst + 1.5) for i in range(0, len(ts))]
target_curvature = [1.0 / (2.0 * (hurst + 1.0)) for i in range(0, len(ts))]

#
# plt.plot(ts, curvature_log_iv, label="iv_curvature", color="black", linestyle="dashed")

# plt.scatter(ts, ratio_curvature, label="iv_curvature / lv_curvature", color="black", marker=".")
# plt.plot(ts, target_curvature, label="%s" % target_curvature[0], color="black", linestyle="dotted", marker="x")


# plt.plot(ts, curvature_log_lv, label="lv curvature", color="black", linestyle="dashdot")
# plt.plot(ts, curvature_log_iv, label="iv curvature", color="black", linestyle="dotted")
plt.scatter(ts, ratio_skew, label="ratio_skew", color="black", marker="o")
# plt.plot(ts, target_skew, label="target skew", color="black", linestyle="dotted")


def f_law(x, a, b):
    return a * np.power(x, b)


# skew
popt, pcov = curve_fit(f_law, ts, ratio_skew)
ratio_skew_fit = f_law(ts, *popt)
plt.plot(ts, ratio_skew_fit, label="%st^%s" % (round(popt[0], 5), round(popt[1], 5)), color="black",
          linestyle="dashdot", marker=".")


# popt, pcov = curve_fit(f_law, ts, curvature_log_lv)
# curvature_fit_log_lv = f_law(ts, *popt)
# plt.plot(ts, curvature_fit_log_lv, label="%st^%s" % (round(popt[0], 5), round(popt[1], 5)), color="black",
#          linestyle="dashdot", marker=".")
#
# popt, pcov = curve_fit(f_law, ts, curvature_log_iv)
# curvature_fit_log_iv = f_law(ts, *popt)
# plt.plot(ts, curvature_fit_log_iv, label="%st^%s" % (round(popt[0], 5), round(popt[1], 5)), color="black",
#          linestyle="dotted", marker="o")

# plt.scatter(ts, target_skew, label="1 / (H + 3/2)", color="black", marker="x")

# plt.plot(ts, ratio_curvature, label="curvature_iv / curvature_lv", color="black", linestyle="dotted", marker="x")

plt.xlabel('T')
plt.legend()
plt.show()
