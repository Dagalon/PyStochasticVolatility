import numpy as np
import matplotlib.pylab as plt

from MC_Engines.MC_RBergomi import RBergomi_Variance_Engine
from Tools import Types
from Tools import RNG
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from MC_Engines.MC_RBergomi import LocalVolRBegomi
from py_vollib.black_scholes import implied_volatility, black_scholes

# simulation info
hurst = 0.3
nu = 1.1
rho = -0.6
v0 = np.power(0.3, 2.0)
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, hurst]

f0 = 100
T = np.arange(7, 90, 5) * 1.0 / 360
# T = np.linspace(1/252, 0.1, 10)

seed = 123

no_time_steps = 10
no_paths = 1000000

atm_lv = []
atm_iv = []
atm_lv_skew_fd = []
atm_lv_skew = []
atm_iv_skew = []
atm_iv_fd_skew = []
atm_lv_skew_derive_estimator = []
var_swap = []
ratio = []
target_skew = []

rnd_generator = RNG.RndGenerator(seed)

for t_i in T:
    rnd_generator.set_seed(seed)

    bump = 0.001
    f_left = (1.0 - bump) * f0
    f_right = (1.0 + bump) * f0

    # Options to compute the skew
    option = EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option_left = EuropeanOption(f_left, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)
    option_right = EuropeanOption(f_right, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i)

    # Rbergomi paths simulation
    map_bergomi_output = RBergomi_Variance_Engine.get_path_multi_step(0.0, t_i, parameters, f0, sigma_0, no_paths,
                                                                      no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                                      rnd_generator)
    # option prices
    price = option.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
    price_left = option_left.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
    price_right = option_right.get_price(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])

    iv = implied_volatility.implied_volatility(price[0], f0, f0, t_i, 0.0, 'c')
    iv_left = implied_volatility.implied_volatility(price_left[0], f0, f_left, t_i, 0.0, 'c')
    iv_right = implied_volatility.implied_volatility(price_right[0], f0, f_right, t_i, 0.0, 'c')

    atm_iv.append(iv)

    # new estimation skew iv
    d2 = - 0.5 * iv * np.sqrt(t_i)
    der_k_bs = - ndtr(d2)
    vega_bs = np.sqrt(t_i) * normal_pdf(0.0, 1.0, d2)
    der_price = - price[2]
    skew_iv_i = (der_price - der_k_bs) / vega_bs
    skew_fd_i = f0 * (iv_right - iv_left) / (f_right - f_left)

    atm_iv_skew.append(skew_iv_i)
    atm_iv_fd_skew.append(skew_fd_i)

    # new lv markovian projection
    lv_i = LocalVolRBegomi.get_local_vol(t_i, f0, f0, rho,
                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    skew_sv_mc = f0 * LocalVolRBegomi.get_skew_local_vol(t_i, f0, f0, rho,
                                                         map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                                         np.sum(map_bergomi_output[Types.RBERGOMI_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))
    atm_lv.append(lv_i)
    atm_lv_skew.append(skew_sv_mc)
    ratio.append(skew_iv_i / skew_sv_mc)
    target_skew.append(1.0/(hurst + 1.5))


# plot aux
plt.scatter(T, atm_iv_skew, label="atm skew formula", color="blue", marker="x")
plt.scatter(T, atm_iv_fd_skew, label="atm skew fd", color="green", marker="o")
plt.xlabel("T")
plt.legend()
plt.title("nu=" + str(nu) + "and H=" + str(hurst))
plt.show()

# plots
# plt.scatter(T, ratio, label="skew_iv / skew_lv", color="blue", marker="o")
# plt.plot(T, target_skew, label="1/(H + 3/2)", color="red", linestyle="dotted", marker="x")
# plt.xlabel("T")
# plt.legend()
# plt.savefig("C:/Users/david/OneDrive/Desktop/plots/ratio_h_%s.jpg" % hurst)
# plt.figure().clear()
# plt.close()
# plt.cla()
# plt.clf()
#
# plt.plot(T, atm_lv, label="atm_lv", color="blue", linestyle="dotted", marker="x")
# plt.plot(T, atm_iv, label="atm_iv", color="red", linestyle="dotted", marker="x")
# plt.xlabel("T")
# plt.legend()
# plt.savefig("C:/Users/david/OneDrive/Desktop/plots/lv_vs_iv_h_%s.jpg" % hurst)
# plt.figure().clear()
# plt.close()
# plt.cla()
# plt.clf()
#
# plt.plot(T, atm_lv_skew, label="skew_lv", color="blue", linestyle="dotted", marker="x")
# plt.plot(T, atm_iv_skew, label="skew_iv", color="orange", linestyle="dashdot", marker="x")
# plt.xlabel("T")
# plt.legend()
# plt.savefig("C:/Users/david/OneDrive/Desktop/plots/skew_lv_vs_iv_h_%s.jpg" % hurst)
# plt.figure().clear()
# plt.close()
# plt.cla()
# plt.clf()
