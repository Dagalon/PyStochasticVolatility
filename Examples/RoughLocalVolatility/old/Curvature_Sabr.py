import numpy as np
import matplotlib.pylab as plt

from MC_Engines.MC_SABR import SABR_Engine
from Tools import Types
from Tools import RNG
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes import implied_volatility
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf, bs_density
from MC_Engines.MC_RBergomi import LocalVolRBegomi

# simulation info
v0 = np.power(0.3, 2.0)
sigma_0 = np.sqrt(v0)
alpha = sigma_0
nu = 0.000001
rho = 0.0

parameters = [alpha, nu, rho]

f0 = 1.0
T = np.arange(1, 90, 5) * 1.0 / 360
# T = np.linspace(1/252, 0.1, 10)

seed = 456

no_time_steps = 2
no_paths = 1000000

atm_lv = []
atm_iv = []
atm_lv_curvature_fd = []
atm_iv_curvature = []

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

    map_sabr_output = SABR_Engine.get_path_multi_step(0.0, t_i, parameters, f0, no_paths,
                                                      no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                      rnd_generator)

    # check simulation
    forward = np.mean(map_sabr_output[Types.SABR_OUTPUT.PATHS][:, -1])
    sigma_mean = np.mean(map_sabr_output[Types.SABR_OUTPUT.SIGMA_PATHS][:, -1])

    int_sigma = (map_sabr_output[Types.SABR_OUTPUT.SIGMA_PATHS][:, -1] - sigma_0) / nu
    m1 = np.mean(int_sigma)
    m2 = np.mean(np.power(int_sigma, 2.0))
    y = np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1)
    y1 = y.mean()
    y2 = np.mean(np.power(y, 2.0))

    vol_swap = np.mean(np.sqrt(np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / t_i))

    # option prices
    left_price = option_left.get_price(map_sabr_output[Types.SABR_OUTPUT.PATHS][:, -1])

    right_price = option_right.get_price(map_sabr_output[Types.SABR_OUTPUT.PATHS][:, -1])

    price = option.get_price(map_sabr_output[Types.SABR_OUTPUT.PATHS][:, -1])

    iv_left = implied_volatility.implied_volatility(left_price[0], f0, f_left, t_i, 0.0, 'c')
    iv = implied_volatility.implied_volatility(price[0], f0, f0, t_i, 0.0, 'c')
    iv_right = implied_volatility.implied_volatility(right_price[0], f0, f_right, t_i, 0.0, 'c')

    atm_iv.append(iv)

    # derive of BS
    d2 = - 0.5 * iv * np.sqrt(t_i)
    d1 = 0.5 * iv * np.sqrt(t_i)
    bs_k = - ndtr(d2)
    bs_sigma = np.sqrt(t_i) * f0 * normal_pdf(0.0, 1.0, d2)
    bs_2_sigma = np.sqrt(t_i) * f0 * normal_pdf(0.0, 1.0, d2) * d1 * d2 / iv
    bs_k_sigma = normal_pdf(0.0, 1.0, d2) * (np.sqrt(t_i) + (d2 / iv))

    # skew
    der_price = - price[2]
    # skew_iv_i = (der_price - bs_k) / bs_sigma
    skew_iv_i = f0 * (iv_right - iv_left) / (f_right - f_left)

    # density function
    pdf_sabr = LocalVolRBegomi.get_pdf(t_i, f0, f0, rho,
                                       map_sabr_output[Types.SABR_OUTPUT.VARIANCE_PATHS][:, -1],
                                       np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                       np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS],1))

    pdf_bs = bs_density(0.0, 0.0, t_i, iv, f0, f0)

    # new lv markovian projection
    lv_left = LocalVolRBegomi.get_local_vol(t_i, f0, f_left, rho,
                                            map_sabr_output[Types.SABR_OUTPUT.VARIANCE_PATHS][:, -1],
                                            np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                            np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS],1))

    lv_i = LocalVolRBegomi.get_local_vol(t_i, f0, f0, rho,
                                         map_sabr_output[Types.SABR_OUTPUT.VARIANCE_PATHS][:, -1],
                                         np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                         np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    lv_right = LocalVolRBegomi.get_local_vol(t_i, f0, f_right, rho,
                                             map_sabr_output[Types.SABR_OUTPUT.VARIANCE_PATHS][:, -1],
                                             np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                             np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    skew_mc = LocalVolRBegomi.get_skew_local_vol(t_i, f0, f0, rho,
                                                 map_sabr_output[Types.SABR_OUTPUT.VARIANCE_PATHS][:, -1],
                                                 np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1),
                                                 np.sum(map_sabr_output[Types.SABR_OUTPUT.INTEGRAL_SIGMA_PATHS_RESPECT_BROWNIANS], 1))

    # iv curvature
    diff_pdf = (pdf_sabr - pdf_bs)
    diff_skew = (der_price - bs_k)
    # curvature_iv_i = (diff_pdf - 2.0 * bs_k_sigma * skew_iv_i - bs_2_sigma * skew_iv_i * skew_iv_i) / bs_sigma
    curvature_iv_i = (diff_pdf - (diff_skew / f0) - (iv * iv * t_i / bs_sigma) * diff_skew * diff_skew) / bs_sigma
    curvature_fd_iv_i = (iv_right - 2.0 * iv + iv_left) / np.power(0.5 * (f_right - f_left), 2.0)
    # log_curvature_iv_i = curvature_iv_i * f0 * f0
    log_curvature_iv_i = f0 * skew_iv_i + curvature_iv_i * f0 * f0

    # lv curvature
    log_lv_curvature_i = f0 * f0 * (lv_right - 2.0 * lv_i + lv_left) / np.power(f_right - f_left, 2.0) + skew_mc * f0
    # log_curvature_i = f0 * f0 * (lv_right - 2.0 * lv_i + lv_left) / np.power(f_right - f_left, 2.0)

    ratio = log_curvature_iv_i / log_lv_curvature_i

    atm_lv_curvature_fd.append(log_lv_curvature_i)
    atm_lv.append(lv_i)

