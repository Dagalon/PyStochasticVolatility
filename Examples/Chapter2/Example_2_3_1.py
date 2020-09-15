import numpy as np

from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from typing import List
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import matplotlib.pylab as plt


def get_smile_for_differents_rho(rho_s: Types.ndarray,
                                 epsilon: float,
                                 k: float,
                                 v0: float,
                                 theta: float,
                                 european_options: List[EuropeanOption]):
    heston_iv = []
    no_rho_s = len(rho_s)
    for i in range(0, no_rho_s):
        rho_i = rho_s[i]
        rho_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta, rho_i, k, epsilon, v0, 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)
            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            rho_i_iv.append(iv)

        heston_iv.append((rho_i, rho_i_iv))

    return heston_iv


def get_smile_for_differents_epsilon(rho: float,
                                     epsilon_s: Types.ndarray,
                                     k: float,
                                     v0: float,
                                     theta: float,
                                     european_options: List[EuropeanOption]):
    heston_iv = []
    no_epsilon_s = len(epsilon_s)
    for i in range(0, no_epsilon_s):
        epsilon_i = epsilon_s[i]
        epsilon_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta, rho, k, epsilon_i, v0, 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)
            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            epsilon_i_iv.append(iv)

        heston_iv.append((epsilon_i, epsilon_i_iv))

    return heston_iv


def get_smile_for_differents_k(rho: float,
                               epsilon: float,
                               k_s: Types.ndarray,
                               v0: float,
                               theta: float,
                               european_options: List[EuropeanOption]):
    heston_iv = []
    no_k_s = len(k_s)
    for i in range(0, no_k_s):
        k_i = k_s[i]
        k_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta, rho, k_i, epsilon, v0, 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)
            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            k_i_iv.append(iv)

        heston_iv.append((k_i, k_i_iv))

    return heston_iv


def get_smile_for_differents_theta(rho: float,
                                   epsilon: float,
                                   k: float,
                                   v0: float,
                                   theta_s: Types.ndarray,
                                   european_options: List[EuropeanOption]):
    heston_iv = []
    no_v0_s = len(theta_s)
    for i in range(0, no_v0_s):
        theta_i = theta_s[i]
        theta_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta_i, rho, k, epsilon, v0, 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)
            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            theta_i_iv.append(iv)

        heston_iv.append((k_i, theta_i_iv))

    return heston_iv


def get_smile_for_differents_k(rho: float,
                               epsilon: float,
                               k_s: Types.ndarray,
                               v0: float,
                               theta: float,
                               european_options: List[EuropeanOption]):
    heston_iv = []
    no_k_s = len(k_s)
    for i in range(0, no_k_s):
        k_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta, rho, k_s[i], epsilon, v0, 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)

            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            k_i_iv.append(iv)

        heston_iv.append((k_i, k_i_iv))

    return heston_iv


def get_smile_for_differents_v0(rho: float,
                                epsilon: float,
                                k: float,
                                v0_s: Types.ndarray,
                                theta: float,
                                european_options: List[EuropeanOption]):
    heston_iv = []
    no_v0_s = len(v0_s)
    for i in range(0, no_v0_s):
        v0_i_iv = []
        for option in european_options:
            price = option.get_analytic_value(0.0, theta, rho, k, epsilon, v0_s[i], 0.0,
                                              model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                              compute_greek=False)

            if option._option_type == TypeEuropeanOption.CALL:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'c')
            else:
                iv = implied_volatility(price, option._spot, option._strike, option._delta_time, 0.0, 0.0, 'p')

            v0_i_iv.append(iv)

        heston_iv.append((k_i, v0_i_iv))

    return heston_iv


# parameters
epsilon = 0.1
k = 0.1
v0 = 0.05
theta = 0.1
rho = -0.75

# options
strikes = np.arange(40.0, 200.0, 10.0)
t_s = np.arange(0.25, 20, 0.5)
f0 = 100
T = 2.0
notional = 1.0
options_strike = []
options_time = []
for k_i in strikes:
    options_strike.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))

for t_i in t_s:
    options_time.append(EuropeanOption(f0, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i))


# v0 effect in the smile
# v0_s = [0.05, 0.1, 1.0, 2.0]
# style_markers = ['*', '.', 'x', '^']
# out_v0_s = get_smile_for_differents_v0(rho, epsilon, k, v0_s, theta, options_time)
#
# no_outputs = len(out_v0_s)
# for i in range(0, no_outputs):
#     plt.plot(t_s, out_v0_s[i][1], label="v0="+str(v0_s[i]), linestyle='dashed', color='black',
#              marker=style_markers[i])
#
# plt.xlabel("t")

# k effect in the smile
# k_s = [0.1, 0.5, 1.0, 2.0]
# style_markers = ['*', '.', 'x', '^']
# out_k_s = get_smile_for_differents_k(rho, epsilon, k_s, v0, theta, options_strike)
#
# no_outputs = len(out_k_s)
# for i in range(0, no_outputs):
#     plt.plot(strikes, out_k_s[i][1], label="k="+str(k_s[i]), linestyle='dashed', color='black',
#              marker=style_markers[i])
#
# plt.xlabel("strike")

# k effect in time dimension
# k_s = [0.1, 0.5, 1.0, 2.0]
# style_markers = ['*', '.', 'x', '^']
# out_k_s = get_smile_for_differents_k(rho, epsilon, k_s, v0, theta, options_time)
#
# no_outputs = len(out_k_s)
# for i in range(0, no_outputs):
#     plt.plot(t_s, out_k_s[i][1], label="k="+str(k_s[i]), linestyle='dashed', color='black',
#              marker=style_markers[i])
#
# plt.xlabel("t")

# rho effect in the smile
# rho_s = np.array([-0.9, -0.5, -0.25, -0.0])
# style_markers = ['*', '.', 'x', '^']
#
# out_rho_s = get_smile_for_differents_rho(rho_s, epsilon, k, v0, theta, options_strike)
#
# no_outputs = len(out_rho_s)
# for i in range(0, no_outputs):
#     plt.plot(strikes, out_rho_s[i][1], label="rho="+str(rho_s[i]), linestyle='dashed', color='black',
#              marker=style_markers[i])
#
# plt.xlabel("strike")

# epsilon effect in the smile
# epsilon_s = np.array([0.1, 0.3, 0.5, 0.9])
# style_markers = ['*', '.', 'x', '^']
# out_epsilon_s = get_smile_for_differents_epsilon(rho, epsilon_s, k, v0, theta, options_strike)
#
# no_outputs = len(out_epsilon_s)
# for i in range(0, no_outputs):
#     plt.plot(strikes, out_epsilon_s[i][1], label="epsilon="+str(epsilon_s[i]), linestyle='dashed',
#              color='black', marker=style_markers[i])
#
# plt.xlabel("strike")

# theta effect in the smile
theta_s = np.array([0.1, 0.07, 0.05, 0.03])
style_markers = ['*', '.', 'x', '^']
out_theta_s = get_smile_for_differents_theta(rho, epsilon, k, v0, theta_s, options_strike)
no_outputs = len(out_theta_s)
for i in range(0, no_outputs):
    plt.plot(strikes, out_theta_s[i][1], label="theta=" + str(theta_s[i]), linestyle='dashed',
             color='black', marker=style_markers[i])

plt.legend()
plt.show()