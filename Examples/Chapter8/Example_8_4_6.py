import matplotlib.pylab as plt
import numpy as np
import os

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

T = np.array([3, 7, 15, 30, 60, 90]) * 1.0 / 365.0
labels = ['3 days', '7 days', '15 days', '1 month', '2 months', '3 months']
markers = ['.', '^', '+', '*', 'v', ',', '>']
no_maturities = len(T)
strikes = np.linspace(80.0, 120.0, 30)

# simulation info
h = 0.1
nu = 0.8
rho = -0.4
v0 = 0.15
sigma_0 = np.sqrt(v0)

# Mc info
no_paths = 1000000

parameters = [nu, rho, h]

# random number generator
seed = 123
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0

for i in range(0, no_maturities):
    options = []
    iv_smile = []
    rnd_generator.set_seed(seed)
    no_time_steps = np.maximum(int(T[i] * 365.0), 5)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, T[i], parameters, f0, sigma_0, no_paths,
                                                     no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                     rnd_generator)
    for k_i in strikes:
        options.append(EuropeanOption(k_i, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T[i]))
        mc_option_price = options[-1].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                               map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

        iv_smile.append(implied_volatility(mc_option_price[0], f0, k_i, T[i], 0.0, 0.0, 'c'))

    plt.plot(strikes, iv_smile, label=labels[i], color='black', marker=markers[i], linestyle="--")
    path_to_keep = os.path.join("D://GitHubRepository//Python//Graficos//Chapter8", "rbergomi_smile_%s" % i + ".png")

    plt.xlabel('K', fontsize=14)
    plt.legend(fontsize=14)
    plt.ylim([0.0, 0.9])
    plt.savefig(path_to_keep)
    plt.clf()

plt.show()
