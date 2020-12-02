import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility


# simulation info
nu = 0.8
rho = -0.4
v0 = 0.15
h = 0.1
sigma_0 = np.sqrt(v0)
parameters = [nu, rho, h]

seed = 123456789
no_paths = 1000000
no_time_steps = 100

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
T_s = [0.1, 0.2, 0.4, 0.6, 1.0]
markers = ['.', '+', '*', '^', 'v']
no_strikes = 22
k_s = np.linspace(80.0, 120.0, no_strikes)

no_k_s = len(k_s)
no_T_s = len(T_s)

options = []
for t_i in T_s:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t_i))

for i in range(0, no_T_s):

    rnd_generator.set_seed(seed)

    map_output = RBergomi_Engine.get_path_multi_step(0.0, T_s[i], parameters, f0, v0, no_paths,
                                                     no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                     rnd_generator)

    options = []
    for k_i in k_s:
        options.append(EuropeanOption(k_i, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T_s[i]))

    implied_vol = []
    for j in range(0, no_k_s):
        mc_option_price = options[j].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                               map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])
        implied_vol.append(implied_volatility(mc_option_price[0], f0, k_s[j], T_s[i], 0.0, 0.0, 'c'))

    plt.plot(k_s, implied_vol, label="T=%s" % round(T_s[i], 5), linestyle='--', marker=markers[i], color='black')

plt.xlabel('K')
plt.ylabel('iv')
plt.legend()
plt.show()
