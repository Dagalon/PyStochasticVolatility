import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

dt = np.arange(7, 30, 1) * 1.0 / 365.0
no_dt_s = len(dt)

# simulation info
nu = 0.8
rho = -0.2
v0 = 0.05
sigma_0 = np.sqrt(v0)

seed = 123456789
no_paths = 500000
no_time_steps = 100

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
T = 0.1
k_s = np.arange(80.0, 120.0, 2.0)
h_s = np.arange(0.05, 0.45, 0.05)
no_k_s = len(k_s)

options = []
for k in k_s:
    options.append(EuropeanOption(k, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))

for h in h_s:
    parameters = [nu, rho, h]
    rnd_generator.set_seed(seed)

    map_output = RBergomi_Engine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths,
                                                     no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                     rnd_generator)
    # output
    implied_vol = []
    for i in range(0, no_k_s):
        mc_option_price = options[i].get_price(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
        implied_vol.append(implied_volatility(mc_option_price[0], f0, k_s[i], T, 0.0, 0.0, 'c'))

    plt.plot(k_s, implied_vol, label="H=%s" % round(h, 5), linestyle='--', marker='x')


plt.xlabel('strike')
plt.ylabel('iv')
plt.legend()
plt.show()