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
h = 0.05
sigma_0 = np.sqrt(v0)
parameters = [nu, rho, h]

seed = 123456789
no_paths = 500000
no_time_steps = 100

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
T_s = np.arange(0.01, 0.101, 0.02)
markers = ['.', '+', 'x', '^', '*']
k_s = np.arange(85.0, 115.0, 2.0)

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
        mc_option_price = options[j].get_price(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])
        implied_vol.append(implied_volatility(mc_option_price[0], f0, k_s[j], T_s[i], 0.0, 0.0, 'c'))

    plt.plot(k_s, implied_vol, label="T=%s" % round(T_s[i], 5), linestyle='--', marker=markers[i], color='black')
    print("Vamos por el paso t_i %s" % T_s[i])

plt.title("H=%s" % h)
plt.xlabel('t')
plt.ylabel('iv')
plt.legend()
plt.show()