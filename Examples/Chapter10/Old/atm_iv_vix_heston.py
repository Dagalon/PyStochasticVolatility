import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

# simulation info
epsilon = 0.3
k = 1.0
rho = -0.9
v0 = 0.05
sigma_0 = np.sqrt(0.05)
theta = 0.06

parameters = [k, theta, epsilon, rho, v0]
no_time_steps = 100
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T_VIX = [0.01, 0.025, 0.05, 0.075, 0.1]

# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []
analytic_value = []

for i in range(0, len(T_VIX)):
    rnd_generator.set_seed(seed)
    map_output = Heston_Engine.get_path_multi_step(0.0, T_VIX[i], parameters, f0, v0, no_paths, no_time_steps,
                                                   Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)
    t_i_vix = T_VIX[i] + delta_vix
    beta_vix = (1.0 - np.exp(- k * delta_vix)) / (delta_vix * k)
    vix_t0 = np.sqrt((v0 - theta) * beta_vix + theta)

    index_t_i = np.searchsorted(map_output[Types.HESTON_OUTPUT.TIMES], T_VIX[i])
    price = np.mean(np.maximum(np.sqrt(beta_vix * (map_output[Types.HESTON_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i] -
                                                   theta) + theta) - vix_t0, 0.0))

    analytic_value.append(0.5 * (epsilon * sigma_0 / (k * vix_t0 * vix_t0)) * beta_vix)
    option_vix_price.append(price)
    implied_vol_vix.append(implied_volatility(option_vix_price[-1], vix_t0, vix_t0, T_VIX[i], 0.0, 0.0, 'c'))

plt.plot(T_VIX, implied_vol_vix, linestyle='--',  label='ATM IV VIX', color='black')
plt.xlabel('T')
plt.legend()
plt.show()