import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

# simulation info
epsilon = 0.3
k = 0.5
rho = -0.6
v0 = 0.5
sigma_0 = np.sqrt(v0)
theta = 0.2

parameters = [k, theta, epsilon, rho, v0]
no_time_steps = 100
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T = 0.1

beta_vix = (1.0 - np.exp(- k * delta_vix)) / (delta_vix * k)
vix_t0 = np.sqrt((v0 - theta) * beta_vix + theta)

no_strikes = 30
strikes = np.linspace(0.98, 1.03, no_strikes) * vix_t0

# random number generator
rnd_generator = RNG.RndGenerator(seed)
map_output = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths, no_time_steps,
                                               Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

option_vix_price = []
implied_vol_vix = []
analytic_value = []

for i in range(0, no_strikes):

    index_t_i = np.searchsorted(map_output[Types.HESTON_OUTPUT.TIMES], T)
    price = np.mean(np.maximum(np.sqrt(beta_vix * (map_output[Types.HESTON_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i] -
                                                   theta) + theta) - strikes[i], 0.0))

    analytic_value.append(0.5 * (epsilon * sigma_0 / (k * vix_t0 * vix_t0)) * beta_vix)
    option_vix_price.append(price)
    implied_vol_vix.append(implied_volatility(option_vix_price[-1], vix_t0, strikes[i], T, 0.0, 0.0, 'c'))

plt.plot(strikes, implied_vol_vix, linestyle='--', label='ATM IV VIX', color='black', )
plt.xlabel('K')
plt.legend()
plt.show()
