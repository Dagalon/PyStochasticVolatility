import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

# simulation info
h = 0.3
nu = 0.5
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

no_time_steps = 200
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T_VIX = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
# markers = ['.', '*', '^', '+', 'v', ',']


# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []

for i in range(0, len(T_VIX)):
    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, T_VIX[i], parameters, f0, sigma_0, no_paths, no_time_steps,
                                                    Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator,)
    t_i_vix = T_VIX[i] + delta_vix
    beta_vix = np.sqrt(np.exp(nu * nu * delta_vix) - 1) / np.sqrt(delta_vix) * nu
    vix_t0 = sigma_0 * beta_vix
    # strikes = np.linspace(0.8, 1.2) * vix_t0

    index_t_i = np.searchsorted(map_output[Types.SABR_OUTPUT.TIMES], T_VIX[i])
    price = np.mean(np.maximum(beta_vix * map_output[Types.SABR_OUTPUT.SIGMA_PATHS][:, index_t_i] - vix_t0, 0.0))
    option_vix_price.append(price)
    implied_vol_vix.append(implied_volatility(option_vix_price[-1], vix_t0, vix_t0, T_VIX[i], 0.0, 0.0, 'c'))

plt.plot(T_VIX, implied_vol_vix, linestyle='--',  label='ATM IV VIX', color='black', marker='.')
plt.legend()
plt.show()