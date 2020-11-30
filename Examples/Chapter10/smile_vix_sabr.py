import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

# simulation info
alpha = 0.5
nu = 0.7
rho = -0.4
parameters = [alpha, nu, rho]
no_time_steps = 200
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T_VIX = [0.01, 0.025, 0.05, 0.075, 0.1]
# markers = ['.', '*', '^', '+', 'v', ',']


# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []

for i in range(0, len(T_VIX)):
    rnd_generator.set_seed(seed)
    map_output = SABR_Engine.get_path_multi_step(0.0, T_VIX[i], parameters, f0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)
    t_i_vix = T_VIX[i] + delta_vix
    beta_vix = np.sqrt(np.exp(nu * nu * delta_vix) - 1) / np.sqrt(delta_vix) * nu
    vix_t0 = alpha * beta_vix
    # strikes = np.linspace(0.8, 1.2) * vix_t0

    index_t_i = np.searchsorted(map_output[Types.SABR_OUTPUT.TIMES], T_VIX[i])
    price = np.mean(np.maximum(beta_vix * map_output[Types.SABR_OUTPUT.SIGMA_PATHS][:, index_t_i] - vix_t0, 0.0))
    option_vix_price.append(price)
    implied_vol_vix.append(round(implied_volatility(option_vix_price[-1], vix_t0, vix_t0, T_VIX[i], 0.0, 0.0, 'c'), 5))

plt.plot(T_VIX, implied_vol_vix, linestyle='--',  label='ATM IV VIX', color='black', marker='.')
plt.xlabel('T')
plt.legend()
plt.show()