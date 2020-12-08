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
no_time_steps = 10
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T = 0.1

beta_vix = np.sqrt(np.exp(nu * nu * delta_vix) - 1) / (np.sqrt(delta_vix) * nu)
vix_t0 = alpha * beta_vix

no_strikes = 30
strikes = np.linspace(0.8, 1.2, no_strikes) * vix_t0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []

for i in range(0, no_strikes):
    rnd_generator.set_seed(seed)
    map_output = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

    index_t_i = np.searchsorted(map_output[Types.SABR_OUTPUT.TIMES], T)
    price = np.mean(np.maximum(beta_vix * map_output[Types.SABR_OUTPUT.SIGMA_PATHS][:, index_t_i] - strikes[i], 0.0))
    option_vix_price.append(price)
    implied_vol_vix.append(round(implied_volatility(option_vix_price[-1], vix_t0, strikes[i], T, 0.0, 0.0, 'c'), 5))

plt.plot(strikes, implied_vol_vix, linestyle='--',  label='Implied Vol VIX', color='black', marker='.')
plt.ylim([0.5, 1.0])
plt.xlabel('K')
plt.legend()
plt.show()