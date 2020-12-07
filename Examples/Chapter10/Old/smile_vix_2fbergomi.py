import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Bergomi2F import Bergomi2fEngine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

# simulation info
nu_x = 0.5
nu_y = 0.6
rho_xy = 0.99
rho_xf = -0.5
rho_yf = -0.5
theta = 0.7
parameters = [theta, nu_x, nu_y, rho_xy, rho_xf, rho_yf]


f0 = 100
v0 = 0.25

seed = 123456789
no_paths = 1000000
no_time_steps = 100

# VIX spot
vix_t0 = np.sqrt(v0)

delta_vix = 1.0 / 12.0
T = 0.1

no_strikes = 30
strikes = np.linspace(0.98, 1.03, no_strikes) * vix_t0

# random number generator
rnd_generator = RNG.RndGenerator(seed)
map_output = Bergomi2fEngine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

# Check model is good
mean_var = np.mean(map_output[Types.BERGOMI2F_OUTPUT.SPOT_VARIANCE_PATHS], axis=0)
mean_asset = np.mean(map_output[Types.BERGOMI2F_OUTPUT.PATHS], axis=0)

option_vix_price = []
implied_vol_vix = []
analytic_value = []

for i in range(0, no_strikes):

    index_t_i = np.searchsorted(map_output[Types.BERGOMI2F_OUTPUT.TIMES], T)
    price = np.mean(np.maximum(np.sqrt(map_output[Types.BERGOMI2F_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i])
                               - strikes[i], 0.0))

    # analytic_value.append(0.5 * (epsilon * sigma_0 / (k * vix_t0 * vix_t0)) * beta_vix)
    option_vix_price.append(price)
    implied_vol_vix.append(implied_volatility(price, vix_t0, strikes[i], T, 0.0, 0.0, 'c'))

plt.plot(strikes, implied_vol_vix, linestyle='--', label='ATM IV VIX', color='black', marker='.')
plt.title('theta = %s' % theta)
plt.xlabel('K')
plt.legend()
plt.show()
