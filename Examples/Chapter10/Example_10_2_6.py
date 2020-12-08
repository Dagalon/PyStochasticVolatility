import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_MixedLogNormal import MixedLogNormalEngine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton import black_scholes_merton

# simulation info
nu_1 = 0.8
nu_2 = 0.7
theta = 0.7
rho = -0.6

parameters = [nu_1, nu_2, theta, rho]

f0 = 100
v0 = 0.25

seed = 123
no_paths = 1000000
no_time_steps = 100

delta_vix = 1.0 / 12.0
T = 0.1

# random number generator
rnd_generator = RNG.RndGenerator(seed)
map_output = MixedLogNormalEngine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths, no_time_steps,
                                                      Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

# VIX spot
index_t_i = np.searchsorted(map_output[Types.MIXEDLOGNORMAL_OUTPUT.TIMES], T)
vix_t0 = np.mean(np.sqrt(map_output[Types.MIXEDLOGNORMAL_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i]))

no_strikes = 50
strikes = np.linspace(0.90, 1.10, no_strikes) * vix_t0

# Check model is good
mean_var = np.mean(map_output[Types.MIXEDLOGNORMAL_OUTPUT.SPOT_VARIANCE_PATHS], axis=0)
mean_asset = np.mean(map_output[Types.MIXEDLOGNORMAL_OUTPUT.PATHS], axis=0)

option_vix_price = []
implied_vol_vix = []
analytic_value = []

for i in range(0, no_strikes):
    price = np.mean(np.maximum(np.sqrt(map_output[Types.MIXEDLOGNORMAL_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i])
                               - strikes[i], 0.0))

    std_error = np.std(np.maximum(np.sqrt(map_output[Types.MIXEDLOGNORMAL_OUTPUT.SPOT_VARIANCE_PATHS][:, index_t_i])
                                  - strikes[i], 0.0)) / np.sqrt(no_paths)

    # price_bs_theta_0 = black_scholes_merton('c', np.exp(- nu_1 * nu_1 * T / 8.0) * vix_t0, strikes[i], T, 0.0, 0.5 * nu_1, 0.0)
    # price_bs_theta_1 = black_scholes_merton('c', np.exp(- nu_2 * nu_2 * T / 8.0) * vix_t0, strikes[i], T, 0.0, 0.5 * nu_2, 0.0)

    implied_vol_vix.append(implied_volatility(price, vix_t0, strikes[i], T, 0.0, 0.0, 'c'))

plt.ylim([0.364, 0.366])
plt.plot(strikes, implied_vol_vix, linestyle='--', label='Implied Vol VIX', color='black', marker='.')
plt.title('delta = %s' % theta)
plt.xlabel('K')
plt.legend()
plt.show()
