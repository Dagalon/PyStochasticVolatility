import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.integrate import quad_vec
from AnalyticEngines.MalliavinMethod import ExpansionTools

# simulation info
h = 0.3
nu = 0.5
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

no_time_steps = 10
f0 = 100

seed = 123456789
no_paths = 1000000

delta_vix = 1.0 / 12.0
T = 0.1

# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []

value = quad_vec(lambda x: np.exp(nu * nu * np.power(x, 2.0 * h)), 0.0, 1.0 / 12.0)
beta_vix_0 = np.sqrt(value[0] / delta_vix)
vix_t0 = sigma_0 * beta_vix_0

vix_t_0_aux = ExpansionTools.get_vix_rbergomi_t(0.000001, delta_vix + 0.000001, delta_vix, nu, h,
                                                np.asfortranarray(v0),
                                                v0, 200)

no_strikes = 30
strikes = np.linspace(0.95, 1.05, no_strikes) * vix_t0

map_output = RBergomi_Engine.get_path_multi_step(0.0, T, parameters, f0, sigma_0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

for i in range(0, no_strikes):
    t_i_vix = T + delta_vix

    index_t_i = np.searchsorted(map_output[Types.RBERGOMI_OUTPUT.TIMES], T)
    vix_t = ExpansionTools.get_vix_rbergomi_t(T, t_i_vix, delta_vix, nu, h,
                                              map_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, index_t_i],
                                              v0, 200)
    price = np.mean(np.maximum(vix_t -
                               strikes[i], 0.0))
    option_vix_price.append(price)

    implied_vol_vix.append(implied_volatility(option_vix_price[-1], vix_t0, strikes[i], T, 0.0, 0.0, 'c'))


analytic_value = nu * np.sqrt(2.0 * h) * v0 * np.power(delta_vix, h - 0.5) / ((h + 0.5) * vix_t0 * vix_t0)
plt.plot(strikes, implied_vol_vix, linestyle='--', label='Implied Vol VIX', color='black', marker='.')

plt.xlabel('K')
plt.legend()
plt.show()
