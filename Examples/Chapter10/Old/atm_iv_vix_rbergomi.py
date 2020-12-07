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
# T_VIX = [0.01, 0.025, 0.035, 0.045, 0.05, 0.065, 0.075, 0.1]
T_VIX = np.linspace(0.01, 0.1, 20)

# random number generator
rnd_generator = RNG.RndGenerator(seed)

option_vix_price = []
implied_vol_vix = []

value = quad_vec(lambda x: np.exp(nu * nu * np.power(x, 2.0 * h)), 0.0, 1.0 / 12.0)
beta_vix_0 = np.sqrt(value[0] / delta_vix)
vix_t0 = sigma_0 * beta_vix_0


for i in range(0, len(T_VIX)):
    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, T_VIX[i], parameters, f0, sigma_0, no_paths, no_time_steps,
                                                     Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator, )
    t_i_vix = T_VIX[i] + delta_vix

    index_t_i = np.searchsorted(map_output[Types.RBERGOMI_OUTPUT.TIMES], T_VIX[i])

    vix_t = ExpansionTools.get_vix_rbergomi_t(T_VIX[i], t_i_vix, delta_vix, nu, h,
                                              map_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, index_t_i],
                                              v0, 200)
    price = np.mean(np.maximum(vix_t -
                               vix_t0, 0.0))
    option_vix_price.append(price)

    implied_vol_vix.append(implied_volatility(option_vix_price[-1], vix_t0, vix_t0, T_VIX[i], 0.0, 0.0, 'c'))


analytic_value = nu * np.sqrt(2.0 * h) * v0 * np.power(delta_vix, h - 0.5) / ((h + 0.5) * vix_t0 * vix_t0)
plt.plot(T_VIX, implied_vol_vix, linestyle='--', label='ATM IV VIX', color='black', marker='.')

plt.xlabel('T')
plt.legend()
plt.show()
