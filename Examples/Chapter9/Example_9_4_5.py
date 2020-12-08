import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.ForwardStartEuropeanInstrument import ForwardStartEuropeanOption
from Instruments.EuropeanInstruments import EuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from MC_Engines.MC_RBergomi import ToolsVariance


# function to compute VIX_t from MC simulation
def get_vix_rbergomi_t(t0, t1, delta_vix, nu, h, v_t, v0, no_integration_points):
    no_elements = len(v_t)
    vix_2_t = np.zeros(no_elements)
    t_i = np.linspace(t0, t1, no_integration_points)
    rho_s_1 = np.zeros(no_integration_points)
    rho_s_2 = np.zeros(no_integration_points)

    for k in range(0, no_integration_points):
        rho_s_1[k] = np.exp(2.0 * nu * ToolsVariance.get_volterra_covariance(t0, t_i[k], h) / np.power(t0, 2.0 * h))
        rho_s_2[k] = np.exp(- nu * ToolsVariance.get_volterra_covariance(t0, t_i[k], h) / np.power(t0, 2.0 * h))

    for k in range(0, no_elements):
        for j in range(1, no_integration_points):
            w_i_h = (np.log(v_t[k] / v_t[0]) + nu * nu * np.power(T, 2.0 * h)) / (2.0 * nu)
            w_i_1 = rho_s_1[j - 1] * rho_s_2[j - 1] * np.exp(w_i_h + 0.5 * nu * nu * np.power(t_i[j-1], 2.0 * h))
            w_i = rho_s_1[j] * rho_s_2[j] * np.exp(w_i_h + 0.5 * nu * nu * np.power(t_i[j], 2.0 * h))
            vix_2_t[k] += 0.5 * (w_i_1 + w_i) * (v0 / delta_vix)

    return np.sqrt(vix_2_t)

# simulation info
h = 0.3
nu = 0.5
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

no_time_steps = 100

seed = 123456789
no_paths = 1000000
d_t_forward = 0.9
T = 1.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
options = []
normal_options = []
no_strikes = 30
strikes = np.linspace(0.7, 1.3, no_strikes)

for k_i in strikes:
    normal_options.append(EuropeanOption(k_i * f0, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL, f0, T))
    options.append(ForwardStartEuropeanOption(k_i, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL,
                                              f0, d_t_forward, T))


# outputs
implied_vol_forward = []
implied_vol_spot = []

rnd_generator.set_seed(seed)
map_output = RBergomi_Engine.get_path_multi_step(0.0, T, parameters, f0, sigma_0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator,
                                                  extra_sampling_points=[d_t_forward])

for i in range(0, no_strikes):
    index_normal_option = np.searchsorted(np.array(map_output[Types.RBERGOMI_OUTPUT.TIMES]), d_t_forward)
    mc_normal_options_price = normal_options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                                          map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    options[i].update_forward_start_date_index(np.array(map_output[Types.RBERGOMI_OUTPUT.TIMES]))
    mc_option_price = options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS],
                                                           map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_forward.append(implied_volatility(mc_option_price[0] / f0, 1.0, strikes[i], T - d_t_forward, 0.0, 0.0, 'c'))
    implied_vol_spot.append(implied_volatility(mc_normal_options_price[0] / f0, 1.0, strikes[i], T, 0.0, 0.0, 'c'))


plt.plot(strikes, implied_vol_forward, label='forward smile rBergomi', color='black', linestyle='--')
plt.plot(strikes, implied_vol_spot, label='spot smile rBegomi', color='black', linestyle='--', marker='.')

plt.xlabel('K')
plt.legend()
plt.show()
