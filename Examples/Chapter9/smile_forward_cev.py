import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_LocalVolEngine import LocalVolEngine, LocalVolFunctionals
from Tools import RNG, Types
from Instruments.ForwardStartEuropeanInstrument import ForwardStartEuropeanOption
from Instruments.EuropeanInstruments import EuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from functools import partial

# simulation info
sigma = 0.3
beta = 0.1
local_vol_mc = partial(LocalVolFunctionals.log_cev_diffusion, beta=beta - 1.0, sigma=sigma)

no_time_steps = 200

seed = 123456789
no_paths = 1000000
d_t_forward = 0.5
T = 1.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 1.0
options = []
normal_options = []
no_strikes = 30
strikes = np.linspace(0.9, 1.10, no_strikes)

for k_i in strikes:
    normal_options.append(EuropeanOption(k_i * f0, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL, f0, T))
    options.append(ForwardStartEuropeanOption(k_i, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL,
                                              f0, d_t_forward, T))

# outputs
implied_vol_forward = []
implied_vol_spot = []

rnd_generator.set_seed(seed)
map_output = LocalVolEngine.get_path_multi_step(0.0, T, f0, no_paths, no_time_steps,
                                                Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, local_vol_mc,
                                                rnd_generator, extra_sampling_points=[d_t_forward])

for i in range(0, no_strikes):
    index_normal_option = np.searchsorted(np.array(map_output[Types.LOCAL_VOL_OUTPUT.TIMES]), T)
    mc_normal_options_price = normal_options[i].get_price_control_variate(map_output[Types.LOCAL_VOL_OUTPUT.PATHS][:, index_normal_option],
                                                                          map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    options[i].update_forward_start_date_index(np.array(map_output[Types.LOCAL_VOL_OUTPUT.TIMES]))
    mc_option_price = options[i].get_price_control_variate(map_output[Types.LOCAL_VOL_OUTPUT.PATHS],
                                                           map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_forward.append(implied_volatility(mc_option_price[0] / f0, 1.0, strikes[i], T - d_t_forward, 0.0, 0.0, 'c'))
    implied_vol_spot.append(implied_volatility(mc_normal_options_price[0] / f0, 1.0, strikes[i], T, 0.0, 0.0, 'c'))

plt.plot(strikes, implied_vol_forward, label='forward smile CEV', color='black', linestyle='--')
plt.plot(strikes, implied_vol_spot, label='spot smile CEV', color='black', linestyle='--', marker='.')

plt.xlabel('K')
plt.legend()
plt.show()
