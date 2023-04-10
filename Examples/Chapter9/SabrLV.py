import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_LocalVol import LocalVolEngine, LocalVolFunctionals
from Tools import RNG, Types
from Instruments.ForwardStartEuropeanInstrument import ForwardStartEuropeanOption
from Instruments.EuropeanInstruments import EuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from functools import partial


# simulation info
sigma = 0.7
beta = 0.3
local_vol_mc = partial(LocalVolFunctionals.local_vol_normal_sabr, beta=beta - 1.0, sigma=sigma)

no_time_steps = 100

seed = 123456789
no_paths = 1000000
d_t_forward = 0.9
T = 1.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 10.0
options = []
normal_options = []
no_strikes = 30
strikes = np.linspace(0.9, 1.1, no_strikes)


for k_i in strikes:
    normal_options.append(EuropeanOption(k_i * f0, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL, f0, T))
    options.append(ForwardStartEuropeanOption(k_i, 1.0, Types.TypeSellBuy.BUY, Types.TypeEuropeanOption.CALL,
                                              f0, d_t_forward, T))

# outputs
implied_vol_forward = []

# implied_vol_hagan = []
implied_vol_spot = []

rnd_generator.set_seed(seed)
map_output = LocalVolEngine.get_path_multi_step(0.0, T, f0, no_paths, no_time_steps,
                                                Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, local_vol_mc,
                                                rnd_generator, extra_sampling_points=[d_t_forward])

for i in range(0, no_strikes):
    index_normal_option = np.searchsorted(np.array(map_output[Types.LOCAL_VOL_OUTPUT.TIMES]), d_t_forward)
    mc_normal_options_price = normal_options[i].get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS][:, -1])

    options[i].update_forward_start_date_index(np.array(map_output[Types.LOCAL_VOL_OUTPUT.TIMES]))
    mc_option_price = options[i].get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS])

    # implied_vol_hagan.append(expansion_hagan.get_implied_vol(T, f0, f0 * strikes[i]))

    implied_vol_forward.append(implied_volatility(mc_option_price[0] / f0, 1.0, strikes[i], T - d_t_forward, 0.0, 0.0, 'c'))
    implied_vol_spot.append(implied_volatility(mc_normal_options_price[0] / f0, 1.0, strikes[i], T, 0.0, 0.0, 'c'))

plt.plot(strikes, implied_vol_forward, label='forward smile CEV', color='green', linestyle='--', marker='x')
# plt.plot(np.log(strikes), implied_vol_hagan, label='spot smile CEV Hagan', color='black', linestyle='--')
plt.plot(strikes, implied_vol_spot, label='spot smile CEV', color='blue', linestyle='--', marker='.')

plt.xlabel('K')
plt.legend()
plt.show()
