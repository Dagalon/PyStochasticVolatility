import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

dt = np.linspace(0.01, 0.1, 30)
no_dt_s = len(dt)

# simulation info
alpha = 0.3
nu = 0.6
rho = -0.6
parameters = [alpha, nu, rho]
no_time_steps = 100

seed = 123456789
no_paths = 750000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
options = []
options_shift_right = []
options_shift_left = []
shift_spot = 0.0001
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_right.append(
        EuropeanOption(f0 * (1.0 + shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_left.append(
        EuropeanOption(f0 * (1.0 - shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
implied_vol_atm = []

skew_atm_mc = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = SABR_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_option_price_shift_right = options_shift_right[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_option_price_shift_left = options_shift_left[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    implied_vol_base = implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c')
    implied_vol_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                 0.0, 0.0, 'c')
    implied_vol_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i], 0.0,
                                                0.0, 'c')
    skew_atm_mc.append(f0 * (implied_vol_shift_right - implied_vol_shift_left) / (2.0 * shift_spot * f0))
    # skew_atm_mc.append(f0 * (implied_vol_shift_right - implied_vol_base) / (shift_spot * f0))

asymptotic_limit = 0.5 * rho * nu

plt.plot(dt, skew_atm_mc, label='skew atm SABR', color='black', linestyle='--')
plt.plot(dt, np.ones(len(dt)) * asymptotic_limit, label='asymptotic limit',
         color='black', marker='.', linestyle='--')

plt.xlabel('T')
plt.legend()
plt.show()
