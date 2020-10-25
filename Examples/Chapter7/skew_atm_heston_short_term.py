import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools


dt = np.arange(1, 30, 1) * 1.0 / 365.0
no_dt_s = len(dt)

# simulation info
epsilon = 0.3
k = 1.0
rho = -0.9
v0 = 0.05
sigma_0 = np.sqrt(0.05)
theta = 0.06

parameters = [k, theta, epsilon, rho, v0]

seed = 123456789
no_paths = 500000
delta_time = 1.0 / 365.0
no_time_steps = 10


# option information
f0 = 120.0
options = []
options_shift_right = []
options_shift_left = []
shift_spot = 0.0001

# random number generator
rnd_generator = RNG.RndGenerator(seed)

for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_right.append(EuropeanOption(f0 * (1.0 + shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_left.append(EuropeanOption(f0 * (1.0 - shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
implied_vol_atm = []
implied_vol_atm_shift_right = []
implied_vol_atm_shift_left = []
skew_atm = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = Heston_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                   no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                   rnd_generator)

    option_price_mc = options[i].get_price(map_output[Types.HESTON_OUTPUT.PATHS][:, -1])
    option_price_right_mc = options_shift_right[i].get_price(map_output[Types.HESTON_OUTPUT.PATHS][:, -1])
    option_price_left_mc = options_shift_left[i].get_price(map_output[Types.HESTON_OUTPUT.PATHS][:, -1])

    implied_vol_atm.append(implied_volatility(option_price_mc[0], f0, f0, dt[i], 0.0, 0.0, 'c'))
    implied_vol_atm_shift_right.append(implied_volatility(option_price_right_mc[0], f0, f0 * (1.0 + shift_spot), dt[i], 0.0, 0.0, 'c'))
    implied_vol_atm_shift_left.append(implied_volatility(option_price_left_mc[0], f0, f0 * (1.0 - shift_spot), dt[i], 0.0, 0.0, 'c'))
    skew_atm.append(f0 * (implied_vol_atm_shift_right[i] - implied_vol_atm_shift_left[i]) / (2.0 * shift_spot * f0))


asymptotic_limit = 0.25 * rho * epsilon / sigma_0

plt.plot(dt, skew_atm, label='skew atm heston')
plt.plot(dt, np.ones(len(dt)) * asymptotic_limit, label='asymptotic limit')


plt.xlabel('T')
plt.legend()
plt.show()

