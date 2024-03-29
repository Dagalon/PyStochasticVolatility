import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit

dt = np.linspace(0.01, 0.1, 20)
no_dt_s = len(dt)

# simulation info
h = 0.3
nu = 0.5
rho = -0.5
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

seed = 123456789
no_paths = 500000

delta_time = 1.0 / 365.0
no_time_steps = 100

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
shift_spot = 0.0001
options = []
options_shift_right = []
options_shift_left = []
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_left.append(
        EuropeanOption(f0 * (1.0 - shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_right.append(
        EuropeanOption(f0 * (1.0 + shift_spot), 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
skew_atm_mc = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                     no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                     rnd_generator)

    mc_option_price = options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_left = options_shift_left[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                                                 map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_right = options_shift_right[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                                                   map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_atm = implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c')
    implied_vol_atm_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot),  dt[i], 0.0, 0.0, 'c')
    implied_vol_atm_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i], 0.0, 0.0, 'c')

    skew_atm_mc.append((implied_vol_atm_shift_right - implied_vol_atm_shift_left) / (2.0 * shift_spot * f0))


def f_law(x, a, b):
    return a * np.power(x, b)


popt, pcov = curve_fit(f_law, dt, skew_atm_mc)
dt_fit = np.linspace(0.0000001, 0.1)
y_fit_values = f_law(dt_fit, *popt)

plt.plot(dt, skew_atm_mc, label='skew atm rBergomi', color='black', linestyle='--')
plt.plot(dt_fit, y_fit_values, label='%s T^(%s)' % (round(popt[0], 5), round(popt[1], 5)), color='black',
          linestyle='--', marker='.')

plt.ylim([-0.0095, -0.0050])
plt.xlim([0.001, 0.1])
plt.xlabel('T')
plt.legend()
plt.show()
