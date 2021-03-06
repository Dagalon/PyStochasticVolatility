import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit

dt = np.linspace(0.00001, 0.001, 15)
no_dt_s = len(dt)

# simulation info
alpha = 0.4
nu = 0.5
rho = - 0.4
parameters = [alpha, nu, rho]
no_time_steps = 3

seed = 123456789
no_paths = 10000000
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

smile_atm_mc = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = SABR_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY, rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_option_price_cv = options[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                              map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])
    mc_option_price_shift_right = options_shift_right[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_option_price_cv_right = options_shift_right[i].get_price_control_variate(
        map_output[Types.SABR_OUTPUT.PATHS][:, -1],
        map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_left = options_shift_left[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_option_price_cv_left = options_shift_left[i].get_price_control_variate(
        map_output[Types.SABR_OUTPUT.PATHS][:, -1],
        map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_base = implied_volatility(mc_option_price_cv[0], f0, f0, dt[i], 0.0, 0.0, 'c')
    implied_vol_shift_right = implied_volatility(mc_option_price_cv_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                 0.0, 0.0, 'c')
    implied_vol_shift_left = implied_volatility(mc_option_price_cv_left[0], f0, f0 * (1.0 - shift_spot), dt[i], 0.0,
                                                0.0, 'c')

    smile_atm_mc.append((implied_vol_shift_right - 2.0 * implied_vol_base +
                        implied_vol_shift_left) / (f0 * f0 * shift_spot * shift_spot))


def f_law(x, a, b, c):
    return a + b * np.power(x, -c)


popt, pcov = curve_fit(f_law, dt, smile_atm_mc)
y_fit_values = f_law(dt, *popt)


asymptotic_limit = (1.0 / 3.0 - 0.5 * rho * rho) * 0.25 * nu * nu / alpha

plt.plot(dt, smile_atm_mc, label='curvature atm SABR', color='black', linestyle='--')
plt.plot(dt, y_fit_values, label='%s + %s t^%s' % (round(popt[0], 5), round(popt[1], 5),
                                                   round(popt[2], 5)), color='black',
         marker='.', linestyle='--')

plt.xlabel('T')
plt.legend()
plt.show()
