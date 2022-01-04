import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Normal_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools.Bachelier import implied_volatility
from scipy.optimize import curve_fit

dt = np.arange(7, 180, 2) * (1.0 / 365.0)
no_dt_s = len(dt)

# simulation info
alpha = 0.5
nu = 0.5
rho = -0.4
parameters = [alpha, nu, rho]
no_time_steps = 100

seed = 123456789
no_paths = 750000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 0.005
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
skew_atm_mc = []
vol_swap_mc = []
vol_swap_approx = []
diff_vol_swap = []
diff_vol_swap_approx = []
diff_log_error = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = SABR_Normal_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
                                                        Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])

    mc_option_price_shift_right = options_shift_right[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])

    mc_option_price_shift_left = options_shift_left[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])

    implied_vol_atm = implied_volatility(mc_option_price[0], f0, f0, dt[i], 'c')

    implied_vol_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                 'c')

    implied_vol_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i], 'c')

    skew_atm_mc.append((implied_vol_shift_right - implied_vol_shift_left) / (2.0 * shift_spot * f0))


def f_law(x, a, b):
    return a * np.power(x, b)


popt, pcov = curve_fit(f_law, dt, skew_atm_mc)
y_fit_values = f_law(dt, *popt)

plt.plot(dt, skew_atm_mc, label='dIV(F0,T)/dF0', color='black', linestyle='--')
plt.plot(dt, y_fit_values, label="%s * t ^ %s" % (round(popt[0], 5), round(popt[1], 5)), marker='.',
         linestyle='--', color='black')

plt.xlabel('T')
plt.legend()
plt.show()
