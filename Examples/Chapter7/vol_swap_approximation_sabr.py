import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

dt = np.arange(1, 180, 2) * (1.0 / 365.0)
no_dt_s = len(dt)

# simulation info
alpha = 0.5
nu = 0.5
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
skew_atm_mc = []
vol_swap_mc = []
vol_swap_approx = []
diff_vol_swap = []
diff_vol_swap_approx = []
diff_log_error = []

for i in range(0, no_dt_s):
    rnd_generator.set_seed(seed)
    map_output = SABR_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
                                                 Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    error_mc_vol_swap = np.std(
        np.sqrt(np.sum(map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])) / np.sqrt(no_paths)

    mc_option_price = options[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_right = options_shift_right[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                                                   map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_left = options_shift_left[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                                                 map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_atm = implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c')

    implied_vol_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                 0.0, 0.0, 'c')

    implied_vol_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i], 0.0,
                                                0.0, 'c')

    skew_atm_mc.append((implied_vol_shift_right - implied_vol_shift_left) / (2.0 * shift_spot * f0))
    vol_swap_approx.append(implied_vol_atm - 0.5 * implied_vol_atm * implied_vol_atm * skew_atm_mc[i] * dt[i])

    diff_vol_swap.append(vol_swap_mc[i] - implied_vol_atm)
    diff_vol_swap_approx.append(vol_swap_mc[i] - vol_swap_approx[i])
    diff_log_error.append(np.log(diff_vol_swap[i] / diff_vol_swap_approx[i]))


plt.plot(dt, diff_log_error, label='ln((E(v_t) - iv) / (E(v_t) - vol swap approx)) ', color='black', linestyle='--')
# plt.plot(dt, diff_vol_swap_approx, label='E(v_t) - vol swap approx', color='black', linestyle='--', marker='.')

plt.xlabel('T')
plt.legend()
plt.show()
