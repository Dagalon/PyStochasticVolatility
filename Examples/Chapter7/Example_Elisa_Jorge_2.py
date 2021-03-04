import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import MixedRBergomi_Engine, RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

T = 5.0
dt = np.linspace(0.05, T, 50)
no_dt_s = len(dt)

# simulation info
h_short = 0.1
h_long = 0.9
nu_short = 0.5
nu_long = 0.5
rho = -0.6
v0 = 0.25
sigma_0 = np.sqrt(v0)

parameters = [nu_short, nu_long, rho, h_short, h_long]
parameters_rbergomi = [nu_short, rho, h_short]

seed = 123
no_paths = 500000
no_time_steps = 200

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
skew_atm_mixed_rbergomi_mc = []
skew_atm_rbergomi_mc = []

for i in range(0, no_dt_s):
    # Mixed Rbergomi
    rnd_generator.set_seed(seed)
    map_output = MixedRBergomi_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, sigma_0, no_paths,
                                                          no_time_steps,
                                                          Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                          rnd_generator)

    mc_option_price = options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_left = options_shift_left[i].get_price_control_variate(
        map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
        map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_right = options_shift_right[i].get_price_control_variate(
        map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
        map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_atm = implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c')
    implied_vol_atm_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                     0.0, 0.0, 'c')
    implied_vol_atm_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i],
                                                    0.0, 0.0, 'c')

    skew_atm_mixed_rbergomi_mc.append(
        (implied_vol_atm_shift_right - implied_vol_atm_shift_left) / (2.0 * shift_spot * f0))

    # rbergomi
    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine.get_path_multi_step(0.0, dt[i], parameters_rbergomi, f0, sigma_0, no_paths,
                                                     no_time_steps,
                                                     Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                                     rnd_generator)

    mc_option_price = options[i].get_price_control_variate(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_left = options_shift_left[i].get_price_control_variate(
        map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
        map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    mc_option_price_shift_right = options_shift_right[i].get_price_control_variate(
        map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1],
        map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    implied_vol_atm = implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c')
    implied_vol_atm_shift_right = implied_volatility(mc_option_price_shift_right[0], f0, f0 * (1.0 + shift_spot), dt[i],
                                                     0.0, 0.0, 'c')
    implied_vol_atm_shift_left = implied_volatility(mc_option_price_shift_left[0], f0, f0 * (1.0 - shift_spot), dt[i],
                                                    0.0, 0.0, 'c')

    skew_atm_rbergomi_mc.append(
        (implied_vol_atm_shift_right - implied_vol_atm_shift_left) / (2.0 * shift_spot * f0))


plt.plot(dt, skew_atm_mixed_rbergomi_mc, label='skew atm mixed rBergomi H_short=%s and H_long=%s' % (h_short, h_long),
         color='black', linestyle='--')

plt.plot(dt, skew_atm_rbergomi_mc, label='skew atm rBergomi H=%s' % h_short, color='green', linestyle='--')


plt.ylim([-0.005, 0.0])
plt.xlim([0.05, T])
plt.xlabel('T')
plt.legend()
plt.show()
