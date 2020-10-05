import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools

dt = np.linspace(0.01, 0.1, 10)
no_dt_s = len(dt)

# simulation info
h = 0.3
nu = 0.7
rho = -0.6
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

seed = 123456789
no_paths = 1000000
# delta_time = 1.0 / 365.0
no_time_steps = 100

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
options = []
implied_vol_atm = []
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
vol_swap_approximation = []
vol_swap_mc = []
implied_vol_atm = []
implied_vol_approx = []
output = []

for i in range(0, no_dt_s):
    # no_time_steps = int(dt[i] / delta_time)
    rnd_generator.set_seed(seed)
    map_output = RBergomi_Engine .get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                      no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                      rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1])

    implied_vol_atm.append(implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c'))
    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    error_mc_vol_swap = np.std(np.sqrt(np.sum(map_output[Types.RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])) / np.sqrt(no_paths)
    implied_vol_approx.append(ExpansionTools.get_iv_atm_rbergomi_approximation(parameters, vol_swap_mc[i], sigma_0, dt[i]))
    output.append((implied_vol_atm[i] - vol_swap_mc[i]))

# curve fit


def f_law(x, b, c):
    return b * np.power(x, 2.0 * c)


popt, pcov = curve_fit(f_law, dt, output)
y_fit_values = f_law(dt, *popt)
#
plt.plot(dt, output, label='(I(t,f0) - E(v_t))', linestyle='--')
# plt.plot(dt, vol_swap_mc, label='E(v_t)', linestyle='--', marker='.')
# plt.plot(dt, implied_vol_atm, label='implied volatility atm', linestyle='--', marker='x')
plt.plot(dt, y_fit_values, label="%s * t^(2 * %s)" % (round(popt[0], 5), round(popt[1], 5)),
         marker='.', linestyle='--')

# plt.plot(dt, implied_vol_approx, label='approximation iv', linestyle='--')
# plt.plot(dt, implied_vol_atm, label='mc iv', linestyle='--')

plt.xlabel('t')
plt.legend()
plt.show()