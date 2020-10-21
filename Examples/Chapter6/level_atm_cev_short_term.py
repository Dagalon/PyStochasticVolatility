import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_LocalVolEngine import LocalVolEngine, LocalVolFunctionals
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools
from functools import partial

dt = np.linspace(0.01, 0.1, 10)
no_dt_s = len(dt)

# simulation info
sigma = 0.3
beta = 0.4
local_vol_mc = partial(LocalVolFunctionals.log_cev_diffusion, beta=beta - 1.0, sigma=sigma)

parameters = [sigma, beta]

seed = 123456789
no_paths = 500000
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
    # expansion info
    lv_f0 = LocalVolFunctionals.cev_diffusion(dt[i], np.asfortranarray(f0), beta - 1.0, sigma)[0]
    fd_lv_f0 = LocalVolFunctionals.first_derive_cev_diffusion(dt[i], np.asfortranarray(f0), beta - 1.0, sigma)[0]
    sd_lv_f0 = LocalVolFunctionals.second_derive_cev_diffusion(dt[i], np.asfortranarray(f0), beta - 1.0, sigma)[0]

    rnd_generator.set_seed(seed)
    map_output = LocalVolEngine .get_path_multi_step(0.0, dt[i], f0, no_paths, no_time_steps,
                                                     Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                     local_vol_mc,
                                                     rnd_generator)
    # implied vol by MC and asymptotic expansion
    mc_option_price = options[i].get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS][:, -1])
    implied_vol_atm.append(implied_volatility(mc_option_price[0], f0, f0, dt[i], 0.0, 0.0, 'c'))
    implied_vol_approx.append(ExpansionTools.get_iv_atm_local_vol_approximation(f0, lv_f0, fd_lv_f0, sd_lv_f0, dt[i]))

    # vol swap approximation
    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS], axis=1) / dt[i])))
    std_mc_vol_swap = np.std(
        np.sqrt(np.sum(map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])) / np.sqrt(no_paths)
    vol_swap_approximation.append(ExpansionTools.get_vol_swap_local_vol(0.0, dt[i], f0, lv_f0, fd_lv_f0, sd_lv_f0))

    output.append((implied_vol_atm[i] - vol_swap_mc[i]))

# curve fit


def f_law(x, a, b):
    return a + b * x


popt, pcov = curve_fit(f_law, dt, output)
y_fit_values = f_law(dt, *popt)

plt.plot(dt, output, label='(I(t,f0) - E(v_t))', linestyle='--', color='black')
# plt.plot(dt, vol_swap_mc, label='E(v_t)', linestyle='--', marker='.')
# plt.plot(dt, implied_vol_atm, label='implied volatility atm', linestyle='--', marker='x')
plt.plot(dt, y_fit_values, label="%s + %s * t" % (round(popt[0], 10), round(popt[1], 10)),
          marker='.', linestyle='--', color='black')

# plt.plot(dt, implied_vol_approx, label='approximation iv', linestyle='--')
# plt.plot(dt, implied_vol_atm, label='mc iv', linestyle='--')

plt.xlabel('T')
plt.legend()
plt.show()