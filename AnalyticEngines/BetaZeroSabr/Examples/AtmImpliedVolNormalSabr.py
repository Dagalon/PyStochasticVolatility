import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Normal_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools.Bachelier import implied_volatility, bachelier
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod.ExpansionTools import get_vol_swap_approximation_sabr
from VolatilitySurface.Tools import SABRTools


dt = np.arange(180, 360, 5) * 1.0 / 365.0
no_dt_s = len(dt)

# simulation info
alpha = 0.3
nu = 0.4
rho = 0.0
parameters = [alpha, nu, rho]
no_time_steps = 100

seed = 123456789
no_paths = 500000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 0.01
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
    rnd_generator.set_seed(seed)
    map_output = SABR_Normal_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
                                                        Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    implied_vol_atm.append(implied_volatility(mc_option_price[0], f0, f0, dt[i], 'c'))
    iv_hagan = SABRTools.sabr_normal_jit(f0, f0, alpha, rho, nu, dt[i])
    option_bachelier_price = bachelier(f0, f0, dt[i], implied_vol_atm[-1], 'c')
    vol_swap_approximation.append(get_vol_swap_approximation_sabr(parameters, 0.0, dt[i], alpha))
    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    output.append((implied_vol_atm[-1] - vol_swap_mc[-1]))

# curve fit


def f_law(x, a, b, c):
    return a + b * np.power(x, c)


popt, pcov = curve_fit(f_law, dt, output)
y_fit_values = f_law(dt, *popt)

# plt.plot(dt, output, label='(I(t,f0) - E(v_t))', linestyle='--', color='black')
plt.plot(dt, output, label='I(t,f0) - E(v_t)', linestyle='--')
plt.plot(dt, y_fit_values, label="%s + %s * t^%s" % (round(popt[0], 5), round(popt[1], 5), round(popt[2], 5)), marker='.',
         linestyle='--')

plt.title("rho=%s" % rho)
plt.xlabel('T')
plt.legend()

plt.show()