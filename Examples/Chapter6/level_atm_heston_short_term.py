import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit
from AnalyticEngines.MalliavinMethod import ExpansionTools

dt = np.arange(7, 30, 1) * 1.0 / 365.0
no_dt_s = len(dt)

# simulation info
epsilon = 0.5
k = 0.5
rho = -0.9
v0 = 0.05
sigma_0 = np.sqrt(0.05)
theta = 0.06

parameters = [k, theta, epsilon, rho, v0]

seed = 123456789
no_paths = 400000
delta_time = 1.0 / 365.0

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
    no_time_steps = int(dt[i] / delta_time)
    rnd_generator.set_seed(seed)
    map_output = Heston_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                   no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                   rnd_generator)

    mc_option_price = options[i].get_price(map_output[Types.HESTON_OUTPUT.PATHS][:, -1])
    option_price = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                 model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                 compute_greek=False)

    implied_vol_approx.append(ExpansionTools.get_iv_atm_heston_approximation(np.array(parameters), dt[i]))
    implied_vol_atm.append(implied_volatility(option_price, f0, f0, dt[i], 0.0, 0.0, 'c'))
    vol_swap_mc.append(np.mean(np.sqrt(np.sum(map_output[Types.HESTON_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1) / dt[i])))
    output.append((implied_vol_atm[-1] - vol_swap_mc[-1]))

# curve fit


def f_law(x, a, b):
    return a + b * x


popt, pcov = curve_fit(f_law, dt, output)
y_fit_values = f_law(dt, *popt)

plt.plot(dt, output, label='(I(t,f0) - E(v_t))', linestyle='--')
plt.plot(dt, y_fit_values, label="%s + %s * t" % (round(popt[0], 5), round(popt[1], 5)), marker='.', linestyle='--')

plt.xlabel('t')
plt.legend()
plt.show()