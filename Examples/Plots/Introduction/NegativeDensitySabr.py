from MC_Engines.MC_SABR import SABR_Normal_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from VolatilitySurface.Tools import SABRTools
from Tools.Bachelier import bachelier, implied_volatility

import numpy as np
import matplotlib.pylab as plt

# sabr parameters
alpha = 0.01
nu = 1.5
rho = 0.3
parameters = [alpha, nu, rho]

# forward and strikes
f0 = 0.0
t = 10.0

no_points = 1601
delta_strike = 1.0 / 10000.0
spreads = np.linspace(-800.0, 800.0, no_points)

strikes = []
options = []
for si in spreads:
    strikes.append(si / 10000.0 + f0)
    options.append(EuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# mc price
seed = 123456789
no_paths = 300000
rnd_generator = RNG.RndGenerator(seed)
no_time_steps = int(50 * t)


map_output = SABR_Normal_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                             rnd_generator)

option_prices_mc = []
option_prices_hagan = []
no_options = len(options)

for i in range(0, no_options):
    result = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    option_prices_mc.append(result[0])
    z = strikes[i] - f0
    iv_hagan = SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t)
    price = bachelier(f0, strikes[i], t, iv_hagan, 'c')

    option_prices_hagan.append(price)

pdf_hagan = []
pdf_mc = []
plt.figure(figsize=(8, 5))
for i in range(1, no_options - 1):
    pdf_hagan.append((option_prices_hagan[i+1] - 2.0 * option_prices_hagan[i] + option_prices_hagan[i-1]) / (delta_strike * delta_strike))
    pdf_mc.append((option_prices_mc[i+1] - 2.0 * option_prices_mc[i] + option_prices_mc[i-1]) / (delta_strike * delta_strike))

grid = range(-5, 270, 10)
plt.yticks(grid)

plt.plot(strikes[1:no_options-1], pdf_hagan, label="Hagan's density", linestyle='--', color='blue',  linewidth=0.5)
plt.plot(strikes[1:no_options-1], pdf_mc, label="MC's density", linestyle='--', markersize=3, color='green',  linewidth=0.5)

plt.xlabel("strike")
plt.title("Negative values for Hagan's density")
plt.legend()
plt.show()