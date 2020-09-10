from MC_Engines.MC_SABR import SABR_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from VolatilitySurface.Tools import SABRTools
from py_vollib.black_scholes_merton import black_scholes_merton

import numpy as np
import matplotlib.pylab as plt

# parameters
nu = 1.5
alpha = 0.4
rho = - 0.6
parameters = [alpha, nu, rho]

f0 = 100
seed = 123456789
no_paths = 1000000
T = 0.25
delta = 1.0 / 100.0
no_time_steps = int(T / delta)

delta_strike = 0.1
k_s = np.arange(30.0, 230.0, delta_strike)
notional = 1.0

rnd_generator = RNG.RndGenerator(seed)

european_options = []

for k_i in k_s:
    european_options.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))

map_output = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY,
                                             rnd_generator)

option_prices_mc = []
option_prices_hagan = []
density_mc = []
density_hagan = []
no_options = len(european_options)
for i in range(0, no_options):
    result = european_options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    option_prices_mc.append(result[0])
    z = np.log(f0 / k_s[i])
    iv_hagan = SABRTools.sabr_vol_jit(alpha, rho, nu, z, T)
    option_prices_hagan.append(black_scholes_merton('c', f0, k_s[i], T, 0.0, iv_hagan, 0.0))

pdf_hagan = []
pdf_mc = []
for i in range(1, no_options - 1):
    pdf_hagan.append((option_prices_hagan[i+1] - 2.0 * option_prices_hagan[i] + option_prices_hagan[i-1]) / (delta_strike * delta_strike))
    pdf_mc.append((option_prices_mc[i+1] - 2.0 * option_prices_mc[i] + option_prices_mc[i-1]) / (delta_strike * delta_strike))

plt.plot(k_s[1:no_options-1], pdf_hagan, label="Hagan's density", linestyle='--', color='black')
plt.plot(k_s[1:no_options-1], pdf_mc, label="MC's density", linestyle='--', marker='.', color='black')

plt.title("Negative values for Hangan's density")
plt.legend()
plt.show()