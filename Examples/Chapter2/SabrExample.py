from MC_Engines.MC_SABR import SABR_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from VolatilitySurface.Tools import SABRTools
from py_vollib.black_scholes_merton import black_scholes_merton

import numpy as np

# parameters
nu = 0.7
alpha = 0.3
rho = - 0.85
parameters = [alpha, nu, rho]

f0 = 100
seed = 123456789
no_paths = 100000
T = 2.0

delta = 1.0 / 32.0
no_time_steps = int(T / delta)

k_s = np.arange(60.0, 200.0, 0.1)
notional = 1.0

rnd_generator = RNG.RndGenerator(seed)

european_options = []

for k_i in k_s:
    european_options.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))


map_output = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                             rnd_generator)

option_prices_mc = []
option_prices_hagan = []
for option in european_options:
    result = option.get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    option_prices_mc.append(result[0])
    z = np.log(f0 / k_s)
    iv_hagan = SABRTools.sabr_vol_jit(alpha, rho, nu, z, T)
    option_prices_hagan.append(black_scholes_merton('c', f0, k_s, T, 0.0, iv_hagan, 0.0))