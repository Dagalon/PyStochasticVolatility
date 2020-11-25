import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.optimize import curve_fit

dt = np.linspace(0.00001, 0.001, 15)
no_dt_s = len(dt)

# simulation info
alpha = 0.4
nu = 1.1
rho = - 0.999
parameters = [alpha, nu, rho]
no_time_steps = 3

seed = 123456789
no_paths = 10000000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
# options
strikes = np.linspace(70.0, 130.0, 30)
no_strikes = len(strikes)
f0 = 100
T = 0.1
notional = 1.0
options = []
for k_i in strikes:
    options.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))


map_output = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY, rnd_generator)

iv_vol = []
for i in range(0, no_strikes):
    rnd_generator.set_seed(seed)
    mc_option_price = options[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])
    
    iv_vol.append(implied_volatility(mc_option_price[0], f0, strikes[i], T, 0.0, 0.0, 'c'))


plt.plot(strikes, iv_vol, label="rho=%s" % rho, marker=".", linestyle="--", color="black")

plt.xlabel("K")
plt.legend()
plt.show()
