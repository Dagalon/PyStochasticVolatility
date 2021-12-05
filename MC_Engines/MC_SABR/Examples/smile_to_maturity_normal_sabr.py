import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Normal_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools.Bachelier import implied_volatility, bachelier

# simulation info
alpha = 0.07
nu = 0.2
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
t = 1.0
options = []

strikes = np.linspace(-0.05, 0.05, 100)
no_strikes = len(strikes)

for k_i in strikes:
    options.append(EuropeanOption(k_i, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# outputs
implied_vol = []

# paths
rnd_generator.set_seed(seed)
map_output = SABR_Normal_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
                                                    Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

for i in range(0, no_strikes):
    npv = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    implied_vol.append(implied_volatility(npv[0], f0, strikes[i], t, 'c'))
    bachelier_npv = bachelier(f0, strikes[i], t, implied_vol[-1], 'c')


plt.title("T=%s and nu=%s" % (t, nu))
plt.plot(strikes, implied_vol, label='iv', linestyle='--', color='black', marker='.')


plt.xlabel('K')
plt.legend()

plt.show()
