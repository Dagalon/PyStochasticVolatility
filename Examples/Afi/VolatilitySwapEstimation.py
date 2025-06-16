import numpy as np
import matplotlib.pylab as plt

from MC_Engines.MC_SABR import SABR_Engine
from Tools.Types import  SABR_OUTPUT
from Tools import RNG, Types
from AnalyticEngines.MalliavinMethod.ExpansionTools import get_variance_swap_sabr, get_volatility_swap_sabr_malliavin, get_vol_swap_approximation_sabr


# model parameters
alpha = 0.3
nu = 0.4
rho = -0.6
parameters = [alpha, nu, rho]
f0 = 100.0
maturities = np.linspace(0.1, 1.0, 10)

# simulation info
no_paths = 250000

seed = 12345
rnd_generator = RNG.RndGenerator(seed)

sampling_vol_swap = []
malliavin_vol_swap = []

for ti in maturities:
    no_time_steps = np.floor(104 * ti) + 1
    rnd_generator.set_seed(seed)
    output = SABR_Engine.get_path_multi_step(0.0, ti, parameters, f0, no_paths, int(no_time_steps), Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)
    sampling_vol_swap.append(np.mean(np.sqrt(np.sum(output[SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS],1)) / np.sqrt(ti)))
    var_swap = get_variance_swap_sabr(parameters, 0, ti)
    vol_swap_approximation = get_vol_swap_approximation_sabr(parameters, 0, ti, alpha)
    malliavin_vol_swap.append(get_volatility_swap_sabr_malliavin(parameters, 0, ti))


plt.plot(maturities, sampling_vol_swap, label="Sampling Volatility Swap", linestyle='-.')
plt.plot(maturities, malliavin_vol_swap, label="Malliavin Volatility Swap", linestyle='-.')

plt.legend()
plt.show()


