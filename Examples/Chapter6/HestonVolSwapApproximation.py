import matplotlib.pylab as plt
import numpy as np

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from AnalyticEngines.MalliavinMethod import ExpansionTools

dt = np.arange(1, 11, 1) * 0.05
no_dt_s = len(dt)

# simulation info
f0 = 100.0
epsilon = 0.5
k = 0.5
rho = -0.9
v0 = 0.05
sigma_0 = np.sqrt(0.05)
theta = 0.06

parameters = [k, theta, epsilon, rho, v0]

seed = 123456789
no_paths = 200000
delta_time = 1.0 / 365.0
T = 0.25

# random number generator
rnd_generator = RNG.RndGenerator(seed)

vol_swap_approximation = []
vol_swap_mc = []

for i in range(0, no_dt_s):
    no_time_steps = int(dt[i] / delta_time)
    rnd_generator.set_seed(seed)
    map_output = Heston_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                   no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                   rnd_generator)
    vol_swap_approximation.append(ExpansionTools.get_vol_swap_approximation_heston(np.array(parameters), 0.0, dt[i], sigma_0))
    vol_swap_mc.append(np.sqrt(np.mean(np.sum(map_output[Types.HESTON_OUTPUT.INTEGRAL_VARIANCE_PATHS], 1)) / dt[i]))

plt.plot(dt, vol_swap_approximation, label='vol_swap_approximation')
plt.plot(dt, vol_swap_mc, label='vol_swap_mc')

plt.legend()
plt.show()