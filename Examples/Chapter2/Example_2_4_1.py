import numpy as np
import matplotlib.pylab as plt
from MC_Engines.MC_Heston import Heston_Engine
from Tools import Types
from Tools import RNG
from AnalyticEngines.VolatilityTools import NonParametricEstimatorSLV

epsilon = 0.9
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05

f0 = 100
T = 0.25

seed = 123456789

delta = 1.0 / 32.0
no_time_steps = int(T / delta)
no_paths = 100000
strike = 120.0

rnd_generator = RNG.RndGenerator(seed)

parameters = [k, theta, epsilon, rho]

# Heston paths simulation
map_heston_output = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths,
                                                      no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                      rnd_generator)

# Compute the conditional expected by kernel estimators
x = np.linspace(60.0, 120.0, 500)

# Julien Guyon paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032&download=yes
h = 1.5 * np.sqrt(T) * np.power(no_paths, - 0.2)

gaussian_estimator_t = NonParametricEstimatorSLV.gaussian_kernel_estimator_slv(map_heston_output[Types.HESTON_OUTPUT.SPOT_VARIANCE_PATHS][:, -1],
                                                                               map_heston_output[Types.HESTON_OUTPUT.PATHS][:, -1],
                                                                               x,
                                                                               h)


plt.plot(x, gaussian_estimator_t, label="gaussian kernel estimator", color="black", linestyle="dotted")
plt.xlabel("S_t")
plt.ylabel("E(V_t|S_t=x)")


plt.legend()
plt.show()
