import matplotlib.pylab as plt
import numpy as np
from FractionalBrownian import fBM
from Tools import RNG

no_paths = 1
no_time_steps = 2 ** 10
t0 = 0.0
t1 = 1.0
z0 = 0.0
seed = 123456789
rng = RNG.RndGenerator(seed)

# Low Hurst parameter
low_hurst_parameter = 0.2
paths_low_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, low_hurst_parameter, no_paths, int(no_time_steps * t1))

# Medium Hurst parameter
rng.set_seed(seed)
medium_hurst_parameter = 0.5
paths_medium_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, medium_hurst_parameter, no_paths,
                                                  int(no_time_steps * t1))

# Large Hurst parameter
rng.set_seed(seed)
large_hurst_parameter = 0.8
paths_large_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, large_hurst_parameter, no_paths,
                                                  int(no_time_steps * t1))

paths_aggregated = paths_low_hurst_parameter[0, :] + paths_large_hurst_parameter[0, :]

fig, axs = plt.subplots(1, 3)

t = np.linspace(t0, t1, no_time_steps)
axs[0].plot(t, paths_low_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[0].set_title('H=' + str(low_hurst_parameter))
axs[0].set_xticks(np.arange(t0, t1 + 0.25, 0.25))


axs[1].plot(t, paths_medium_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[1].set_title('H=' + str(medium_hurst_parameter))
axs[1].set_xticks(np.arange(t0, t1 + 0.25, 0.25))


axs[2].plot(t, paths_large_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[2].set_title('H=' + str(large_hurst_parameter))
axs[2].set_xticks(np.arange(t0, t1 + 0.25, 0.25))


plt.show()

