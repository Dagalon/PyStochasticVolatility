import matplotlib.pylab as plt
import numpy as np
from Tools import RNG
from FractionalBrownian import fBM

no_paths = 1
no_time_steps = 2**10
t0 = 0.0
t1 = 1.0
z0 = 0.0
seed = 123456789
rng = RNG.RndGenerator(seed)

# Time steps to compute
t = np.linspace(t0, t1, int(no_time_steps * t1))

# Low Hurst parameter
low_hurst_parameter = 0.3
paths_low_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, low_hurst_parameter, no_paths, int(no_time_steps * t1))

# Medium Hurst parameter
rng.set_seed(seed)
medium_hurst_parameter = 0.5
paths_medium_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, medium_hurst_parameter, no_paths, int(no_time_steps * t1))

# Large Hurst parameter
rng.set_seed(seed)
large_hurst_parameter = 0.7
paths_large_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, large_hurst_parameter, no_paths, int(no_time_steps * t1))


fig, axs = plt.subplots(1, 3, figsize=(7, 3))

axs[0].plot(t, paths_low_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[0].set_title('$\it{H}=' + str(low_hurst_parameter))
axs[0].set_xticks(np.arange(t0, t1 + 0.5, 0.5))

axs[1].plot(t, paths_medium_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[1].set_title('\it{H}=' + str(medium_hurst_parameter))
axs[1].set_xticks(np.arange(t0, t1 + 0.5, 0.5))

axs[2].plot(t, paths_large_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[2].set_title('\it{H}=' + str(large_hurst_parameter))
axs[2].set_xticks(np.arange(t0, t1 + 0.5, 0.5))

plt.show()
