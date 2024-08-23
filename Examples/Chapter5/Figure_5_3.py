import matplotlib.pylab as plt
import numpy as np
from FractionalBrownian import fBM
from Tools import RNG

no_paths = 1

t0 = 0.0
t1 = 1.0
no_time_steps = 2**10
z0 = 0.0
seed = 258
rng = RNG.RndGenerator(seed)

# Low Hurst parameter
low_hurst_parameter = 0.3
paths_low_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, low_hurst_parameter, no_paths, no_time_steps)

# Medium Hurst parameter
rng.set_seed(seed)
medium_hurst_parameter = 0.5
paths_medium_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, medium_hurst_parameter, no_paths,
                                                   no_time_steps)

# Large Hurst parameter
rng.set_seed(seed)
large_hurst_parameter = 0.7
paths_large_hurst_parameter = fBM.cholesky_method(t0, t1, z0, rng, large_hurst_parameter, no_paths,
                                                  no_time_steps)

paths_aggregated = paths_low_hurst_parameter[0, :] + paths_large_hurst_parameter[0, :]

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

t = np.linspace(t0, t1, no_time_steps)
axs[0].plot(t, paths_low_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[0].set_title('H=' + str(low_hurst_parameter))
axs[0].set_xticks(np.arange(t0, t1 + 0.25, 0.25))


axs[1].plot(t, paths_large_hurst_parameter[0, :].reshape(t.shape), color='black')
axs[1].set_title('H=' + str(large_hurst_parameter))
axs[1].set_xticks(np.arange(t0, t1 + 0.25, 0.25))


axs[2].plot(t, paths_aggregated, color='black')
axs[2].set_title('X_t process')
axs[2].set_xticks(np.arange(t0, t1 + 0.25, 0.25))

plt.savefig('C://Users//david//OneDrive//Desktop//PlotBooks//figure_5_6', dpi=500)
plt.show()

