import matplotlib.pylab as plt
import numpy as np
from Tools import RNG
from FractionalBrownian import fBM, ToolsFBM

no_paths = 1
no_time_steps = 2**11
t0 = 0.0
t1 = 1.0
z0 = 0.0
hurst_parameter = 0.7
seed = 123456789

rng = RNG.RndGenerator(seed)
paths = fBM.cholesky_method(t0, t1, z0, rng, hurst_parameter, no_paths, int(no_time_steps * t1))


# Estimation Hurst parameter from the increments of the fractional brownian motion
noise = np.diff(paths[0, :])
output = ToolsFBM.get_estimator_rs(noise, 5, 10)

plt.plot(output[2], output[3], label='empirical_rs')
plt.plot(output[2], output[4], label='y='+str(round(output[0], 4)) + '+' + str(round(output[1], 4)) + '*x')

plt.legend()
plt.title('Estimation Hurst parameter H=' + str(hurst_parameter) + ' from noisefbm.')
plt.show()





