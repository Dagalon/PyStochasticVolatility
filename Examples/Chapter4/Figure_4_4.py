import matplotlib.pylab as plt
import numpy as np
from Tools import RNG, Functionals
from FractionalBrownian import fBM

no_paths = 1
no_time_steps = 2**10
t0 = 0.0
t1 = 1.0
z0 = 0.0
hurst_parameter = 0.3
seed = 123456789

rng = RNG.RndGenerator(seed)
paths = fBM.cholesky_method(t0, t1, z0, rng, hurst_parameter, no_paths, int(no_time_steps * t1))

# Time steps to compute
t = np.linspace(t0, t1, int(no_time_steps * t1))

plt.plot(t, paths[0, :])

plt.legend()
plt.title('Path fBM with H=' + str(hurst_parameter))
plt.show()
