import matplotlib.pylab as plt
import numpy as np

from Tools import RNG
from FractionalBrownian import fBM
from statsmodels.graphics.tsaplots import plot_acf

# simulation info
h = 0.8
no_time_steps = 2000
T = 1.0
d_t = np.linspace(0.0, T, no_time_steps)
seed = 2357575

# random number generator
rnd_generator = RNG.RndGenerator(seed)

rnd_generator.set_seed(seed)
path = fBM.truncated_fbm(0.0, T, 0.0, rnd_generator, h, 1, no_time_steps)
plot_acf(np.diff(path[0, :].reshape(d_t.shape)), lags=30, title='$\it{H}=$' + str(h))

plt.show()
