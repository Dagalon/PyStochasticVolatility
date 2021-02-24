import matplotlib.pylab as plt
import numpy as np

from Tools import RNG
from FractionalBrownian import fBM
from statsmodels.graphics.tsaplots import plot_acf

# simulation info
h_s = [0.1, 0.3, 0.6, 0.8]
no_time_steps = 2000
T = 1.0
d_t = np.linspace(0.0, T, no_time_steps)
seed = 2357575

# random number generator
rnd_generator = RNG.RndGenerator(seed)

fig, axs = plt.subplots(4, 1)

for i in range(0, len(h_s)):
    rnd_generator.set_seed(seed)
    path = fBM.truncated_fbm(0.0, T, 0.0, rnd_generator, h_s[i], 1, no_time_steps)
    axs[i].plot(d_t, path[0, :].reshape(d_t.shape), color='black')
    axs[i].set_title('$\it{H}=$' + str(h_s[i]))

plt.show()
