import numpy as np
import numba as nb
import matplotlib.pylab as plt

from ncephes import ndtri
from Tools import VolatilityEstimators
from scipy.integrate import quad
from functools import partial


# numba function to get market paths
@nb.jit("(f8,f8,f8,f8,f8,f8,i8,i8,i8)", nopython=True, nogil=True)
def get_paths(p0: float, sigma0: float, t: float, theta: float, w: float, k: float,
              no_paths: int, no_time_steps: int, seed: int):

    paths = np.empty(shape=(no_paths, no_time_steps))
    v_t = np.empty(shape=(no_paths, no_time_steps))

    paths[:, 0] = p0
    v_t[:, 0] = sigma0 * sigma0

    t_i_s = np.linspace(0.0, t, no_time_steps)
    np.random.rand(seed)
    nu = np.sqrt(2.0 * k * theta)

    for i in range(0, no_paths):
        u_s = np.random.rand(no_time_steps)
        u_sigma = np.random.rand(no_time_steps)

        for j in range(1, no_time_steps):
            delta_time = (t_i_s[j] - t_i_s[j - 1])
            z_sigma_i = ndtri(u_sigma[j])
            z_s_i = ndtri(u_s[j])
            exp_t = np.exp(- theta * delta_time)
            v_t[i, j] = v_t[i, j - 1] * exp_t + w * (1.0 - exp_t) + \
                        nu * np.sqrt(0.5 * ((1.0 - exp_t * exp_t) / theta)) * v_t[i, j - 1] * z_sigma_i
            paths[i, j] = paths[i, j - 1] + np.sqrt(v_t[i, j - 1]) * np.sqrt(delta_time) * z_s_i

    return paths, v_t, t_i_s


# market simulation
theta = 0.035
w = 0.6365
k = 0.2962

p0 = np.log(100)
sigma0 = np.sqrt(0.6365)

t = 1.0
seed = 123456

no_time_steps = 365 * 2
no_paths = 1

# Simulated integrated variance
paths, v_t, t_i_s = get_paths(p0, sigma0, t, theta, w, k, no_paths, no_time_steps, seed)

# We compute the fourier spot volatility estimator and we will compare with the simulated path.
spot_volatility_estimator = []
simulated_path_vol = []
t_j = []

for i in range(1, no_time_steps):
    t_j.append(t_i_s[i])
    estimator = VolatilityEstimators.get_spot_variance_fourier(paths, t_i_s, no_paths, t_i_s[i])
    simulated_path_vol.append(v_t[0, i])
    spot_volatility_estimator.append(estimator)

plt.plot(t_j, np.array(spot_volatility_estimator), color='black', label='estimator_path')
plt.plot(t_j, np.array(simulated_path_vol), linestyle='dashed', color='black', label='simulated_path',)
plt.legend()
plt.show()
