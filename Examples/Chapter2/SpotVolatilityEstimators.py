import numpy as np
import numba as nb
import matplotlib.pylab as plt

from ncephes import ndtri
from Tools import VolatilityEstimators
from scipy.integrate import quad
from functools import partial


# Compute analytic variance of the underlying process at time t_i_s[-1]
def get_analytic_variance(t: float, b0: float, b1: float, b2: float):
    f = partial(VolatilityEstimators.get_mean_sigma, b0=beta0, b1=beta1, b2=beta2)
    integral_value = quad(f, 0.0, t)
    return integral_value[0]


# numba function to get market paths
@nb.jit("(f8,f8,f8,f8,f8,f8,f8,f8,i8,i8,i8)", nopython=True, nogil=True)
def get_paths(p0: float, sigma0: float, beta0: float, beta1: float, beta2: float, mu: float, rho: float, t: float,
              no_paths: int, no_time_steps: int, seed: int):
    paths = np.empty(shape=(no_paths, no_time_steps))
    sigma_t = np.empty(shape=(no_paths, no_time_steps))
    integral_sigma_t = np.empty(shape=(no_paths, no_time_steps - 1))
    np.random.rand(seed)

    paths[:, 0] = p0
    t_i_s = np.linspace(0.0, t, no_time_steps)
    # sigma_tau_0 = - 0.5 / beta2

    rho_inv = np.sqrt(1.0 - rho * rho)

    for i in range(0, no_paths):
        u_s = np.random.rand(no_time_steps)
        u_sigma = np.random.rand(no_time_steps)
        # tau_t_i_1 = ndtri(np.random.rand()) * sigma_tau_0
        tau_t_i_1 = sigma0
        sigma_t[i, 0] = np.exp(beta0 + beta1 * tau_t_i_1)

        for j in range(1, no_time_steps):
            delta_time = (t_i_s[j] - t_i_s[j - 1])
            z_sigma_i = ndtri(u_sigma[j])
            z_s_i = ndtri(u_s[j])
            var_tau = 0.5 * (np.exp(2.0 * beta2 * delta_time) - 1.0) / beta2
            tau_t_i = tau_t_i_1 * np.exp(beta2 * delta_time) + np.sqrt(var_tau) * z_sigma_i
            sigma_t[i, j] = np.exp(beta0 + beta1 * tau_t_i)
            mid_sigma = 0.5 * (sigma_t[i, j - 1] + sigma_t[i, j])
            paths[i, j] = paths[i, j - 1] + mu * delta_time + \
                          mid_sigma * np.sqrt(delta_time) * (rho * z_sigma_i + rho_inv * z_s_i)
            integral_sigma_t[i, j - 1] = delta_time * sigma_t[i, j - 1] * sigma_t[i, j - 1]
            tau_t_i_1 = tau_t_i

    return paths, sigma_t, t_i_s, integral_sigma_t


# market simulation
mu = 0.03
# mu = 0.0
beta1 = 0.125
# beta1 = 0.0
beta2 = -0.025
beta0 = 0.5 * beta1 / beta2
# beta0 = np.log(0.3)
rho = - 0.3

p0 = np.log(9)
sigma0 = 0.3

# t = (1.0 / 365.0) * (1.0 / 24.0)
t = 1.0
seed = 123456

# no_time_steps = 3600
no_time_steps = 365
no_paths = 5000

# Simulated integrated variance
# Sampling each 1sg during 1h
paths, sigma_t, t_i_s, integral_sigma_t = get_paths(p0, sigma0, beta0, beta1, beta2, mu, rho, t, no_paths, no_time_steps, seed)

mean = p0 + mu * t_i_s[-1]
empirical_mean = np.mean(paths[:, -1])

variance = get_analytic_variance(t_i_s[-1], beta0, beta1, beta2)
empirical_variance = np.var(paths[:, -1]) * (no_paths - 1) / no_paths
estimator_integrated_variance = VolatilityEstimators.get_integrated_variance_fourier(paths, t_i_s, no_paths)

# We compute the fourier spot volatility estimator and we will compare with the simulated path.
spot_volatility_estimator = []
simulated_path_vol = []
t_j = []

for i in range(1, no_time_steps, 2):
    t_j.append(t_i_s[i])
    estimator = VolatilityEstimators.get_spot_variance_fourier(paths, t_i_s, no_paths, t_i_s[i])
    simulated_path_vol.append(sigma_t[0, i] * sigma_t[0, i])
    spot_volatility_estimator.append(estimator.mean())

plt.plot(t_j, np.array(spot_volatility_estimator), label='estimator_path')
plt.plot(t_j, np.array(simulated_path_vol), label='simulated_path')
plt.legend()
plt.show()
