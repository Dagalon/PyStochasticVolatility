import numpy as np
import FractionalBrownian as FBM
from Tools import RNG, Functionals
from FractionalBrownian import fBM, ToolsFBM

no_paths = 1
no_time_steps = 2**12
t0 = 0.0
t1 = 1.0
z0 = 0.0
hurst_parameter = 0.5
seed = 123456789

rng = RNG.RndGenerator(seed)
paths = fBM.cholesky_method(t0, t1, z0, rng, hurst_parameter, no_paths, int(no_time_steps * t1))

# Time steps to compute
t = np.linspace(t0, t1, int(no_time_steps * t1))

# Compute mean
empirical_mean = np.mean(paths, axis=0)

# Compute variance
empirical_variance = np.var(paths, axis=0)
exact_variance = [fBM.covariance(t_i, t_i, hurst_parameter) for t_i in t]

# Compute covariance
no_full_time_steps = len(t)
empirical_covariance = np.zeros(shape=(no_time_steps, no_full_time_steps))
exact_covariance = np.zeros(shape=(no_time_steps, no_full_time_steps))

# for i in range(0, no_time_steps):
#     for j in range(0, i):
#         empirical_covariance[i, j] = np.mean(Functionals.dot_wise(paths[:, i], paths[:, j])) - \
#                                      empirical_mean[i] * empirical_mean[j]
#         exact_covariance[i, j] = fBM.covariance(t[i], t[j], hurst_parameter)
#         empirical_covariance[j, i] = empirical_covariance[i, j]
#         exact_covariance[j, i] = exact_covariance[i, j]

error_covariance = np.max(np.abs(empirical_covariance - exact_covariance))
error_variance = np.max(np.abs(exact_variance - empirical_variance))
error_mean = np.max(np.abs(empirical_mean))

# Estimation Hurst parameter
output = ToolsFBM.get_estimator_rs(paths[0, :], 7, 11)







