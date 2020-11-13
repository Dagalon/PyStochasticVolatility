import numpy as np
import numba as nb

from Tools.Types import ndarray, TYPE_STANDARD_NORMAL_SAMPLING
from Tools.RNG import RndGenerator
from MC_Engines.MC_RBergomi import ToolsVariance


@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def h(x: float, hurst_parameter: float):
    return np.power((x - 1.0), 2.0 * hurst_parameter) - 2.0 + np.power((1.0 - x), 2.0 * hurst_parameter)


@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def gamma(k: float, hurst_parameter: float):
    return 0.5 * np.power(k, 2.0 * hurst_parameter) * h(1.0 / k, hurst_parameter)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def covariance(s: float, t: float, hurst_parameter: float):
    alpha = 2.0 * hurst_parameter
    return 0.5 * (np.power(t, alpha) + np.power(s, alpha) - np.power(np.abs(t - s), alpha))


@nb.jit("f8[:](i8, f8)", nopython=True, nogil=True)
def get_spectral_representation(n: int, hurst_parameter: float):
    output = np.zeros(2 * n)
    no_nodes = 2 * n
    for i in range(0, n):
        output[i] = gamma(np.float(i), hurst_parameter)
        output[no_nodes - i - 1] = output[i + 1]

    return output


@nb.jit("f8[:,:](f8[:],f8, f8[:,:], f8, i8, i8)", nopython=True, nogil=True)
def cholesky_method_jit(t_i: ndarray, z0: float, z: ndarray,  hurst_parameter: float, no_paths: int, no_time_steps):
    paths = np.zeros(shape=(no_paths, no_time_steps))
    sigma = np.zeros(shape=(no_time_steps - 1, no_time_steps - 1))

    paths[:, 0] = z0

    for i in range(0, no_time_steps - 1):
        for j in range(0, i + 1):
            sigma[i, j] = covariance(t_i[i + 1], t_i[j + 1], hurst_parameter)
            sigma[j, i] = sigma[i, j]

    lower_diag = np.linalg.cholesky(sigma)

    for i in range(0, no_paths):
        for i_time in range(1, no_time_steps):
            for j in range(0, i_time):
                paths[i, i_time] += z[i, j] * lower_diag[i_time - 1, j]

    return paths


def cholesky_method(t0: float,
                    t1: float,
                    z0: float,
                    rng_generator: RndGenerator,
                    hurst_parameter: float,
                    no_paths: int,
                    no_time_steps: int):

    t_i = np.linspace(t0, t1, no_time_steps)
    z_i = rng_generator.normal(mu=0.0, sigma=1.0, size=(no_paths, no_time_steps),
                               sampling_type=TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC)

    return cholesky_method_jit(t_i, z0, z_i, hurst_parameter, no_paths, no_time_steps)


@nb.jit("f8[:,:](f8[:],f8, f8[:,:], f8, i8, i8)", nopython=True, nogil=True)
def truncated_fbm_jit(t_i: ndarray, z0: float, z: ndarray,  hurst_parameter: float, no_paths: int, no_time_steps):
    paths = np.zeros(shape=(no_paths, no_time_steps))
    sigma = np.zeros(shape=(no_time_steps - 1, no_time_steps - 1))

    paths[:, 0] = z0

    for i in range(0, no_time_steps - 1):
        for j in range(0, i + 1):
            sigma[i, j] = ToolsVariance.get_volterra_covariance(t_i[i + 1], t_i[j + 1], hurst_parameter)
            sigma[j, i] = sigma[i, j]

    lower_diag = np.linalg.cholesky(sigma)

    for i in range(0, no_paths):
        for i_time in range(1, no_time_steps):
            for j in range(0, i_time):
                paths[i, i_time] += z[i, j] * lower_diag[i_time - 1, j]

    return paths


def truncated_fbm(t0: float,
                  t1: float,
                  z0: float,
                  rng_generator: RndGenerator,
                  hurst_parameter: float,
                  no_paths: int,
                  no_time_steps: int):

    t_i = np.linspace(t0, t1, no_time_steps)
    z_i = rng_generator.normal(mu=0.0, sigma=1.0, size=(no_paths, no_time_steps),
                               sampling_type=TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC)

    return cholesky_method_jit(t_i, z0, z_i, hurst_parameter, no_paths, no_time_steps)













