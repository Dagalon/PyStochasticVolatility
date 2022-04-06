import numpy as np
import numba as nb

from scipy.integrate import quad_vec, quad, quadrature
from Tools.Types import ndarray
from functools import partial


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def get_variance(t: ndarray, beta: float):
    return np.power(-np.log(t), -beta)


@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def get_kernel(x, beta):
    return np.sqrt(x * np.power(-np.log(x), beta + 1.0))


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_kernel_cov(u, s, t, beta):
    if s < t:
        d_t_s = (t - s)
        return np.power((d_t_s + u) * u, -0.5) * np.power(np.log(u) * np.log(d_t_s + u), -0.5 * (beta + 1.0))
    else:
        d_t_s = (s - t)
        return np.power((d_t_s + u) * u, -0.5) * np.power(np.log(u) * np.log(d_t_s + u), -0.5 * (beta + 1.0))


def get_volterra_covariance(s: float, t: float, beta: float):
    if s > 0.0 and t > 0.0:
        f_kernel = partial(get_kernel_cov, s=s, t=t, beta=beta)
        min_t_s = np.minimum(t, s)
        integral_value = quad(f_kernel, 0.0, min_t_s)

        return integral_value[0] * beta

    else:
        return 0.0


def get_covariance_w_v_w_t(s: float, t: float, rho: float, beta: float):
    if s > 0.0 and t > 0.0:
        f_kernel = partial(get_kernel, beta=beta)
        max_s_t = np.maximum(t - s, 0.0)
        integral_value = quad(f_kernel, 0.0, max_s_t)
        return np.sqrt(beta) * rho * integral_value[0]
    else:
        return 0.0


def get_covariance_matrix(t_i_s: ndarray, beta: float, rho: float):
    no_time_steps = len(t_i_s)
    cov = np.zeros(shape=(2 * no_time_steps, 2 * no_time_steps))

    for i in range(0, no_time_steps):
        for j in range(0, no_time_steps):
            cov[i, j] = np.minimum(t_i_s[i], t_i_s[j])

    for i in range(0, no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[j - no_time_steps], t_i_s[i], rho, beta)
            cov[j, i] = cov[i, j]

    for i in range(0, no_time_steps):
        for j in range(0, no_time_steps):
            cov[i + no_time_steps, j + no_time_steps] = get_volterra_covariance(t_i_s[i], t_i_s[j], beta)

    return cov
