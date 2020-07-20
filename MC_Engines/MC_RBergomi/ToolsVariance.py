import numpy as np
import numba as nb
from Tools.Types import ndarray
from ncephes import hyp2f1
from Tools import Functionals


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def get_volterra_covariance(s: float, t: float, h: float):
    # We suppose that t > s
    if s > 0.0:
        gamma = 0.5 - h
        x = s / t

        return ((1.0 - 2.0 * gamma) / (1.0 - gamma)) * np.power(x, gamma) * hyp2f1(1.0, gamma, 2.0 - gamma, x)

    else:
        return 0.0


@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def get_volterra_variance(t: float, h: float):
    return np.power(t, - 2.0 * h)


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_covariance_w_v_w_t(s: float, t: float, rho: float, h: float):
    gamma = 0.5 - h
    d_h = np.sqrt(2.0 * h) / (h + 0.5)
    return rho * d_h * (np.power(t, h + 0.5) - np.power(t - np.minimum(s, t), h + 0.5))


@nb.jit("f8[:,:](f8[:], f8, f8)", nopython=True, nogil=True)
def get_covariance_matrix(t_i_s: ndarray, h: float, rho: float):
    no_time_steps = len(t_i_s)
    cov = np.zeros(shape=(2 * no_time_steps, 2 * no_time_steps))
    for i in range(0, no_time_steps):
        for j in range(0, i + 1):
            cov[i, j] = t_i_s[j]
            cov[j, i] = cov[i, j]

    for i in range(0, no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[i], t_i_s[j - no_time_steps], h, rho)
            cov[j, i] = cov[i, j]

    for i in range(0, no_time_steps):
        for j in range(0, i + 1):
            cov[i + no_time_steps, j + no_time_steps] = get_volterra_covariance(t_i_s[j], t_i_s[i], h)
            cov[j + no_time_steps, i + no_time_steps] = cov[i + no_time_steps, j + no_time_steps]

    return cov


def generate_paths(s0: float, v0: float, nu: float, noise: ndarray, cholk_cov: ndarray, t_i_s: ndarray, no_paths: int):
    no_time_steps = len(t_i_s)
    delta_i_s = np.diff(t_i_s)

    paths = np.zeros(no_paths, no_time_steps)
    w_i_s = np.zeros(2 * no_time_steps - 1)
    v_i_1 = np.zeros(no_time_steps)

    v_i_1.fill(v0)
    paths[:, 0] = s0

    for k in range(0, no_paths):
        w_i_s = Functionals.apply_lower_tridiagonal_matrix(cholk_cov, noise[:, k])

        for j in range(1, no_time_steps):
            d_w_v_t = w_i_s
