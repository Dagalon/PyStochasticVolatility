import numpy as np
import numba as nb
from Tools.Types import ndarray
from ncephes import hyp2f1
from Tools import AnalyticTools
from math import gamma


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def get_volterra_covariance(s: float, t: float, h: float):
    if s < t:
        x = s / t
        alpha = 2.0 * np.power(s, h + 0.5) * np.power(t, h - 0.5) * h / (h + 0.5)
        return alpha * hyp2f1(0.5 - h, 1.0, h + 1.5, x)

    elif t < s:
        x = t / s
        alpha = 2.0 * np.power(t, h + 0.5) * np.power(s, h - 0.5) * h / (h + 0.5)
        return alpha * hyp2f1(0.5 - h, 1.0, h + 1.5, x)

    else:
        return np.power(s, 2.0 * h)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def get_fbm_covariance(s: float, t: float, h: float):
    return 0.5 * (np.power(np.abs(t), 2.0 * h) + np.power(np.abs(s), 2.0 * h) - np.power(np.abs(t - s), 2.0 * h))


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def get_fbm_variance(t: float, h: float):
    return np.power(t, 2.0 * h)


@nb.jit("f8(f8, f8, f8, f8)", nopython=False, nogil=True)
def get_covariance_fbm_w_t(s: float, t: float, rho: float, h: float):
    h_3_2 = h + 1.5
    h_1_2 = h + 0.5
    if s < t:
        return (rho / gamma(h_3_2)) * np.power(s, h_1_2)
    else:
        return (rho / gamma(h_3_2)) * (np.power(s, h_1_2) - np.power(s - t, h_1_2))


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def get_volterra_variance(t: ndarray, h: float):
    no_elements = len(t)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = np.power(t[i], 2.0 * h)
    return output


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_covariance_w_v_w_t(s: float, t: float, rho: float, h: float):
    d_h = np.sqrt(2.0 * h) / (h + 0.5)
    return rho * d_h * (np.power(s, h + 0.5) - (np.power(s - np.minimum(s, t), h + 0.5)))


@nb.jit("f8[:,:](f8[:], f8, f8)", nopython=True, nogil=True)
def get_covariance_matrix(t_i_s: ndarray, h: float, rho: float):
    no_time_steps = len(t_i_s)
    cov = np.zeros(shape=(2 * no_time_steps, 2 * no_time_steps))
    for i in range(0, no_time_steps):
        for j in range(0, no_time_steps):
            cov[i, j] = np.minimum(t_i_s[i], t_i_s[j])
            # cov[j, i] = cov[i, j]

    for i in range(0, no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[j - no_time_steps], t_i_s[i], rho, h)
            cov[j, i] = cov[i, j]

    # for i in range(no_time_steps, 2 * no_time_steps):
    #     for j in range(0, no_time_steps):
    #         cov[i, j] = get_covariance_w_v_w_t(t_i_s[i - no_time_steps], t_i_s[j], rho, h)

    for i in range(0, no_time_steps):
        # for j in range(0, i + 1):
        for j in range(0, no_time_steps):
            cov[i + no_time_steps, j + no_time_steps] = get_volterra_covariance(t_i_s[i], t_i_s[j], h)
            # cov[i + no_time_steps, j + no_time_steps] = get_fbm_covariance(t_i_s[j], t_i_s[i], h)

    return cov


@nb.jit("(f8, f8, f8, f8, f8[:,:], f8[:,:], f8[:], i8)", nopython=True, nogil=True)
def generate_paths_rbergomi(s0: float,
                            sigma_0: float,
                            nu: float,
                            h: float,
                            noise: ndarray,
                            cholk_cov: ndarray,
                            t_i_s: ndarray,
                            no_paths: int):
    no_time_steps = len(t_i_s)

    paths = np.zeros(shape=(no_paths, no_time_steps))
    int_v_t = np.zeros(shape=(no_paths, no_time_steps - 1))
    sigma_i_1 = np.zeros(shape=(no_paths, no_time_steps))

    sigma_i_1[:, 0] = sigma_0
    paths[:, 0] = s0

    # we compute before a loop of variance of the variance process
    var_w_t = get_volterra_variance(t_i_s[1:], h)

    for k in range(0, no_paths):
        w_t_k = AnalyticTools.apply_lower_tridiagonal_matrix(cholk_cov, noise[:, k])

        w_i_s_1 = 0.0
        w_i_h_1 = 0.0
        var_w_t_i_1 = 0.0

        for j in range(1, no_time_steps):
            delta_i_s = t_i_s[j] - t_i_s[j - 1]

            # Brownian and Gaussian increments
            d_w_i_s = w_t_k[j - 1] - w_i_s_1
            d_w_i_h = w_t_k[j + no_time_steps - 2] - w_i_h_1

            sigma_i_1[k, j] = sigma_i_1[k, j - 1] * np.exp(- 0.5 * nu * nu * (var_w_t[j - 1] - var_w_t_i_1) +
                                                           nu * d_w_i_h)
            int_v_t[k, j - 1] = delta_i_s * 0.5 * (sigma_i_1[k, j - 1] * sigma_i_1[k, j - 1] +
                                                   sigma_i_1[k, j] * sigma_i_1[k, j])
            # int_v_t[k, j - 1] = delta_i_s * sigma_i_1[k, j - 1] * sigma_i_1[k, j - 1]
            paths[k, j] = paths[k, j - 1] * np.exp(- 0.5 * int_v_t[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * d_w_i_s)

            # Keep the last brownians and variance of the RL process
            w_i_s_1 = w_t_k[j - 1]
            w_i_h_1 = w_t_k[j + no_time_steps - 2]
            var_w_t_i_1 = var_w_t[j - 1]

    return paths, sigma_i_1, int_v_t


@nb.jit("(f8, f8, f8, f8, f8[:,:], f8[:,:], f8[:], i8)", nopython=True, nogil=True)
def generate_paths_rexpou1f(s0: float,
                            sigma_0: float,
                            nu: float,
                            h: float,
                            noise: ndarray,
                            cholk_cov: ndarray,
                            t_i_s: ndarray,
                            no_paths: int):
    no_time_steps = len(t_i_s)

    paths = np.zeros(shape=(no_paths, no_time_steps))
    int_v_t = np.zeros(shape=(no_paths, no_time_steps - 1))
    sigma_i_1 = np.zeros(shape=(no_paths, no_time_steps))

    sigma_i_1[:, 0] = sigma_0
    paths[:, 0] = s0

    # we compute before a loop of variance of the variance process
    var_w_t = get_volterra_variance(t_i_s[1:], h)

    for k in range(0, no_paths):
        w_t_k = AnalyticTools.apply_lower_tridiagonal_matrix(cholk_cov, noise[:, k])

        w_i_s_1 = 0.0
        w_i_h_1 = 0.0
        var_w_t_i_1 = 0.0

        for j in range(1, no_time_steps):
            delta_i_s = t_i_s[j] - t_i_s[j - 1]

            # Brownian and Gaussian increments
            d_w_i_s = w_t_k[j - 1] - w_i_s_1
            d_w_i_h = w_t_k[j + no_time_steps - 2] - w_i_h_1

            sigma_i_1[k, j] = sigma_i_1[k, j - 1] * np.exp(- 0.5 * nu * nu * (var_w_t[j - 1] - var_w_t_i_1) +
                                                           nu * d_w_i_h)
            int_v_t[k, j - 1] = delta_i_s * 0.5 * (sigma_i_1[k, j - 1] * sigma_i_1[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * sigma_i_1[k, j - 1])
            paths[k, j] = paths[k, j - 1] * np.exp(- 0.5 * int_v_t[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * d_w_i_s)

            # Keep the last brownians and variance of the RL process
            w_i_s_1 = w_t_k[j - 1]
            w_i_h_1 = w_t_k[j + no_time_steps - 2]
            var_w_t_i_1 = var_w_t[j - 1]

    return paths, sigma_i_1, int_v_t



@nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def beta(t, m):
    if t < 1.0e-05:
        return t
    else:
        return (1.0 - np.exp(- m * t)) / m
