import numba as nb
import numpy as np

from Tools.Types import ndarray
from MC_Engines.MC_SRoughVolatility import ToolsVariance
from Tools import AnalyticTools


@nb.jit("(f8, f8, f8, f8, f8[:,:], f8[:,:], f8[:], i8)", nopython=True, nogil=True)
def generate_paths_exp_super_rough(s0: float,
                                   sigma_0: float,
                                   nu: float,
                                   beta: float,
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
    var_w_t = ToolsVariance.get_variance(t_i_s[1:], beta)

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

            paths[k, j] = paths[k, j - 1] * np.exp(- 0.5 * int_v_t[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * d_w_i_s)

            # Keep the last brownians and variance of the RL process
            w_i_s_1 = w_t_k[j - 1]
            w_i_h_1 = w_t_k[j + no_time_steps - 2]
            var_w_t_i_1 = var_w_t[j - 1]

    return paths, sigma_i_1, int_v_t


@nb.jit("(f8, f8, f8, f8, f8[:,:], f8[:,:], f8[:], i8)", nopython=True, nogil=True)
def generate_paths_normal_super_rough(s0: float,
                                      sigma_0: float,
                                      nu: float,
                                      beta: float,
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

    for k in range(0, no_paths):
        w_t_k = AnalyticTools.apply_lower_tridiagonal_matrix(cholk_cov, noise[:, k])

        w_i_s_1 = 0.0
        w_i_h_1 = 0.0

        for j in range(1, no_time_steps):
            delta_i_s = t_i_s[j] - t_i_s[j - 1]

            # Brownian and Gaussian increments
            d_w_i_s = w_t_k[j - 1] - w_i_s_1
            d_w_i_h = w_t_k[j + no_time_steps - 2] - w_i_h_1

            # We must do that the volatility be positive. For the we will apply reflection
            sigma_i_1[k, j] = np.abs(sigma_i_1[k, j - 1] + nu * d_w_i_h)

            int_v_t[k, j - 1] = delta_i_s * 0.5 * (sigma_i_1[k, j - 1] * sigma_i_1[k, j - 1] +
                                                   sigma_i_1[k, j] * sigma_i_1[k, j])

            paths[k, j] = paths[k, j - 1] * np.exp(- 0.5 * int_v_t[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * d_w_i_s)

            # Keep the last brownians and variance of the RL process
            w_i_s_1 = w_t_k[j - 1]
            w_i_h_1 = w_t_k[j + no_time_steps - 2]

    return paths, sigma_i_1, int_v_t