import numpy as np
import numba as nb

from Tools import Types


@nb.jit("f8[:](f8[:],f8)", nopython=True, nogil=True)
def get_sabr_moments(parameters: Types.ndarray, t: float):
    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]
    output = np.zeros(2)
    exp_nu_t = np.exp(nu * nu * t)
    output[0] = np.power(alpha / nu, 2.0) * (exp_nu_t - 1.0)
    output[1] = (np.power(alpha, 4.0) / (12.0 * np.power(nu, 2))) * (np.power(exp_nu_t, 6.0) - 1.0)
    return output


@nb.jit("Tuple((f8[:], f8))(f8[:,:],f8[:],f8[:],i8,i8)", nopython=True, nogil=True)
def get_sabr_asymptotic(sigma_t: Types.ndarray, t_s: Types.ndarray, parameters: Types.ndarray,
                        no_paths: int, no_time_steps: int):
    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]
    mean_output = np.zeros(no_paths)

    sabr_int_moments = get_sabr_moments(parameters, t_s[-1])

    for i in range(0, no_paths):
        for j in range(1, no_time_steps):
            delta_time = t_s[j] - t_s[j - 1]
            mean_output[i] += (0.25 * delta_time * np.power(sigma_t[i, j - 1], 4.0) -
                               0.5 * nu * rho * delta_time * np.power(sigma_t[i, j - 1], 3.0))

    return mean_output, 4.0 * sabr_int_moments[1]


# @nb.jit("Tuple((f8[:], f8[:]))(f8[:,:],f8[:],f8[:],i8,i8)", nopython=True, nogil=True)
def get_discretization(s_t: Types.ndarray, t_s: Types.ndarray, parameters: Types.ndarray,
                       no_paths: int, no_time_steps: int):

    output = np.zeros(2)
    mean_log_returns = np.zeros(no_paths)
    variance_log_returns = np.zeros(no_paths)

    sabr_int_moments = get_sabr_moments(parameters, t_s[-1])

    for i in range(0, no_paths):
        for j in range(1, no_time_steps):
            r_i_j = np.log(s_t[i, j] / s_t[i, j - 1])
            mean_log_returns[i] += np.power(r_i_j, 2.0)
        mean_log_returns[i] -= sabr_int_moments[0]
        variance_log_returns[i] = np.power(mean_log_returns[i], 2.0)

    return mean_log_returns, variance_log_returns
