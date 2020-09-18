import numpy as np
import numba as nb


@nb.jit("f8(f8,f8,f8, f8)", nopython=True, nogil=True)
def get_variance_swap(v0: float, k: float, theta: float, t: float):
    exp_t = np.exp(- k * t)
    first_term = v0 * (1.0 - exp_t) / k
    second_term = theta * (t - (1.0 - exp_t) / k)
    return np.sqrt((first_term + second_term) / t)


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_rho_term_var_swap(v0: float, k: float, theta: float, epsilon: float, t: float):
    exp_t = np.exp(- k * t)
    first_term = ((epsilon * v0) / k) * ((1.0 - exp_t) / k - exp_t * t)
    second_term = ((epsilon * theta) / k) * (t - 2.0 * (1.0 - exp_t) / k + t * exp_t)
    return first_term + second_term
