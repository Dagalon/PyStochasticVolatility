import numba as nb
import numpy as np
from Tools import Types


@nb.jit("f8(f8[:],f8,f8,f8)", nopython=True, nogil=True)
def get_vol_swap_approximation_sabr(parameters: Types.ndarray, t0: float, t1: float, sigma_t0: float):
    alpha = parameters[0]
    return sigma_t0 * (1.0 + alpha * alpha * (t1 - t0)) / 12.0


@nb.jit("f8(f8[:],f8,f8,f8)", nopython=True, nogil=True)
def get_vol_swap_approximation_heston(parameters: Types.ndarray, t0: float, t1: float, sigma_t0: float):
    k = parameters[0]
    theta = parameters[1]
    epsilon = parameters[2]

    adjusment = 0.5 * (((0.5 * k * theta - epsilon * epsilon / 12.0) * (1.0 / sigma_t0)) - 0.5 * k * sigma_t0) + (
                epsilon * epsilon) / (48.0 * sigma_t0)

    return sigma_t0 + (t1 - t0) * adjusment


# @nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def get_iv_atm_heston_approximation(parameters: Types.ndarray, t: float):
    epsilon = parameters[2]
    rho = parameters[3]
    v0 = parameters[4]
    sigma_0 = np.sqrt(v0)

    vol_swap = get_vol_swap_approximation_heston(parameters, 0.0, t, sigma_0)
    adjustment = np.power(epsilon * rho, 2.0) / (96.0 * sigma_0) + 0.125 * rho * epsilon

    return vol_swap + t * adjustment


# @nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def get_iv_atm_rbergomi_approximation(parameters: Types.ndarray, vol_swap: float, sigma_0: float, t: float):
    nu = parameters[0]
    rho = parameters[1]
    h = parameters[2]

    h_1_2 = (h + 0.5)
    h_3_2 = (h + 1.5)
    h_1 = (h + 1.0)
    adjustment = sigma_0 * np.power(nu * rho, 2.0) * (h / h_1_2) * (0.75 / (h_1_2 * h_3_2 * h_3_2) - 0.125 / (h_1 * h_1_2) - 1.0 / h_1)
    return vol_swap + adjustment * np.powe(t, 2.0 * h)