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


def get_vol_swap_local_vol(t0: float, t1: float, f0: float, lv_f0: float,
                           fd_lv_f0: float, sd_lv_f0: float):
    adjustment = f0 * f0 * (0.25 * sd_lv_f0 * lv_f0 + fd_lv_f0 * fd_lv_f0 * (0.25 * lv_f0 * lv_f0 - 2.0 / 3.0))
    return lv_f0 * (1.0 + (t1 - t0) * adjustment)


def get_iv_atm_local_vol_approximation(f0: float, lv_f0: float, fd_lv_f0: float, sd_lv_f0: float, t: float):
    vol_swap_approx = get_vol_swap_local_vol(0.0, t, f0, lv_f0, fd_lv_f0, sd_lv_f0)
    adjustment = - 0.125 * fd_lv_f0 * fd_lv_f0 * lv_f0 * f0 - (1.0 / 6.0) * sd_lv_f0 * lv_f0 * lv_f0 * f0 + \
                 (1.0 / 12.0) * fd_lv_f0 * lv_f0 * lv_f0

    return vol_swap_approx + f0 * t * adjustment


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
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
    adjustment = sigma_0 * np.power(nu * rho, 2.0) * (h / h_1_2) * (
            0.75 / (h_1_2 * h_3_2 * h_3_2) - 1.0 / (h_1 * h_1_2))
    return vol_swap + adjustment * np.power(t, 2.0 * h)
