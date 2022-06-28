import numba as nb
import numpy as np

from Tools.Types import ndarray
from Tools.AnalyticTools import normal_pdf


@nb.jit("f8(f8, f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_local_vol(t: float, f0: float, strike: float, rho: float, v_t: ndarray, int_v_t: ndarray, int_sigma_t: ndarray):
    no_paths = len(v_t)
    m = np.log(strike / f0)
    rho_inv = np.sqrt(1.0 - rho * rho)
    sqrt_time = np.sqrt(t)
    numerator = 0.0
    denominator = 0.0

    for i in range(0, no_paths):
        k_i = m + 0.5 * t * int_v_t[i] - rho * int_sigma_t[i]
        vol_swap = np.sqrt(int_v_t[i] / t)
        phi_i = np.exp(- 0.5 * (k_i * k_i) / (rho_inv * rho_inv * t * vol_swap * vol_swap)) / (sqrt_time * vol_swap)
        numerator += phi_i * v_t[i]
        denominator += phi_i

    return np.sqrt(numerator / denominator)


@nb.jit("f8(f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_local_vol_rough(t: float, f0: float, strike: float, v_t: ndarray, int_v_t: ndarray, int_sigma_rho_t: ndarray):
    no_paths = len(v_t)
    m = np.log(strike / f0)
    sqrt_time = np.sqrt(t)
    numerator = 0.0
    denominator = 0.0

    for i in range(0, no_paths):
        vol_swap = np.sqrt(int_v_t[i] / t)
        d_i = (m + 0.5 * t * vol_swap * vol_swap) / (sqrt_time * vol_swap)
        phi_i = normal_pdf(0.0, 1.0, d_i) / (strike * sqrt_time * vol_swap)
        numerator += phi_i * v_t[i]
        denominator += phi_i

    return np.sqrt(numerator / denominator)


@nb.jit("f8(f8, f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_skew_local_vol(t: float, f0: float, strike: float, rho: float, v_t: ndarray, int_v_t: ndarray, int_sigma_t: ndarray):
    no_paths = len(v_t)
    m = np.log(strike / f0)
    rho_inv = np.sqrt(1.0 - rho * rho)
    sqrt_time = np.sqrt(t)
    numerator = 0.0
    numerator_skew = 0.0
    numerator_skew_var = 0.0
    denominator = 0.0

    for i in range(0, no_paths):
        vol_swap = np.sqrt(int_v_t[i] / t)
        k_i = m + 0.5 * t * vol_swap * vol_swap - rho * int_sigma_t[i]
        phi_i = np.exp(- 0.5 * (k_i * k_i) / (rho_inv * rho_inv * t * vol_swap * vol_swap)) / (sqrt_time * vol_swap)
        partial_phi_i = (phi_i * k_i) / (strike * t * vol_swap * vol_swap)
        numerator_skew_var += partial_phi_i * v_t[i]
        numerator_skew += partial_phi_i
        numerator += phi_i * v_t[i]
        denominator += phi_i

    term_1 = (np.sqrt(numerator) * numerator_skew) / np.power(denominator, 1.5)
    term_2 = numerator_skew_var / np.sqrt(numerator * denominator)

    return (term_1 - term_2) / (2.0 * rho_inv * rho_inv)


@nb.jit("f8(f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_skew_local_rough(t: float, f0: float, strike: float, v_t: ndarray, int_v_t: ndarray, int_sigma_rho_t: ndarray):
    no_paths = len(v_t)
    m = np.log(strike / f0)
    sqrt_time = np.sqrt(t)
    numerator = 0.0
    numerator_skew = 0.0
    numerator_skew_var = 0.0
    denominator = 0.0

    for i in range(0, no_paths):
        vol_swap = np.sqrt(int_v_t[i] / t)
        d_i = (m + 0.5 * t * vol_swap * vol_swap) / (sqrt_time * vol_swap)
        phi_i = normal_pdf(0.0, 1.0, d_i) / (strike * sqrt_time * vol_swap)
        partial_phi_i = - (1.0 / strike) * phi_i * (1.0 + d_i / (strike * t * vol_swap * vol_swap))
        numerator_skew_var += partial_phi_i * v_t[i]
        numerator += phi_i * v_t[i]
        denominator += phi_i
        numerator_skew += partial_phi_i

    lv_square = numerator / denominator
    term_1 = numerator_skew_var / denominator
    term_2 = lv_square * numerator_skew / denominator

    return term_1 - term_2

