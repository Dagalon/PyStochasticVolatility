import numba as nb
import numpy as np

from Tools.Types import ndarray


@nb.jit("f8(f8, f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_pdf(t: float, f0: float, strike: float, rho: float, v_t: ndarray, int_v_t: ndarray, int_sigma_t: ndarray):
    no_paths = len(v_t)
    log_f0 = np.log(f0)
    log_strike = np.log(strike)
    rho_inv = np.sqrt(1.0 - rho * rho)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    mean = 0.0

    alpha = 1.0 / no_paths

    for i in range(0, no_paths):
        vol_swap = np.sqrt(int_v_t[i] / t)
        v_sigma = rho_inv * np.sqrt(t) * vol_swap
        log_m = log_f0 - 0.5 * rho * rho * t * vol_swap * vol_swap + rho * int_sigma_t[i]
        mean += (alpha * np.exp(- 0.5 * np.power(log_strike - log_m + 0.5 * v_sigma * v_sigma, 2.0) / (v_sigma * v_sigma)) / (strike * sqrt_2pi * v_sigma))

    return mean


@nb.jit("f8(f8, f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_local_vol(t: float, f0: float, strike: float, rho: float, v_t: ndarray, int_v_t: ndarray, int_sigma_t: ndarray):
    no_paths = len(v_t)
    m = np.log(strike / f0)
    rho_inv = np.sqrt(1.0 - rho * rho)
    sqrt_time = np.sqrt(t)
    numerator = 0.0
    denominator = 0.0

    for i in range(0, no_paths):
        vol_swap = np.sqrt(int_v_t[i] / t)
        k_i = m + 0.5 * t * vol_swap * vol_swap - rho * int_sigma_t[i]
        phi_i = np.exp(- 0.5 * (k_i * k_i) / (rho_inv * rho_inv * t * vol_swap * vol_swap)) / (sqrt_time * vol_swap)
        numerator += phi_i * v_t[i]
        denominator += phi_i

    return np.sqrt(numerator / denominator)


@nb.jit("f8(f8, f8, f8, f8, f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def get_skew_local_vol(t: float, f0: float, strike: float, rho: float, v_t: ndarray, int_v_t: ndarray, int_sigma_rho_t: ndarray):
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
        k_i = m + 0.5 * t * vol_swap * vol_swap - rho * int_sigma_rho_t[i]
        phi_i = np.exp(- 0.5 * (k_i * k_i) / (rho_inv * rho_inv * t * vol_swap * vol_swap)) / (sqrt_time * vol_swap)
        partial_phi_i = (phi_i * k_i) / (strike * t * vol_swap * vol_swap)
        numerator_skew_var += partial_phi_i * v_t[i]
        numerator_skew += partial_phi_i
        numerator += phi_i * v_t[i]
        denominator += phi_i

    term_1 = (np.sqrt(numerator) * numerator_skew) / np.power(denominator, 1.5)
    term_2 = numerator_skew_var / np.sqrt(numerator * denominator)

    return (term_1 - term_2) / (2.0 * rho_inv * rho_inv)

