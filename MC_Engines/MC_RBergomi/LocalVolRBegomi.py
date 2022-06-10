import numba as nb
import numpy as np

from Tools.Types import ndarray


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
        k_i = m + 0.5 * t * int_v_t[i] - rho * int_sigma_t[i]
        vol_swap = np.sqrt(int_v_t[i] / t)
        phi_i = np.exp(- 0.5 * (k_i * k_i) / (rho_inv * rho_inv * t * vol_swap * vol_swap)) / (sqrt_time * vol_swap)
        partial_phi_i = (phi_i * k_i) / (strike * rho_inv * rho_inv * t * vol_swap * vol_swap)
        numerator_skew_var += partial_phi_i * v_t[i]
        numerator_skew += partial_phi_i
        numerator += phi_i * v_t[i]
        denominator += phi_i

    lv_value = numerator / denominator
    skew_val_aux = (lv_value * numerator_skew - numerator_skew_var) / denominator
    return 0.5 * skew_val_aux / np.sqrt(lv_value)

