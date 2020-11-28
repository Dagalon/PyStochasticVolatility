import numba as nb
import numpy as np

from Tools import Types
from ncephes import ndtr


@nb.jit("f8(f8,f8,f8,f8,i8)", nopython=True, nogil=True)
def black_scholes(f0: float, k: float, sigma: float, t: float, is_call: int):
    sqtr_t = np.sqrt(t)
    d_1 = (np.log(f0 / k) / (sigma * sqtr_t)) + 0.5 * sigma * sqtr_t
    d_2 = d_1 - sigma * sqtr_t
    return f0 * ndtr(d_1) - k * ndtr(d_2)


@nb.jit("f8[:](f8,i8,f8[:,:])", nopython=True, nogil=True)
def forward_start_call_operator(k: float, index_strike: int, x: Types.ndarray):
    no_paths = x.shape[0]
    no_time_steps = x.shape[1]
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0
    for i in range(0, no_paths):
        val = np.maximum((x[i, no_time_steps - 1] / x[i, index_strike]) - k, 0.0)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results


@nb.jit("f8[:](f8,i8,f8[:,:])", nopython=True, nogil=True)
def forward_start_put_operator(k: float, index_strike: int, x: Types.ndarray):
    no_paths = x.shape[0]
    no_time_steps = x.shape[1]
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0
    for i in range(0, no_paths):
        val = np.maximum(k - (x[i, no_time_steps - 1] / x[i, index_strike]), 0.0)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results


@nb.jit("f8[:](f8[:],f8[:],f8[:],f8,f8,f8)", nopython=True, nogil=True)
def forward_call_operator_control_variate(x, x1, vol_swap_t, k, t1, t2):
    no_paths = len(x)
    bs_prices = np.zeros(no_paths)
    v_prices = np.zeros(no_paths)

    results = np.empty(2)

    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        bs_prices[i] = black_scholes(1.0, k, vol_swap_t[i], (t2 - t1), 1) * x1[i]
        v_prices[i] = np.maximum(x[i] - x1[i] * k, 0.0)

    mean_bs_price = np.sum(bs_prices) / no_paths
    mean_v_price = np.sum(v_prices) / no_paths
    num_b = 0.0
    den_b = 0.0
    for i in range(0, no_paths):
        num_b += (v_prices[i] - mean_v_price) * (bs_prices[i] - mean_bs_price)
        den_b += (bs_prices[i] - mean_bs_price) * (bs_prices[i] - mean_bs_price)

    b_estimated = (num_b / den_b)

    for i in range(0, no_paths):
        value = v_prices[i] - b_estimated * (bs_prices[i] - mean_bs_price)
        acum += value
        acum_pow += value * value

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results


@nb.jit("f8[:](f8[:],f8[:],f8[:],f8,f8,f8)", nopython=True, nogil=True)
def forward_put_operator_control_variate(x, x1, vol_swap_t, k, t1, t2):
    no_paths = len(x)
    bs_prices = np.zeros(no_paths)
    v_prices = np.zeros(no_paths)

    results = np.empty(2)

    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        bs_prices[i] = black_scholes(1.0, k, vol_swap_t[i], (t2 - t1), 1) * x1[i]
        v_prices[i] = np.maximum(x1[i] * k - x[i], 0.0)

    mean_bs_price = np.sum(bs_prices) / no_paths
    mean_v_price = np.sum(v_prices) / no_paths
    num_b = 0.0
    den_b = 0.0
    for i in range(0, no_paths):
        num_b += (v_prices[i] - mean_v_price) * (bs_prices[i] - mean_bs_price)
        den_b += (bs_prices[i] - mean_bs_price) * (bs_prices[i] - mean_bs_price)

    b_estimated = (num_b / den_b)

    for i in range(0, no_paths):
        value = v_prices[i] - b_estimated * (bs_prices[i] - mean_bs_price)
        acum += value
        acum_pow += value * value

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results

