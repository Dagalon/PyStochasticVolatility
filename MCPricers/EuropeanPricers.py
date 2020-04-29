import numba as nb
import numpy as np


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def call_operator(x, strike):
    no_paths = len(x)
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        val = np.maximum(x[i] - strike, 0.0)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def put_operator(x, strike):
    no_paths = len(x)
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        val = np.maximum(strike - x[i], 0.0)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def h_delta(x, delta, strike):
    if x < (strike - delta):
        return 0.0
    elif x > (strike + delta):
        return 1.0
    else:
        return 0.5 * (x - (strike - delta)) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def g_put_delta(x, delta, strike):
    if x < (strike - delta):
        return - (delta + (strike - delta - x))
    elif x > (strike + delta):
        return 0.0
    else:
        return - 0.25 * (strike + delta - x) * (strike + delta - x) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def g_call_delta(x, delta, strike):
    if x < (strike - delta):
        return 0.0
    elif x > (strike + delta):
        # return delta + (x - strike - delta)
        return x - strike
    else:
        return 0.25 * (x - (strike - delta)) * (x - (strike - delta)) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_call_delta(x, delta, strike):
    return np.maximum(x - strike, 0.0) - g_call_delta(x, delta, strike)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def i_gamma(x, delta, strike):
    if np.abs(x - strike) < delta:
        return 0.5 / delta
    else:
        return 0.0


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_gamma(x, delta, strike):
    return np.maximum(x - strike, 0.0) - g_call_delta(x, delta, strike)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_put_delta(x, delta, strike):
    return np.maximum(strike - x, 0.0) - g_put_delta(x, delta, strike)


@nb.jit("f8[:](f8[:], f8, f8, f8[:], f8)", nopython=True, nogil=True)
def malliavin_delta_call_put(x, strike, f0, weights, call_put):
    no_paths = len(x)
    results = np.empty(2)
    delta = 0.001
    # delta = 0.01 * strike
    acum = 0.0
    acum_pow = 0.0

    if call_put > 0:
        for i in range(0, no_paths):
            acum_part_1 = (x[i] / f0) * h_delta(x[i], delta, strike)
            acum_part_2 = f_call_delta(x[i], delta, strike) * weights[i]
            val = (acum_part_1 + acum_part_2)
            acum += val
            acum_pow += val * val

        results[0] = acum / no_paths
        results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

        return results
    else:
        for i in range(0, no_paths):
            acum_part_1 = (x[i] / f0) * (1.0 - h_delta(x[i], delta, strike))
            acum_part_2 = f_put_delta(x[i], delta, strike) * weights[i]
            val = (acum_part_1 + acum_part_2)
            acum += val
            acum_pow += val * val

        results[0] = acum / no_paths
        results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

        return results


@nb.jit("f8[:](f8[:], f8, f8, f8[:])", nopython=True, nogil=True)
def malliavin_gamma_call_put(x, strike, f0, gamma_weights):
    no_paths = len(x)
    results = np.empty(2)
    # delta = 0.01
    delta = 0.0001 * strike
    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        acum_part_1 = np.power((x[i] / f0), 2.0) * i_gamma(x[i], delta, strike)
        acum_part_2 = gamma_weights[i] * f_gamma(x[i], delta, strike)
        val = (acum_part_1 + acum_part_2)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results



