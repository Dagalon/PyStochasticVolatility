import numba as nb
import numpy as np
from Tools import Types


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
