import numba as nb
import numpy as np
from Tools import Types


@nb.jit("f8[:](f8,f8[:],f8[:])", nopython=True, nogil=True)
def forward_start_call_operator(k: float, x0: Types.ndarray, x1: Types.ndarray):
    no_paths = len(x0)
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0
    for i in range(0, no_paths):
        val = np.maximum((x1[i] / x0[i]) - k, 0.0)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results


@nb.jit("f8[:](f8,f8[:],f8[:])", nopython=True, nogil=True)
def forward_start_put_operator(k: float, x0: Types.ndarray, x1: Types.ndarray):
    no_paths = len(x0)
    results = np.empty(2)
    acum = 0.0
    acum_pow = 0.0
    for i in range(0, no_paths):
        val = np.maximum(k - (x1[i] / x0[i]), 0.0)
        acum += np.maximum(x1[i] / x0[i] - k, 0.0)
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    return results
