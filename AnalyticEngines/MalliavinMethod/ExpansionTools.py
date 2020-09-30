import numba as nb
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
