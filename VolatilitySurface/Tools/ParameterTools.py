import numba as nb
import numpy as np


@nb.jit("f8[:](f8,f8,f8,f8,f8, f8, f8)",nopython=True, nogil=True)
def get_modify_parameters(a, b, rho, sigma, m, spot, delta_t_T):
    b = b / delta_t_T
    m = m + np.log(spot)
    a = a / delta_t_T

    y = a
    x = m

    slope_l = b * (rho - 1.0)
    slope_r = b * (1.0 + rho)

    return np.array([x, y, slope_l, slope_r, sigma])


@nb.jit("f8[:](f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_base_parameter_from_term_struct(v_t, b_t, rho_t, x_t, lambda_t):
    b = b_t
    rho = rho_t
    sigma = lambda_t * np.sqrt(1.0 - rho * rho)
    m = x_t + rho_t * lambda_t
    a = v_t - b * sigma * np.sqrt(1.0 - rho * rho)

    return np.array([a, b, rho, sigma, m])


@nb.jit("f8[:](f8,f8,f8,f8,f8, f8, f8)", nopython=True, nogil=True)
def get_modify_parameters_from_desk_param(x, y, slope_left, slope_right, sigma, spot, delta_t_T):
    a = y * delta_t_T
    m = x - np.log(spot)
    rho = (slope_left - slope_right) / (slope_left + slope_right)
    b = 0.5 * delta_t_T * (slope_right - slope_left)

    return np.array([a, b, rho, sigma, m])


@nb.jit("f8(f8,f8,f8,f8)", nopython=True, nogil=True)
def alpha_atm_sabr(rho, v, sigma_atm, t):
    a = 1 + t * np.power(v, 2) * (2.0 - 3.0 * np.power(rho, 2)) / 24
    multiplier = t * v * rho
    val = (-a + np.sqrt(np.power(a, 2) + sigma_atm * multiplier)) / (0.5 * multiplier)
    return val


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def nu_sabr(p, t):
    # delta_t_bound = 2.0 / 52.0
    # if t < delta_t_bound:
    #     return p[0] + (p[1] + p[2] * delta_t_bound) * np.power(delta_t_bound, - p[3])
    #else:
    return p[0] + (p[1] + p[2] * t) * np.power(t, - p[3])


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def rho_sabr(p, t):
    return p[0] + (p[1] + p[2] * t) * np.exp(- p[3] * t)


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def nu_sabr_partial_t(p, t):
    # delta_t_bound = 2.0 / 52.0
    # if t < delta_t_bound:
    #     return 0.0
    # else:
    pow_t = np.power(t, - p[3])
    return p[2] * pow_t - p[3] * (p[1] + p[2] * t) * (pow_t / t)


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def rho_sabr_partial_t(p, t):
    return (p[2] - p[3] * (p[1] + p[2] * t)) * np.exp(- p[3] * t)
