import numpy as np
import numba as nb

from functools import partial
from scipy import optimize

p_sub_calibrators = np.zeros(3)


@nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def var_raw_svi(z, a, b, rho, sigma, m):
    diff = (z - m)
    return a + b * (rho * diff + np.sqrt(diff * diff + sigma * sigma))


@nb.jit("f8(f8,f8,f8,f8)", nopython=True, nogil=True)
def var_reduced_svi(y, a, d, c):
    return a + d * y + c * np.sqrt(y * y + 1.0)


@nb.jit("f8[:](f8[:],f8,f8,f8,f8[:],f8[:], f8)", nopython=True, nogil=True)
def get_jac_reduced_svi(p, p_m_i, p_sigma_i, forward, strikes_t, sigma_t, dt):
    jacobian = np.zeros(3)
    no_elements = len(strikes_t)
    y_i = np.zeros(no_elements)
    multiplier = 1.0

    for k in range(0, no_elements):
        y_i[k] = (np.log(forward / strikes_t[k]) - p_m_i) / p_sigma_i
        var_model = var_reduced_svi(y_i[k], p[0], p[1], p[2])
        jacobian[0] += 2.0 * multiplier * (var_model - sigma_t[k] * sigma_t[k] * dt)
        jacobian[1] += 2.0 * multiplier * y_i[k] * (var_model - sigma_t[k] * sigma_t[k] * dt)
        jacobian[2] += 2.0 * multiplier * np.sqrt(y_i[k] * y_i[k] + 1.0) * (var_model - sigma_t[k] * sigma_t[k] * dt)

    return jacobian / no_elements


@nb.jit("f8(f8[:],f8,f8[:],f8[:])", nopython=True, nogil=True)
def get_raw_error_slice(p, forward, strikes_t, sigma_t):
    no_elements = len(strikes_t)
    error = 0.0

    for i in range(0, no_elements):
        z_i = np.log(forward / strikes_t[i])
        var_model = var_raw_svi(z_i, p[0], p[1], p[2], p[3], p[4])
        error += np.power(var_model - sigma_t[i] * sigma_t[i], 2.0)

    return error / no_elements


# @nb.jit("f8(f8[:],f8,f8,f8,f8[:],f8[:], f8)", nopython=True, nogil=True)
def get_reduced_error_slice(p, p_m_i, p_sigma_i, forward, strikes_t, sigma_t, dt):
    no_elements = len(strikes_t)
    error = 0.0
    multiplier = 1.0

    for i in range(0, no_elements):
        y_i = (np.log(strikes_t[i] / forward) - p_m_i) / p_sigma_i
        var_model = var_reduced_svi(y_i, p[0], p[1], p[2])
        error += multiplier * np.power(var_model - sigma_t[i] * sigma_t[i] * dt, 2.0)

    return error / no_elements


def get_constraints_reduced_svi(sigma_i):
    constraints = [{'type': 'ineq', 'fun': lambda p: p[2] - np.abs(p[1])},
                   {'type': 'ineq', 'fun': lambda p: (4.0 * sigma_i - p[2]) - np.abs(p[1])}]

    return constraints


def get_raw_parameters(a_r, d, c, sigma_i, dt):
    return np.array([a_r / dt, c/(sigma_i * dt), d/c])


def get_reduced_initial_guess(y_i_left, y_i, y_i_right, v_i_left, v_i, v_i_right):
    a = np.zeros(shape=(3, 3))
    b = np.zeros(3)
    b[0] = v_i
    b[1] = v_i_left
    b[2] = v_i_right
    a[0, 0] = 1.0
    a[1, 0] = 1.0
    a[2, 0] = 1.0
    a[0, 1] = y_i
    a[1, 1] = y_i_left
    a[2, 1] = y_i_right
    a[0, 2] = np.sqrt(y_i * y_i + 1.0)
    a[1, 2] = np.sqrt(y_i_left * y_i_left + 1.0)
    a[2, 2] = np.sqrt(y_i_right * y_i_right + 1.0)

    return np.linalg.solve(a, b)


def get_global_error(p_m_sigma, forward, strikes, sigmas, t):
    f_error = partial(get_reduced_error_slice, p_m_i=p_m_sigma[1], p_sigma_i=p_m_sigma[0], forward=forward,
                      strikes_t=strikes, sigma_t=sigmas, dt=t)

    f_jac_error = partial(get_jac_reduced_svi, p_m_i=p_m_sigma[1], p_sigma_i=p_m_sigma[0], forward=forward,
                          strikes_t=strikes, sigma_t=sigmas, dt=t)

    no_strikes = len(strikes)
    w_t = np.zeros(no_strikes)
    y_i = np.zeros(no_strikes)
    for i in range(0, no_strikes):
        w_t[i] = sigmas[i] * sigmas[i] * t
        z_i = np.log(strikes[i] / forward)
        y_i[i] = (z_i - p_m_sigma[1]) / p_m_sigma[0]

    max_w_t = np.max(w_t)
    const = get_constraints_reduced_svi(p_m_sigma[0])
    bounds = ((0.0, max_w_t), (0.0, None), (0.0, None))
    p0 = get_reduced_initial_guess(y_i[0], y_i[2], y_i[4], w_t[0], w_t[2], w_t[4])

    p_a_d_c = optimize.minimize(f_error, p0, method='SLSQP', bounds=bounds, constraints=const, jac=f_jac_error, tol=1e-12)

    p_a_b_rho = get_raw_parameters(p_a_d_c.x[0], p_a_d_c.x[1], p_a_d_c.x[2], p_m_sigma[0], t)

    # compute global error
    error = 0.0
    p_sub_calibrators[0] = p_a_b_rho[0]
    p_sub_calibrators[1] = p_a_b_rho[1]
    p_sub_calibrators[2] = p_a_b_rho[2]

    for i in range(0, no_strikes):
        z_i = np.log(strikes[i] / forward)
        var_model = var_raw_svi(z_i, p_a_b_rho[0], p_a_b_rho[1], p_a_b_rho[2], p_m_sigma[0], p_m_sigma[1])
        error += np.power(var_model - sigmas[i] * sigmas[i], 2.0)

    return error / no_strikes

