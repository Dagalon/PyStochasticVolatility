import numpy as np
import numba as nb
from scipy.special import ndtr
from functools import partial
from scipy.integrate import quad


@nb.jit("f8(f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_b_min(alpha, rho, nu, f0, k):
    diff = (k - f0)
    l_value = -rho * (diff * nu + alpha * rho) + np.sqrt(
        alpha * alpha + 2.0 * alpha * nu * rho * diff + nu * nu * diff * diff)

    return np.arccosh(l_value / (alpha * (1.0 - rho * rho)))


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def g(b, nu, t):
    t_transform = 0.5 * nu * nu * t
    sqrt_overt_2t = 1.0 / np.sqrt(2.0 * t_transform)
    e_value = 2.0 * ndtr((b - t_transform) * sqrt_overt_2t) - 1.0
    f_value = np.exp(b) * (2.0 * ndtr((b + t_transform) * sqrt_overt_2t) - 1.0)

    return - np.exp(-0.5 * b) * (-1.0 - np.exp(b) + e_value + f_value) / (8.0 * np.pi)


@nb.jit("f8(f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_n(x, alpha, rho, nu, f0, k):
    diff = f0 - k

    return np.power(-nu * diff - rho * x + rho * alpha, 2.0) + (1.0 - rho * rho) * np.power(x - alpha, 2.0)


@nb.jit("f8(f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def d(b, alpha, rho, nu, f0, k):
    n_value = get_n(b, alpha, rho, nu, f0, k)
    d_value = np.arccosh(1.0 + 0.5 * n_value / ((1.0 - rho * rho) * alpha * b))

    return d_value


@nb.jit("f8(f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_p(b, alpha, rho, nu, f0, k):
    diff = f0 - k

    return -2.0 * diff * nu * rho + 2.0 * alpha * rho * rho + 2.0 * alpha * np.cosh(b) \
           - 2.0 * alpha * rho * rho * np.cosh(b)


@nb.jit("f8(f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_q(b, alpha, rho, nu, f0, k):
    diff = f0 - k
    part_1 = 4.0 * (-alpha * alpha - np.power(diff * nu, 2.0) + 2.0 * diff * alpha * nu * rho)
    p_value = get_p(b, alpha, rho, nu, f0, k)
    q_value = part_1 + p_value * p_value

    return q_value


@nb.jit("f8(f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def h_integrator(x, b, alpha, rho, nu, f0, k):
    d_alpha = d(x, alpha, rho, nu, f0, k)
    x_value = np.cosh(b) - np.cosh(d_alpha)

    # if abs(d_alpha - b) < Tools.Types.min_value:
    #     return 1.0 / np.sqrt(Tools.Types.min_value)

    return 1.0 / np.sqrt(x_value)


def h(b, alpha, rho, nu, f0, k):
    p = get_p(b, alpha, rho, nu, f0, k)
    q = get_q(b, alpha, rho, nu, f0, k)
    a_min = 0.5 * (p - np.sqrt(q))
    a_max = 0.5 * (p + np.sqrt(q))

    if a_max > a_min:
        f = partial(h_integrator, b=b, alpha=alpha, rho=rho, nu=nu, f0=f0, k=k)
        integral_value = quad(f, a_min, a_max)
        return integral_value[0]
    else:
        return 0.0


def call_option_price(f0, k, t, alpha, rho, nu):
    intrinsic_value = np.maximum(f0 - k, 0.0)
    multiplier = (1.0 / nu) * np.sqrt(2.0 / (1.0 - rho * rho))
    b_min = get_b_min(alpha, rho, nu, f0, k)

    partial_h = partial(h, alpha=alpha, rho=rho, nu=nu, f0=f0, k=k)
    partial_g = partial(g, nu=nu, t=t)

    f = lambda b: partial_g(b) * partial_h(b)
    stochastic_value = quad(f, b_min, 5.0)

    return intrinsic_value + multiplier * stochastic_value[0]
