import numpy as np

from Tools.Types import ndarray, min_value


def a_cev(rho: float, mu: float, sigma: float, z_t: float, t: float, x: float):
    x_epsilon = np.maximum(min_value, x)
    return (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(x_epsilon, -2.0))


def f_analytic_cev(rho: float, mu: float, sigma: float, z_t: float, t: ndarray):
    multiplier = (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(z_t, -2.0))
    return z_t * np.exp(multiplier * t)
