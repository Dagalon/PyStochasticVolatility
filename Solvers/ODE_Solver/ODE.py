import numpy as np

from Type import nd_array


def a_cev(rho: float, mu: float, sigma: float, z_t: float, t: float, x: float):
    return (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(x, -2.0))


def f_analytic_cev(rho: float, mu: float, sigma: float, z_t: float, t: nd_array):
    multiplier = (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(z_t, -2.0))
    return z_t * np.exp(multiplier * t)
