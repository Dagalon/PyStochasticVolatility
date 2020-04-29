from Tools.Types import min_value, ndarray
import numpy as np


def derive_cev_drift(mu: float, t: float, x: ndarray):
    return mu * x


def derive_cev_sigma(sigma: float, rho: float, t: float, x: ndarray):
    x_epsilon = np.maximum(min_value, x)
    return sigma * rho * np.power(x_epsilon, rho - 1.0)


def cev_drift(mu: float, t: float, x: ndarray):
    return mu * x


def cev_sigma(sigma: float, rho: float, t: float, x: ndarray):
    return sigma * np.power(x, rho)


def z_drift(mu: float, rho: float, sigma: float, t: float, x: ndarray):
    return (1.0 - rho) * np.add(mu * x, np.divide(-0.5 * rho * sigma * sigma, x))


def z_sigma(sigma: float, rho: float, t: float, x: ndarray):
    return (1.0 - rho) * sigma * np.ones(len(x))


def bs_drift_flat(t: float, x: ndarray, rate_t: float, dividend_t: float,):
    return (rate_t - dividend_t) * x


def bs_sigma_flat(t: float, x: ndarray, sigma_t: float):
    return sigma_t * x



