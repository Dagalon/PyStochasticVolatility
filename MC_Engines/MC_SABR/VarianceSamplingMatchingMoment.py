import numpy as np
import numba as nb

from scipy.special import ndtr
from Tools.Functionals import normal_pdf


def get_conditional_moment_order_one(alpha_t,
                                     alpha,
                                     nu,
                                     t):
    no_paths = len(alpha_t)
    mean_z = np.empty(no_paths)
    nu_t = nu * np.sqrt(t)
    mult = 0.5 * alpha * alpha * np.sqrt(t) / nu

    for i in range(0, no_paths):
        x_z = np.log(alpha_t[i] / alpha) / nu_t
        d_right = x_z + nu_t
        d_left = x_z - nu_t

        mean_z[i] = (ndtr(d_right) - ndtr(d_left)) / normal_pdf(0.0, 1.0, d_right)

    return mult * np.mean(mean_z)


def get_conditional_moment_order_two(alpha_t,
                                     alpha,
                                     nu,
                                     t):
    no_paths = len(alpha_t)
    moment_z = np.empty(no_paths)
    nu_t = nu * np.sqrt(t)
    mult = 0.25 * np.power(alpha, 4.0) * np.sqrt(t) / np.power(nu, 3.0)

    for i in range(0, no_paths):
        shift = (1.0 + np.power(alpha_t[i] / alpha, 2.0))
        x_z = np.log(alpha_t[i] / alpha) / nu_t
        d_right = x_z + nu_t
        d_left = x_z - nu_t
        term_1 = mult * shift * (ndtr(d_right) - ndtr(d_left)) / normal_pdf(0.0, 1.0, d_right)
        term_2 = mult * (ndtr(d_right + nu_t) - ndtr(d_left - nu_t)) / normal_pdf(0.0, 1.0, d_right + nu_t)
        moment_z[i] = - term_1 + term_2

    return np.mean(moment_z)


def get_variance(alpha,
                 nu,
                 alpha_t,
                 t,
                 z):
    no_paths = len(alpha_t)
    path = np.empty(no_paths)

    m_1 = get_conditional_moment_order_one(alpha_t, alpha, nu, t)
    m_2 = get_conditional_moment_order_two(alpha_t, alpha, nu, t)

    ln_mu = 2.0 * np.log(m_1) - 0.5 * np.log(m_2)
    ln_sigma = np.sqrt(np.log(m_2) - 2.0 * np.log(m_1))

    for i in range(0, no_paths):
        path[i] = np.exp(ln_mu + ln_sigma * z[i])

    return path
