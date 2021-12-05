import numpy as np
import numba as nb
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf
from Tools.Bachelier import binary_flag


def CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho):
    sqrt_t = np.sqrt(t)
    y = (k - f) / (alpha * sqrt_t)
    phi_y = normal_pdf(0.0, 1.0, y)
    rho_inv = np.sqrt(1.0 - rho * rho)

    g_y = phi_y - y * ndtr(-y)
    first_term = 0.5 * np.sqrt(t) * phi_y * (rho * nu * y)
    second_term = t * phi_y * (np.power(nu * rho, 2.0) * ((3.0 * np.power(y, 4.0) - 12.0 * np.power(y, 2.0) - 1.0) / 24.0) + np.power(nu * rho_inv, 2.0) * (4.0 * np.power(y, 2.0) - 6.0 / 24.0))

    price = alpha * sqrt_t * (g_y + first_term + second_term)
    return price


def EuropeanOptionPrice(f, k, t, alpha, nu, rho, flag, expansion_type):
    if expansion_type == "Watanabe":
        if binary_flag[flag] > 0:
            return CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho)
        else:
            return CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho) - (f - k)
    else:
        Exception("The expansion type " + expansion_type + " is unknown.")
