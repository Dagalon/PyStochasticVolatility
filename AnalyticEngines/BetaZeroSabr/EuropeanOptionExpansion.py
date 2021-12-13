import numpy as np
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf
from Tools.Bachelier import binary_flag


def CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho):
    sqrt_t = np.sqrt(t)
    y = (k - f) / (alpha * sqrt_t)
    phi_y = normal_pdf(0.0, 1.0, y)
    rho_inv = np.sqrt(1.0 - rho * rho)

    g_y = phi_y - y * ndtr(-y)
    g1 = 0.5 * np.sqrt(t) * phi_y * (rho * nu * y)
    g21 = 0.5 * t * phi_y * np.power(nu * rho, 2.0) * (np.power(rho, 2.0) * (y - 3.0) + 2.0) - 0.25 * phi_y * nu * nu * t
    g22 = 0.5 * phi_y * t * (np.power(nu * rho_inv, 2.0) * (y * y / 3.0 + 1.0 / 6.0) +
                             np.power(nu * rho, 2.0) * (1.5 * np.power(rho_inv * y, 2.0) + 0.25 * np.power(y, 4.0))) + \
        0.375 * nu * nu * phi_y * np.power(rho_inv, 4.0) * t

    price = alpha * sqrt_t * (g_y + g1 + g21 + g22)
    return price


def EuropeanOptionPrice(f, k, t, alpha, nu, rho, flag, expansion_type):
    if expansion_type == "Watanabe":
        if binary_flag[flag] > 0:
            return CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho)
        else:
            return CallOptionWatanabeExpansion(f, k, t, alpha, nu, rho) - (f - k)
    else:
        Exception("The expansion type " + expansion_type + " is unknown.")
