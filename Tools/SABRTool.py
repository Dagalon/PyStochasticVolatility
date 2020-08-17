import numba as nb
import numpy as np


# We will suppose parameter beta=1, basically this parameter has he same effect in the skew that rho.
@nb.jit("f8(f8,f8,f8)", nopython=True, nogil=True)
def get_variance_swap(sigma0t: float, nu: float, t: float):
    sigma_t = (sigma0t / nu) * np.sqrt(np.exp(nu * nu * t) - 1.0)
    return sigma_t / np.sqrt(t)


@nb.jit("f8(f8,f8,f8)", nopython=True, nogil=True)
def get_rho_term_var_swap(alpha: float, nu: float, t: float):
    exp_nu_t = np.exp(nu * nu * t)
    result = np.power(alpha / nu, 3.0) * (2.0 + np.power(exp_nu_t, 3.0) - 3.0 * exp_nu_t)
    return result / 3.0


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def ln_hagan_vol(alpha, rho, v, z, t):
    epsilon = 1E-07
    v = np.minimum(v, 1000.0)

    if t == 0.0 or alpha == 0.0:
        return 0.0

    else:
        if v < epsilon:
            sigma = alpha
        else:
            if 0.0 <= z < epsilon:
                z = epsilon
            elif -epsilon < z < 0.0:
                z = -epsilon

            y = (v / alpha) * z
            y_prime = (np.sqrt(1 - 2 * rho * y + y ** 2) + y - rho) / (1 - rho)
            x = np.log(y_prime)

            order_0 = alpha * (y / x)
            order_1 = order_0 * ((0.25 * rho * v * alpha + (2 - 3 * np.power(rho, 2)) * (np.power(v, 2) / 24)) * t)
            sigma = order_0 + order_1

        return sigma

