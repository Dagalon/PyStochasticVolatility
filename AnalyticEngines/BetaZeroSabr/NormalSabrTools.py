import numpy as np
import numba as nb

from scipy.integrate import quad_vec


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def IntegrandG(u, t, s):
    return u * np.exp(-0.5 * u * u / t) * np.sqrt(np.cosh(u) - np.cosh(s))


def G_Exactly(t, s):
    multiplier = 2.0 * np.sqrt(2.0) * np.exp(-0.125 * t) / (t * np.sqrt(2.0 * np.pi * t))
    value = quad_vec(lambda x: IntegrandG(x, t, s), s, 5.0)

    return multiplier * value[0]


@nb.jit("f8(f8)", nopython=True, nogil=True)
def g(s):
    return s * np.cosh(s) / np.sinh(s) - 1.0


# @nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def R(t, s):
    g_s = g(s)
    term1 = 1.0
    term2 = 3.0 * t * g_s / (8.0 * s * s)
    term3 = 5.0 * t * t * (-40.0 * s * s + 3.0 * g_s * g_s + 24.0 * g_s) / (128.0 * np.power(s, 4.0))
    term4 = 35.0 * np.power(t, 3.0) * (-40.0 * s * s + 3.0 * np.power(g_s, 3.0) + 24.0 * g_s * g_s + 120.0 * g_s) / (1024.0 * np.power(s, 6.0))
    return term1 + term2 - term3 + term4


# @nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def DeltaR(t, s):
    return np.exp(-0.125 * t) - (3072.0 + 384.0 * t + 24.0 * t * t + t * t * t) / 3072.0


# @nb.jit("f8(f8, f8)", nopython=True, nogil=True)
def G_Approximation(t, s):
    if s > 10.0:
        multiplier = np.power(s, -0.5) * np.exp(- 0.5 * s * s / t - 0.125 * t)
    else:
        multiplier = np.sqrt(np.sinh(s) / s) * np.exp(- 0.5 * s * s / t - 0.125 * t)

    return multiplier * np.exp(-0.5 * s * s / t) * (R(s, t) + DeltaR(t, s))


# @nb.jit("f8(f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def IntegratorOptionPrice(t, s, alpha, nu, rho, f0, strike):
    v0 = alpha / nu
    k = (strike - f0) / v0 + rho

    if s > 20.0:
        multiplier = 1.0 - np.power((0.5 * k * np.exp(-s) - rho), 2.0)
    else:
        multiplier = 1.0 - np.power((k - rho * np.cosh(s)) / np.sinh(s), 2.0)

    g_aux = G_Exactly(nu * nu * t, s)
    g_value = G_Approximation(nu * nu * t, s)
    value = g_value * np.sqrt(multiplier)
    return value


def CallOptionPrice(t, f0, strike, alpha, nu, rho):
    intrinsic_value = np.maximum(f0 - strike, 0.0)
    v0 = alpha / nu
    k = (strike - f0) / v0 + rho
    rho_inv_pow_2 = (1.0 - rho * rho)
    s0 = np.arccosh(-rho * k + np.sqrt(k * k + rho_inv_pow_2) / rho_inv_pow_2)
    integral = quad_vec(lambda s: IntegratorOptionPrice(t, s, alpha, nu, rho, f0, strike), s0, np.inf)
    return intrinsic_value + (v0 / np.pi) * integral[0]


def PutOptionPrice(t, strike, v0, f0, nu, rho):
    call_price = CallOptionPrice(t, strike, v0, f0, nu, rho)
    return call_price - (f0 - strike)
