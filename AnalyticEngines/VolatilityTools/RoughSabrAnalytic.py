import numpy as np
import numba as nb


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def G0(y, rho):
    no_y = len(y)
    output = np.zeros(no_y)
    inv_rho = np.sqrt(1.0 - rho * rho)
    m = 2.0 * rho / inv_rho
    for i in range(0, no_y):
        part1 = np.log(1.0 + 2.0 * rho * y[i] + y[i] * y[i])
        part2 = m * (np.arctan(0.5 * m) - np.arctan(0.5 * m + y[i] / inv_rho))
        output[i] = part1 + part2

    return output


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def GOneHalf(y, rho):
    no_y = len(y)
    output = np.zeros(no_y)
    for i in range(0, no_y):
        alpha = np.sqrt(1.0 + rho * y[i] + 0.25 * y[i] * y[i])
        mLog = np.log((alpha - rho - 0.5 * y[i]) / (1.0 - rho))
        output[i] = 4.0 * mLog * mLog

    return output


@nb.jit("f8[:](f8[:], f8, f8)", nopython=True, nogil=True)
def Gh(y, rho, h):
    no_y = len(y)
    g0 = G0(y / (2.0 * h + 1.0), rho)
    gOneHalf = GOneHalf(2.0 * y / (2.0 * h + 1.0), rho)
    w1 = np.power(2.0 * h + 1.0, 2.0) * 3.0 * (1.0 - 2.0 * h) / (2.0 * h + 3.0)
    w2 = np.power(2.0 * h + 1.0, 2.0) * 2.0 * h / (2.0 * h + 3.0)
    output = np.zeros(no_y)
    for i in range(0, no_y):
        output[i] = w1 * g0[i] + w2 * gOneHalf[i]

    return output


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def KernelPowerLaw(tau, h, nu):
    return nu * np.sqrt(2.0 * h) * np.power(tau, h - 0.5)


# @nb.jit("f8[:](f8, f8[:], f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def RBergomiImpliedVol(f, strike, t, h, rho, nu, sigma_atm, vol_swap):
    no_ks = len(strike)
    k = np.log(strike / f)
    y_k = KernelPowerLaw(t, h, nu) * k / vol_swap
    g_a = Gh(y_k, rho, h)

    output = np.zeros(no_ks)

    for i in range(0, no_ks):
        if np.abs(k[i]) < 1e-08:
            output[i] = sigma_atm
        else:
            output[i] = sigma_atm * np.abs(y_k[i]) / np.sqrt(g_a[i])

    return output

