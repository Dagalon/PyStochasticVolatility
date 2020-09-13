import numpy as np
import numba as nb
from Tools import Types


@nb.jit("f8[:](f8,f8,f8,f8,f8[:])", nopython=True, nogil=True, parallel=True)
def digital_block(a: float, b: float, c: float, d: float, k_s: Types.ndarray):
    alpha_d = (d - a) / (b - a)
    alpha_c = (c - a) / (b - a)
    no_k_s = len(k_s)
    output = np.zeros(no_k_s)
    for i in range(0, no_k_s):
        if k_s[i] == 0:
            output[i] = d - c
        else:
            output[i] = np.sin(k_s[i] * np.pi * alpha_d) - np.sin(k_s[i] * np.pi * alpha_c)

    return output


@nb.jit("f8[:](f8,f8,f8,f8,f8[:])", nopython=True, nogil=True, parallel=True)
def call_put_block(a: float, b: float, c: float, d: float, k_s: Types.ndarray):
    no_k_s = len(k_s)
    alpha_d = (d - a) / (b - a)
    alpha_c = (c - a) / (b - a)
    exp_d = np.exp(d)
    exp_c = np.exp(c)
    output = np.zeros(no_k_s)
    for i in range(0, no_k_s):
        beta = np.pi * k_s[i] / (b - a)
        a_k_1 = np.cos(k_s[i] * np.pi * alpha_d) * exp_d - np.cos(k_s[i] * np.pi * alpha_c) * exp_c
        a_k_2 = beta * (np.sin(k_s[i] * np.pi * alpha_d) * exp_d - np.sin(k_s[i] * np.pi * alpha_c) * exp_c)
        multiplier = 1.0 / (1 + beta * beta)
        output[i] = multiplier * (a_k_1 + a_k_2)

    return output
