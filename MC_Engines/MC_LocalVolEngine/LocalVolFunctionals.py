import numba as nb
import numpy as np

from Tools import Types


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * np.power(x[i], beta)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def first_derive_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * beta * np.power(x[i], beta - 1.0)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def second_derive_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * beta * (beta - 1.0) * np.power(x[i], beta - 2.0)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def log_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * np.power(np.exp(np.maximum(x[i], 0.00001)), beta)

    return output
