import numpy as np
import numba as nb
from typing import Callable
from Tools import Types
from AnalyticEngines.COSMethod import CosBlocksOptions
from Instruments.EuropeanInstruments import TypeEuropeanOption
from Tools import Functionals


@nb.jit("f8[:](f8,f8,i8,i8,c16[:],f8[:])", nopython=True, nogil=True, parallel=True)
def get_cos_estimation_density_jit(a: float, b: float, no_terms: int, no_x: int, cf_k: Types.ndarray, x: Types.ndarray):
    output = np.zeros(no_x)
    f_k = np.zeros(no_terms)
    beta = np.pi / (b - a)
    alpha = 2.0 / (b - a)

    for i in range(0, no_terms):
        f_k[i] = alpha * (cf_k[i] * np.exp(-1j * i * a * beta)).real

    for i in range(0, no_x):
        b_k = 1.0
        output[i] = 0.5 * f_k[0] * b_k
        for j in range(1, no_terms):
            b_k = np.cos(j * beta * (x[i] - a))
            output[i] += f_k[j] * b_k

    return output


def get_cos_density(a: float, b: float, no_terms: int, cf: Callable[[Types.ndarray], Types.ndarray], x: Types.ndarray):
    no_x = len(x)
    cf_k = cf((np.pi / (b - a)) * np.arange(0, no_terms, 1))
    return get_cos_estimation_density_jit(a, b, no_terms, no_x, cf_k, x)


@nb.jit("f8[:](f8,f8,i8,c16[:])", nopython=True, nogil=True, parallel=True)
def get_cos_coefficients_jit(a: float, b: float, no_terms: int, cf_k: Types.ndarray):
    a_k = np.zeros(no_terms)
    beta = np.pi / (b - a)
    for i in range(0, no_terms):
        a_k[i] = (cf_k[i] * np.exp(-1j * i * a * beta)).real

    return a_k


def get_european_option_price(option_type: TypeEuropeanOption, a: float, b: float, no_terms: int, f0: float, strikes: float, t: float,  cf: Callable[[Types.ndarray], Types.ndarray]):
    k_s = np.arange(0, no_terms, 1)
    cf_k = cf((np.pi / (b - a)) * k_s)
    a_k = get_cos_coefficients_jit(a, b, no_terms, cf_k)
    x_s = np.log(f0 / strikes)

    if option_type == TypeEuropeanOption.CALL:
        chi_k_s = CosBlocksOptions.call_put_block(a, b, 0.0, b, k_s)
        phi_k_s = CosBlocksOptions.digital_block(a, b, 0.0, b, k_s)
        v_k = (2.0 / (b - a)) * (chi_k_s - phi_k_s) 
        return 0.5 * v_k[0] * a_k[0] + Functionals.scalar_product(v_k[1:], a_k[1:])
    else:
        chi_k_s = CosBlocksOptions.call_put_block(a, b, a, 0.0, k_s)
        phi_k_s = CosBlocksOptions.digital_block(a, b, a, 0.0, k_s)
        v_k = (2.0 / (b - a)) * (- chi_k_s + phi_k_s)
        return 0.5 * v_k[0] * a_k[0] + Functionals.scalar_product(v_k[1:], a_k[1:])





