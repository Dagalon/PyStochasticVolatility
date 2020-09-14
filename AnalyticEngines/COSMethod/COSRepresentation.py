import numpy as np
import numba as nb
from typing import Callable
from Tools import Types
from AnalyticEngines.COSMethod import COSBlocksOptions
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


# @nb.jit("c16[:](f8,f8,i8,c16[:])", nopython=True, nogil=True, parallel=True)
def get_cos_coefficients_jit(a: float, b: float, no_terms: int, cf_k: Types.ndarray):
    a_k = np.zeros(no_terms, dtype=np.complex_)
    beta = np.pi / (b - a)
    for i in range(0, no_terms):
        a_k[i] = cf_k[i] * np.exp(-1j * i * a * beta)

    return a_k


def apply_adjustment_strike_cf(a_k: Types.ndarray, v_k: Types.ndarray, k_s: Types.ndarray, strikes: Types.ndarray, no_terms: int):
    no_strikes = len(strikes)
    prices = np.zeros(no_strikes)
    log_strikes = np.log(strikes)

    for i in range(0, no_strikes):
        prices[i] = 0.5 * (a_k[0] * np.exp(- 1j * k_s[0] * log_strikes[i])).real * v_k[0]
        for j in range(0, no_terms):
            prices[i] += (a_k[j] * np.exp(- 1j * k_s[j] * log_strikes[i])).real * v_k[j]

    return prices * strikes


def get_european_option_price(option_type: TypeEuropeanOption, a: float, b: float, no_terms: int,
                              strikes: Types.ndarray, cf: Callable[[Types.ndarray], Types.ndarray]):
    k_s = np.arange(0, no_terms, 1) * (np.pi / (b - a))
    cf_k = cf(k_s)
    a_k = get_cos_coefficients_jit(a, b, no_terms, cf_k)

    if option_type == TypeEuropeanOption.CALL:
        chi_k_s = COSBlocksOptions.call_put_block(a, b, 0.0, b, k_s)
        phi_k_s = COSBlocksOptions.digital_block(a, b, 0.0, b, k_s)
        v_k = (2.0 / (b - a)) * (chi_k_s - phi_k_s)
        return apply_adjustment_strike_cf(a_k, v_k, k_s, strikes, no_terms)

    else:
        chi_k_s = COSBlocksOptions.call_put_block(a, b, a, 0.0, k_s)
        phi_k_s = COSBlocksOptions.digital_block(a, b, a, 0.0, k_s)
        v_k = (2.0 / (b - a)) * (- chi_k_s + phi_k_s)
        return apply_adjustment_strike_cf(a_k, v_k, k_s, strikes, no_terms)





