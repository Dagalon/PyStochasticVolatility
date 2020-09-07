import numpy as np
import numba as nb
from typing import List
from Tools import Types, Functionals, HestonTool
from VolatilitySurface.Tools import SABRTools
from Instruments.EuropeanInstruments import TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton import black_scholes_merton


@nb.jit("f8(f8,f8,f8,f8)", nopython=True, nogil=True)
def delta_vega(k: float, spot: float, sigma: float, t: float):
    x = np.log(spot)
    k_lg = np.log(k)
    sqrt_t = np.sqrt(t)
    d_1 = (x - k_lg) / (sigma * sqrt_t) + 0.5 * sigma * sqrt_t
    partial_sigma_d_1 = - (x - k_lg) / (sigma * sigma * sqrt_t) + 0.5 * sqrt_t
    return spot * Functionals.normal_pdf(0.0, 1.0, d_1) * partial_sigma_d_1 / (sigma * t)


def get_var_swap_apprx_price(strike: float,
                             notional: float,
                             buy_sell: TypeSellBuy,
                             option_type: TypeEuropeanOption,
                             spot: float,
                             delta_time: float,
                             parameters: List[float],
                             model: Types.TypeModel):

    if model == Types.TypeModel.SABR:
        alpha = parameters[0]
        rho = parameters[1]
        nu = parameters[2]
        var_swap = SABRTools.get_variance_swap(alpha, nu, delta_time)
        if option_type == TypeEuropeanOption.CALL:
            bs0 = black_scholes_merton('c', spot, strike, delta_time, 0.0, var_swap, 0.0)
            h_0 = delta_vega(strike, spot, var_swap, delta_time)
            rho_term = SABRTools.get_rho_term_var_swap(alpha, nu, delta_time) * h_0
            if buy_sell == TypeSellBuy.BUY:
                return notional * (bs0 + 0.5 * rho * rho_term)
            else:
                return - notional * (bs0 + 0.5 * rho * rho_term)
        else:
            bs0 = black_scholes_merton('p', spot, strike, delta_time, 0.0, var_swap, 0.0)
            h_0 = delta_vega(strike, spot, var_swap, delta_time)
            rho_term = SABRTools.get_rho_term_var_swap(alpha, nu, delta_time) * h_0
            forward = (spot - strike)
            call_price = notional * (bs0 + 0.5 * rho * rho_term)
            if buy_sell == TypeSellBuy.BUY:
                return call_price - forward
            else:
                return forward - call_price

    elif model == Types.TypeModel.HESTON:
        k = parameters[0]
        theta = parameters[1]
        epsilon = parameters[2]
        rho = parameters[3]
        v0 = parameters[4]

        var_swap = HestonTool.get_variance_swap(v0, k, theta, delta_time)
        if option_type == TypeEuropeanOption.CALL:
            bs0 = black_scholes_merton('c', spot, strike, delta_time, 0.0, var_swap, 0.0)
            h_0 = delta_vega(strike, spot, var_swap, delta_time)
            rho_term = HestonTool.get_rho_term_var_swap(v0, k, theta, epsilon, delta_time) * h_0
            if buy_sell == TypeSellBuy.BUY:
                return notional * (bs0 + 0.5 * rho * rho_term)
            else:
                return - notional * (bs0 + 0.5 * rho * rho_term)
        else:
            bs0 = black_scholes_merton('p', spot, strike, delta_time, 0.0, var_swap, 0.0)
            h_0 = delta_vega(strike, spot, var_swap, delta_time)
            rho_term = HestonTool.get_rho_term_var_swap(v0, k, theta, delta_time) * h_0
            forward = (spot - strike)
            call_price = notional * (bs0 + 0.5 * rho * rho_term)
            if buy_sell == TypeSellBuy.BUY:
                return call_price - forward
            else:
                return forward - call_price

    elif model == Types.TypeModel.ROUGH_BERGOMI:
        return 0.0
    else:
        print(f'At the moment we have not implmented the approximation price for {str(model)} ')
