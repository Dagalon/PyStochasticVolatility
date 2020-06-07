import matplotlib.pyplot as plt
import numpy as np

from MC_Engines.MC_SABR import SABR_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Expansions.ImpliedVolATM import iv_atm_ln_hagan, iv_atm_second_order, iv_atm_variance_swap
from Tools import RNG, Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility, black_scholes_merton
from scipy.optimize import curve_fit

alpha = 0.4
nu = 0.2
rho = -0.4
parameters = [alpha, nu, rho]
F0 = 100
seed = 123456789
no_paths = 200000
T_maturity = 1.0
T_grid = np.linspace(1.0 / 365.0, T_maturity, 104)

k = 100.0
notional = 1.0
no_maturities = len(T_grid)

rnd_generator = RNG.RndGenerator(seed)

option_hagan = np.empty(no_maturities)
option_second_order = np.empty(no_maturities)
option_mc = np.empty(no_maturities)
option_variance_swap = np.empty(no_maturities)
std_mc = np.empty(no_maturities)
mc_iv = np.empty(no_maturities)

delta_time = 0.1

for i in range(0, no_maturities):
    # paths_sabr = SABR_Engine.get_path_one_step(0.0, T_grid[i], parameters, F0, no_paths, rnd_generator)
    no_time_steps_i = np.maximum(np.floor(T_grid[i] / delta_time) + 1, 2.0)
    map_sabr = SABR_Engine.get_path_multi_step(0.0, T_grid[i], parameters, F0, no_paths, int(no_time_steps_i),
                                               Types.TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY, rnd_generator)

    european_option = EuropeanOption(k, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, F0, T_grid[i])
    results = european_option.get_price(map_sabr[Types.SABR_OUTPUT.PATHS])
    option_mc[i] = results[0]
    std_mc[i] = results[1]

    second_order_iv = iv_atm_second_order(T_grid[i], parameters)
    hagan_iv = iv_atm_ln_hagan(T_grid[i], parameters)
    variance_swap_iv = iv_atm_variance_swap(T_grid[i], parameters)
    mc_iv[i] = implied_volatility(option_mc[i], F0, k, T_grid[i], 0.0, 0.0, 'c')

    option_hagan[i] = black_scholes_merton('c', F0, k, T_grid[i], 0.0, hagan_iv, 0.0)
    option_second_order[i] = black_scholes_merton('c', F0, k, T_grid[i], 0.0, variance_swap_iv, 0.0)

# plt.plot(T, option_mc, label='npv mc')
# plt.plot(T, option_hagan, label='npv hagan')
# plt.plot(T, option_variance_swap, label='npv variance swap')
# plt.plot(T, option_second_order, label='npv second order')

# plt.xlabel('T')
# plt.ylabel('option npv')
# plt.legend()plt.legend()
# plt.show()

atm_iv_variance_swap = [iv_atm_variance_swap(t_i, parameters) for t_i in T_grid]
atm_iv_second_order = [iv_atm_second_order(t_i, parameters) for t_i in T_grid]
atm_iv_hagan = [iv_atm_ln_hagan(t_i, parameters) for t_i in T_grid]

plt.plot(T_grid, atm_iv_second_order, label='iv second order')
plt.plot(T_grid, atm_iv_hagan, label='iv hagan')
plt.plot(T_grid, atm_iv_variance_swap, label='iv variance swap')


# --------------------------------------------------------------------------------------------------
def f_mc_fit(x, a, b, c):
    return c + b * x + a * x * x


popt, pcov = curve_fit(f_mc_fit, T_grid, mc_iv)
smooth_mc_iv = [f_mc_fit(ti, *popt) for ti in T_grid]
plt.plot(T_grid, smooth_mc_iv, label='iv mc')

plt.title("rho = " + str(rho) + " nu = " + str(nu) + " alpha = " + str(alpha))
plt.xlabel('T')
plt.ylabel('Implied Volatility')

plt.legend()
plt.show()
