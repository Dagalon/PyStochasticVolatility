import numpy as np
import time
import matplotlib.pyplot as plt

from MC_Engines.MC_SABR import SABR_Engine
from Tools import RNG
from MC_Pricers.EuropeanPricers import call_operator
from VolatilitySurface.Tools.SABRTools import sabr_vol_jit
from py_vollib.black_scholes import black_scholes
from Tools.AnalyticTools import third_derive_bs_log_spot
from AnalyticEngines.MalliavinMethod.ExpansionTools import get_vol_swap_approximation_sabr


# model parameters
alpha = 0.3
nu = 0.5
rho = -0.6
parameters = [alpha, nu, rho]
f0 = 100.0
strike = 100.0
ts_i= np.arange(0.1, 2.1, 0.1)


# simulation info
no_paths = 1000000
seed = 12345

mc_call_option = []
hagan_call_option = []
malliavin_call_option = []

for t in ts_i:
    rnd_generator = RNG.RndGenerator(seed)

    start_time = time.time()
    output = SABR_Engine.get_path_one_step(0.0, t, parameters, f0, no_paths, rnd_generator)
    end_time = time.time()
    diff = (end_time - start_time)
    print(f"Time {t}: {diff}")

    # Call option price
    call_option_price, error, probability = call_operator(output, strike)
    mc_call_option.append(call_option_price)

    # Hagan implied vol approximation
    z = np.log(f0/strike)
    hagan_sigma = sabr_vol_jit(alpha, rho, nu, z, t)
    hagan_call_price = black_scholes('c', f0, strike, t, 0.0, hagan_sigma)
    hagan_call_option.append(hagan_call_price)

    # Malliavin approximation call option
    volatility_swap_approximation = get_vol_swap_approximation_sabr(np.array(parameters), 0.0, t, alpha)
    call_rho_zero = black_scholes('c', f0, strike, t, 0.0, volatility_swap_approximation)
    adjustment_integral = np.power(alpha / nu, 3.0) * np.exp(nu * nu * t) * (3.0 * np.exp(2.0 * nu * nu * t) - 2.0 * np.exp(3.0 * nu * nu * t) - 1.0) / 3.0
    adjustment_rho = 0.5 * rho * adjustment_integral *  third_derive_bs_log_spot(0, 0, t, volatility_swap_approximation, f0)


    malliavin_call_option.append(call_rho_zero + adjustment_rho)

plt.plot(ts_i, mc_call_option, label="MC", linestyle='-.')
plt.plot(ts_i, hagan_call_option, label="Hagan", linestyle='-.')
plt.plot(ts_i, malliavin_call_option, label="Malliavin",linestyle='-.')

plt.legend()
plt.show()