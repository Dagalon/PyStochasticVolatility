import matplotlib.pylab as plt

from MC_Engines.MC_SRoughVolatility import SRoughVolatility_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility


# simulation info
beta = 1.1
nu = 0.5
rho = -0.4
parameters = [nu, rho, beta]
no_time_steps = 100

seed = 123456789
no_paths = 10
T = 0.5

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 100.0
sigma_0 = 0.3
implied_vol = []
options_price = []
options = []
strikes = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
no_strikes = len(strikes)

for k_i in strikes:
    options.append(EuropeanOption(k_i, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))


map_output = SRoughVolatility_Engine.get_path_exp_multi_step(0.0, T, parameters, f0, sigma_0, no_paths, no_time_steps,
                                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                             rnd_generator)

for i in range(0, no_strikes):

    mc_option_price = options[i].get_price_control_variate(map_output[Types.SABR_OUTPUT.PATHS][:, -1],
                                                           map_output[Types.SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])

    options_price.append(mc_option_price)

    implied_vol.append(implied_volatility(mc_option_price[0], f0, strikes[i], T, 0.0, 0.0, 'c'))


plt.plot(strikes, implied_vol, label='implied vol', color='black', linestyle='--')

plt.xlabel('k')
plt.legend()
plt.show()
