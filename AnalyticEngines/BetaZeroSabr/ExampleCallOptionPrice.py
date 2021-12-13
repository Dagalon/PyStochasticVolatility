from AnalyticEngines.BetaZeroSabr.NormalSabrTools import CallOptionPrice
from MC_Engines.MC_SABR import SABR_Normal_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from AnalyticEngines.BetaZeroSabr import EuropeanOptionExpansion
from VolatilitySurface.Tools import SABRTools
from Tools.Bachelier import bachelier, implied_volatility

# option info
f0 = 0.01
strike = 0.01
t = 1.0

# sabr parameters
alpha = 0.3
nu = 0.7
rho = 0.0
parameters = [alpha, nu, rho]

# Antonov price
antonov_option_price = CallOptionPrice(t, f0, strike, alpha, nu, rho)

# watanabe expansion
watanabe_price = EuropeanOptionExpansion.EuropeanOptionPrice(f0, strike, t, alpha, nu, rho, 'c', "Watanabe")
iv_watanabe = implied_volatility(watanabe_price, f0, strike, t, 'c')
# Hagan
iv_hagan = SABRTools.sabr_normal_jit(f0, strike, alpha, rho, nu, t)
hagan_price = bachelier(f0, strike, t, iv_hagan, 'c')

# mc price
seed = 123456789
no_paths = 500000
rnd_generator = RNG.RndGenerator(seed)
no_time_steps = 100

option = EuropeanOption(strike, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t)

map_output = SABR_Normal_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
                                                    Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

mc_option_price = option.get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
mc_price = mc_option_price[0]
iv_mc = implied_volatility(mc_price, f0, strike, t, 'c')
