import matplotlib.pylab as plt

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from MC_Engines.MC_SABR import SABR_Normal_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from VolatilitySurface.Tools import SABRTools
from Tools.Bachelier import bachelier, implied_volatility

# option info
f0 = 0.02
t = 1.0
spreads = [-200.0, -150.0, -100.0, -50.0, -25.0, -0.1, 0.0, 0.1, 25.0, 50.0, 100.0, 150.0, 200.0]

strikes = []
options = []
for si in spreads:
    strikes.append(si / 10000.0 + f0)
    options.append(EuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# sabr parameters
alpha = 0.007
nu = 0.7
rho = -0.6
parameters = [alpha, nu, rho]

# mc price
seed = 123456789
no_paths = 500000
rnd_generator = RNG.RndGenerator(seed)
no_time_steps = 200

map_output = SABR_Normal_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
                                                    Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

no_options = len(options)
iv_hagan = []
iv_watanabe = []
iv_mc = []

for i in range(0, no_options):
    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS][:, -1])
    mc_price = mc_option_price[0]
    iv_mc.append(implied_volatility(mc_price, f0, strikes[i], t, 'c'))
    iv_hagan.append(SABRTools.sabr_normal_jit(f0, strikes[i], alpha, rho, nu, t))
    iv_watanabe.append(ExpansionTools.get_iv_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho))

plt.plot(strikes, iv_hagan, label='iv hagan', linestyle='dotted')
plt.plot(strikes, iv_watanabe, label='iv watanabe', linestyle='dashed')
plt.plot(strikes, iv_mc, label='iv mc')

plt.title("T=%s, F= %s" % (t, f0))

plt.legend()
plt.show()



