import matplotlib.pylab as plt

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from MC_Engines.MC_SABR import SABR_Normal_Engine
from Instruments.EuropeanInstruments import QuadraticEuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from VolatilitySurface.Tools import SABRTools

# option info
f0 = 0.03
t = 15.0
spreads = [-400.0, -300.0, -200.0, -100.0, -75.0, -50.0, -25.0, -10.0, -1.0, 0.0, 1.0, 10.0, 25.0, 50.0, 75.0,
           100.0, 200.0, 300.0, 400.0, 600.0]


strikes = []
options = []

for si in spreads:
    strikes.append(si / 10000.0 + f0)
    options.append(QuadraticEuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# sabr parameters
alpha = 61.5/10000.0
nu = 0.21
rho = 0.16

parameters = [alpha, nu, rho]

# mc price
seed = 123456789
no_paths = 250000
rnd_generator = RNG.RndGenerator(seed)
no_time_steps = int(50 * t)

# map_output = SABR_Normal_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
#                                                      Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC, rnd_generator)

map_output = SABR_Normal_Engine.get_path_one_step(0.0, t, parameters, f0, no_paths, rnd_generator)

no_options = len(options)
price_watanabe = []
price_watanabe_replication = []
price_mc = []
quadratic_price_hagan = []

for i in range(0, no_options):
    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS])
    mc_price = mc_option_price[0]
    price_hagan = SABRTools.quadratic_european_normal_sabr(f0, strikes[i], alpha, rho, nu, t, 'c')
    watanabe_price = ExpansionTools.get_quadratic_option_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho)
    watanabe_price_replication = ExpansionTools.get_quadratic_option_normal_sabr_watanabe_expansion_replication(f0, strikes[i], t, alpha, nu, rho)

    quadratic_price_hagan.append(price_hagan)
    price_mc.append(mc_price)
    price_watanabe.append(watanabe_price)
    price_watanabe_replication.append(watanabe_price_replication)

plt.plot(strikes, price_mc, label='mc price', linestyle='dashdot')
# plt.plot(strikes, price_watanabe, label='watanabe price', linestyle='dashed')
plt.scatter(strikes, quadratic_price_hagan, label='hagan price')
plt.scatter(strikes, price_watanabe_replication, label='watanabe price replication')

plt.title("T=%s" % t)

plt.legend()
plt.show()
