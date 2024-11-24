import matplotlib.pylab as plt

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from MC_Engines.MC_SABR import SABR_Normal_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption, QuadraticEuropeanOption
from VolatilitySurface.Tools import SABRTools

# option info
f0 = 0.03
t = 10.0
spreads = [-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0,
           150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]

strikes = []
options = []
for si in spreads:
    strikes.append(si / 10000.0 + f0)
    options.append(QuadraticEuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# sabr parameters
alpha = 75.5 / 10000.0
nu = 0.23
rho = 0.235
parameters = [alpha, nu, rho]

# mc price
seed = 123456789
no_paths = 300000
rnd_generator = RNG.RndGenerator(seed)
no_time_steps = int(50 * t)

map_output = SABR_Normal_Engine.get_path_one_step(0.0, t, parameters, f0, no_paths, rnd_generator)

no_options = len(options)
price_watanabe = []
price_mc = []
price_hagan = []
price_watanabe_replciation = []


for i in range(0, no_options):
    mc_option_price = options[i].get_price(map_output[Types.SABR_OUTPUT.PATHS])
    mc_price = mc_option_price[0]
    watanabe = ExpansionTools.get_quadratic_option_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho)

    hagan = SABRTools.quadratic_european_normal_sabr(f0, strikes[i], alpha, rho, nu, t, 'c')

    price_mc.append(mc_price)
    price_hagan.append(hagan)
    price_watanabe.append(watanabe)

plt.plot(strikes, price_mc, label='mc price', linestyle='dashdot', color='k')
plt.scatter(strikes, price_watanabe, label='watanabe price', s=8, color='orange')
plt.scatter(strikes, price_hagan, label='hagan price', s=8, color='green')
# plt.scatter(strikes, price_hagan, label='hagan price', s=8, color='m')

plt.title("T=%s, F= %s" % (t, f0))

plt.legend()
plt.show()
