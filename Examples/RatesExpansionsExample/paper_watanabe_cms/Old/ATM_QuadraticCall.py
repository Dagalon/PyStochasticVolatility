import matplotlib.pylab as plt
import os
import pandas as pd
from pathlib import Path

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from MC_Engines.MC_SABR import SABR_Normal_Engine
from Instruments.EuropeanInstruments import QuadraticEuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types
from VolatilitySurface.Tools import SABRTools

current_directory = os.path.dirname(os.path.realpath(__file__))
folder_directory = Path(current_directory)
path_parameters = os.path.join(folder_directory, 'Data', 'sabr.csv')

parameters = pd.read_csv(path_parameters, header=None, names=["T", "alpha", "beta", "vol-of-vol", "rho"], sep=";")

n = len(parameters) - 1

# option info
f0 = 0.03

# mc price
seed = 123456789
no_paths = 250000

atm_hagan_price = []
atm_watanabe_price = []
atm_mc_price = []
atm_upper_bound_mc = []
atm_lower_boud_mc = []
maturities = []

for i in range(0, n):
    rnd_generator = RNG.RndGenerator(seed)
    p = [float(parameters.alpha[i + 1]), float(parameters['vol-of-vol'][i + 1]), float(parameters.rho[i + 1])]

    if 'M' in parameters['T'][i + 1]:
        ts = float(parameters['T'][i + 1][0:len(parameters['T'][i + 1]) - 1]) / 12.0
    else:
        ts = float(parameters['T'][i + 1][0:len(parameters['T'][i + 1]) - 1])

    maturities.append(ts)

    option = QuadraticEuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, ts)

    map_output = SABR_Normal_Engine.get_path_one_step(0.0, ts, p, f0, no_paths, rnd_generator)

    mc_option_price = option.get_price(map_output[Types.SABR_OUTPUT.PATHS])
    mc_price = mc_option_price[0]
    price_hagan = SABRTools.quadratic_european_normal_sabr(f0, f0, p[0], p[2], p[1], ts, 'c')

    watanabe_price_replication = ExpansionTools.get_quadratic_option_normal_sabr_watanabe_expansion_replication(f0, f0, ts, p[0], p[1], p[2])

    atm_hagan_price.append(price_hagan)
    atm_watanabe_price.append(watanabe_price_replication)
    atm_mc_price.append(mc_price)
    atm_upper_bound_mc.append(mc_price + mc_option_price[1])
    atm_lower_boud_mc.append(mc_price - mc_option_price[1])

plt.plot(maturities, atm_mc_price, label='mc price', linestyle='dashdot')
# plt.plot(maturities, atm_mc_price, label='mc price upper bound', linestyle='dashdot')
# plt.plot(maturities, atm_mc_price, label='mc price lower bound', linestyle='dashdot')

plt.scatter(maturities, atm_hagan_price, label='hagan price')
plt.scatter(maturities, atm_watanabe_price, label='watanabe price replication')

plt.ylim((0.0, 0.0012))

plt.legend()
plt.show()
