import matplotlib.pylab as plt
from MC_Engines.MC_LocalVol import LocalVolFunctionals, LocalVolEngine
from Tools import RNG, Types
from functools import partial
from VolatilitySurface.Tools import SABRTools
from AnalyticEngines.BetaZeroSabr import ExpansionTools
from Instruments.EuropeanInstruments import QuadraticEuropeanOption, TypeSellBuy, TypeEuropeanOption
import numpy as np


# option info
f0 = 0.03
t = 10.0

# spreads = [-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0,
#            150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]

# spreads = [700]

spreads = [-200, -150, -100.0, -50.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450, 500, 600, 700, 800]

# spreads = [250]

strikes = []
options = []


for si in spreads:
    strikes.append(si / 10000.0)
    # options.append(QuadraticEuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))
    options.append(QuadraticEuropeanOption(strikes[-1], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t))

# sabr parameters
alpha = 75.5/10000.0
nu = 0.24
rho = 0.235
parameters = [alpha, nu, rho]

# local vol info
sabr_loc_vol = partial(LocalVolFunctionals.local_vol_normal_sabr, x0=f0, alpha=alpha, rho=rho, nu=nu)

# mc price
seed = 123456789
no_paths = 300000
generator = RNG.RndGenerator(seed)
no_time_steps = int(50 * t)

mc_price = []
call_watanabe_price = []
price_hagan = []

diff_hagan_mc = []
diff_watanabe_mc = []
limit = [1 for k in strikes]


map_output = LocalVolEngine.get_bachelier_path_multi_step(0.0, t, f0, no_paths, no_time_steps, local_vol=sabr_loc_vol,
                                                          type_random_number= Types.TYPE_STANDARD_NORMAL_SAMPLING, rnd_generator=generator)

for i in range(0, len(strikes)):
    # mc price
    mc_option_price = options[i].get_price(map_output[Types.LOCAL_VOL_OUTPUT.PATHS])
    mc_price.append(mc_option_price[0])
    std_error = mc_option_price[1]

    # hagan price
    hagan = SABRTools.quadratic_european_normal_sabr(f0, strikes[i], alpha, rho, nu, t, 'c')
    price_hagan.append(hagan)

    diff_hagan_mc.append(100.0 * np.abs(hagan - mc_option_price[0])/mc_option_price[0])

    # watanabe price
    watanabe_price = ExpansionTools.get_quadratic_option_lv_normal_sabr_watanabe_expansion(f0, strikes[i], t, alpha, nu, rho)
    call_watanabe_price.append(watanabe_price)
    diff_watanabe_mc.append(100.0 * np.abs(watanabe_price - mc_option_price[0])/mc_option_price[0])


plt.plot(strikes, mc_price, label='mc price', linestyle='dashdot', color='k')
# plt.plot(strikes, diff_watanabe_mc, label='%abs(watanabe price - mc price) / mc price', linestyle='dashdot', color='c')
# plt.plot(strikes, diff_hagan_mc, label='%abs(hagan price - mc price) / mc price', linestyle='dashdot', color='y')
plt.scatter(strikes, call_watanabe_price, label='watanabe price', s=5, color='c')
plt.scatter(strikes, price_hagan, label='Hagan price', s=5, color='red')


plt.title("T=%s, F= %s" % (t, f0))
plt.legend()
plt.show()



