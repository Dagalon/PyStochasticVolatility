import matplotlib.pylab as plt

from Tools import RNG, Types
from VolatilitySurface.Tools import SABRTools
from Instruments.EuropeanInstruments import TypeSellBuy, TypeEuropeanOption
from AnalyticEngines.MalliavinMethod import EuropeanOptionExpansion
from py_vollib.black_scholes_merton import black_scholes_merton

# option datas (We will suppose that r=0)
dt = [1.0 / 52, 1.0 / 12.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
f0 = 100.0

# simulation info
parameters = [0.3, -0.5, 0.5]
seed = 123456789
no_paths = 100000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# ouputs container
mc_option_price = []
var_swap_apprx_price = []
hagan_apprx_price = []

no_dt = len(dt)

for i in range(0, no_dt):
    no_time_steps = int(dt[i] / delta_time)
    rnd_generator.set_seed(seed)
    # european_option = EuropeanOption(f0, 1, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, dt[i])
    # map_output = SABR_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, no_paths, no_time_steps,
    #                                              Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
    #                                              rnd_generator)
    #
    # results = european_option.get_price(map_output[Types.SABR_OUTPUT.PATHS])
    # mc_option_price.append(results[0])

    # price the option with var swap approximation
    analytic_price = EuropeanOptionExpansion.get_var_swap_apprx_price(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL,
                                                                      f0, dt[i], parameters, Types.TypeModel.SABR)
    var_swap_apprx_price.append(analytic_price)

    # hagan's price
    iv_hagan = SABRTools.ln_hagan_vol(parameters[0], parameters[1], parameters[2], 0.0, dt[i])
    hagan_price = black_scholes_merton('c', f0, f0, dt[i], 0.0, iv_hagan, 0.0)
    hagan_apprx_price.append(hagan_price)

plt.plot(dt, mc_option_price, label='mc price')
plt.plot(dt, var_swap_apprx_price, label='variance swap approximation')
plt.plot(dt, hagan_apprx_price, label='Hagan approximation')

plt.legend()
plt.title('ATM option price')
plt.show()