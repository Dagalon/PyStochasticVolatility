import matplotlib.pylab as plt

from MC_Engines.MC_Heston import Heston_Engine
from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from AnalyticEngines.MalliavinMethod import EuropeanOptionExpansion

# option datas (We will suppose that r=0)
dt = [1.0 / 52, 1.0 / 12.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
f0 = 100.0

# simulation info
epsilon = 0.5
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.06

parameters = [k, theta, epsilon, rho, v0]

seed = 123456789
no_paths = 100000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# ouputs container
mc_option_price = []
var_swap_apprx_price = []

no_dt = len(dt)

for i in range(0, no_dt):
    no_time_steps = int(dt[i] / delta_time)
    rnd_generator.set_seed(seed)
    european_option = EuropeanOption(f0, 1, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, dt[i])

    map_output = Heston_Engine.get_path_multi_step(0.0, dt[i], parameters, f0, v0, no_paths,
                                                   no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                   rnd_generator)

    characteristic_function_price = european_option.get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                       model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                                                       compute_greek=False)

    results = european_option.get_price(map_output[Types.HESTON_OUTPUT.PATHS])
    mc_option_price.append(results[0])

    # price the option with var swap approximation
    analytic_price = EuropeanOptionExpansion.get_var_swap_apprx_price(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL,
                                                                      f0, dt[i], parameters, Types.TypeModel.HESTON)
    var_swap_apprx_price.append(analytic_price)


plt.plot(dt, mc_option_price, label='mc price')
plt.plot(dt, var_swap_apprx_price, label='variance swap approximation')

plt.legend()
plt.title('ATM option price')
plt.show()