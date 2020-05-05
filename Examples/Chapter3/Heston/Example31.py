from tabulate import tabulate
from MC_Engines.MC_Heston import Heston_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from prettytable import PrettyTable


epsilon = 1.1
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05

f0 = 100
T = 2.0

seed = 123456789

delta = 1.0 / 32.0
no_time_steps = int(T / delta)
no_paths = 1000000
strike = 120.0

rnd_generator = RNG.RndGenerator(seed)

parameters = [k, theta, epsilon, rho]

notional = 1.0
european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
parameters_option_price = [0.0, theta, rho, k, epsilon, v0, 0.0]

# Compute price, delta and gamma by numerical integration in Heston model
analytic_output = european_option.get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                     model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                                     compute_greek=True)

# Compute price, delta and gamma by MC and Malliavin in Heston model
map_heston_output = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths,
                                                      no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                      rnd_generator)

result = european_option.get_price(map_heston_output[Types.HESTON_OUTPUT.PATHS])

malliavin_delta = european_option.get_malliavin_delta(map_heston_output[Types.HESTON_OUTPUT.PATHS],
                                                      map_heston_output[Types.HESTON_OUTPUT.DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])

malliavin_gamma = european_option.get_malliavin_gamma(map_heston_output[Types.HESTON_OUTPUT.PATHS],
                                                      map_heston_output[Types.HESTON_OUTPUT.GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])

# Compute delta and gamma by bumping in Heston model
delta_shift = 0.0001
f0_shift = f0 * (1.0 + delta_shift)
f0_shift_left = f0 * (1.0 - delta_shift)

rnd_generator.set_seed(seed)
map_heston_output_shift = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0_shift, v0,
                                                            no_paths, no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                            rnd_generator)

result_shift = european_option.get_price(map_heston_output_shift[Types.HESTON_OUTPUT.PATHS])
heston_delta_fd = (result_shift[0] - result[0]) / (delta_shift * f0)

rnd_generator.set_seed(seed)
map_heston_output_shift_left = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0_shift_left, v0,
                                                                 no_paths, no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                                 rnd_generator)

result_shift_left = european_option.get_price(map_heston_output_shift_left[Types.HESTON_OUTPUT.PATHS])
heston_gamma_fd = (result_shift[0] - 2 * result[0] + result_shift_left[0]) / (delta_shift * f0)**2

# Outputs
table = PrettyTable()
table.field_names = ["Numerical Method", "Price", "Delta", "Gamma"]
table.add_row(["Numerical Integration", '{0:.5g}'.format(analytic_output[0]), '{0:.5g}'.format(analytic_output[1][Types.TypeGreeks.DELTA]),
               '{0:.5g}'.format(analytic_output[1][Types.TypeGreeks.GAMMA])])
table.add_row(["MC and Malliavin", ['{0:.5g}'.format(result[0]), '{0:.5g}'.format(result[1])],
               ['{0:.5g}'.format(malliavin_delta[0]), '{0:.5g}'.format(malliavin_delta[1])],
               ['{0:.5g}'.format(malliavin_gamma[0]), '{0:.6g}'.format(malliavin_gamma[1])]])
table.add_row(["MC and Finite Differences", ['{0:.5g}'.format(result[0]), '{0:.5g}'.format(result[1])], '{0:.5g}'.format(heston_delta_fd), '{0:.5g}'.format(heston_gamma_fd)])

print(tabulate(table, tablefmt="latex"))




