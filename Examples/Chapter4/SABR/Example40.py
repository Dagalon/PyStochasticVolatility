from MC_Engines.MC_SABR import SABR_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from prettytable import PrettyTable
from tabulate import tabulate

import time

alpha = 0.4
nu = 1.1
rho = -0.8
parameters = [alpha, nu, rho]
f0 = 100
seed = 123456789
no_paths = 100000
T = 2.0

delta = 1.0 / 32.0
no_time_steps = int(T / delta)

strike = 150.0
notional = 1.0

european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

rnd_generator = RNG.RndGenerator(seed)

# Compute price, delta and gamma by MC and Malliavin in SABR model
start_time = time.time()
map_output = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                             rnd_generator)
end_time = time.time()
delta_time = (end_time - start_time)

result = european_option.get_price(map_output[Types.SABR_OUTPUT.PATHS])

price = result[0]
wide_ci = result[1]
malliavin_delta = european_option.get_malliavin_delta(map_output[Types.SABR_OUTPUT.PATHS],
                                                      map_output[Types.SABR_OUTPUT.DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])

malliavin_gamma = european_option.get_malliavin_gamma(map_output[Types.SABR_OUTPUT.PATHS],
                                                      map_output[Types.SABR_OUTPUT.GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])

# Compute delta and gamma by bumping in SABR model
delta_shift = 0.001
f0_right_shift = f0 * (1.0 + delta_shift)
rnd_generator.set_seed(seed)
map_output_right_shift = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0_right_shift, no_paths, no_time_steps,
                                                         Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                         rnd_generator)
result_right_shift = european_option.get_price(map_output_right_shift[Types.SABR_OUTPUT.PATHS])
price_right_shift = result_right_shift[0]
wide_ci_shift_right = result_right_shift[1]
sabr_delta_fd = (price_right_shift - price) / (f0 * delta_shift)

# Compute delta and gamma by bumping in SABR model
f0_left_shift = f0 * (1.0 - delta_shift)
rnd_generator.set_seed(seed)
map_output_left_shift = SABR_Engine.get_path_multi_step(0.0, T, parameters, f0_left_shift, no_paths, no_time_steps,
                                                        Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                        rnd_generator)

result_left_shift = european_option.get_price(map_output_left_shift[Types.SABR_OUTPUT.PATHS])
price_left_shift = result_left_shift[0]
wide_ci_shift_left = result_left_shift[1]
sabr_gamma_fd = (price_right_shift - 2 * price + price_left_shift) / (delta_shift * f0)**2

# Outputs
table = PrettyTable()
table.field_names = ["Numerical Method", "Price", "Delta", "Gamma"]
table.add_row(["MC and Malliavin", ['{0:.5g}'.format(result[0]), '{0:.5g}'.format(result[1])],
               ['{0:.5g}'.format(malliavin_delta[0]), '{0:.5g}'.format(malliavin_delta[1])],
               ['{0:.5g}'.format(malliavin_gamma[0]), '{0:.5g}'.format(malliavin_gamma[1])]])
table.add_row(["MC and Finite Differences", ['{0:.5g}'.format(result[0]), '{0:.5g}'.format(result[1])],
               sabr_delta_fd, sabr_gamma_fd])

print(tabulate(table, tablefmt="latex"))

