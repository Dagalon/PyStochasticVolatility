import time
from MC_Engines.MC_LocalVolEngine import LocalVolEngine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import RNG, Types

f0 = 100
seed = 123456789
no_paths = 100000
T = 2.0

# CEV parameter
beta = 0.5


delta = 1.0 / 32.0
no_time_steps = int(T / delta)

strike = 150.0
notional = 1.0

european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

rnd_generator = RNG.RndGenerator(seed)

# Compute the price of the option by MC
start_time = time.time()
map_output = LocalVolEngine.get_path_multi_step(0.0, T, f0, no_paths, no_time_steps,
                                             Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                             rnd_generator)
end_time = time.time()
delta_time = (end_time - start_time)

