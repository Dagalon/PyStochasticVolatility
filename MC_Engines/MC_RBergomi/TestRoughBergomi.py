import numpy as np

from time import time
from MC_Engines.MC_RBergomi import ToolsVariance, RBergomi_Engine
from Tools import RNG
from Tools.Types import TYPE_STANDARD_NORMAL_SAMPLING

hurst_parameter = 0.3

t0 = 0.0
t1 = 2.0
no_time_steps = 100
no_paths = 50000
seed = 123456789
rng_generator = RNG.RndGenerator(seed)

nu = 0.5
h = 0.3
rho = 0.00001

f0 = 100.0
v0 = 0.09

t_i_s = np.linspace(1.0 / 100.0, t1, 1000)

start_time = time()
cov = ToolsVariance.get_covariance_matrix(t_i_s, hurst_parameter, rho)
end_time = time()
diff = end_time - start_time

start_time = time()
a = np.linalg.cholesky(cov)
end_time = time()
diff = end_time - start_time
print(diff)

start_time = time()
paths = RBergomi_Engine.get_path_multi_step(t0,
                                            t1,
                                            [nu, rho, h],
                                            f0,
                                            v0,
                                            no_paths,
                                            no_time_steps,
                                            TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                            rng_generator)

end_time = time()
diff = end_time - start_time
print(diff)



