import numpy as np

from time import time
from MC_Engines.MC_RBergomi import ToolsVariance

hurst_parameter = 0.3

t0 = 0.0
t1 = 2.0
n = 5
no_paths = 100000
h = 0.3
rho = 0.5

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


