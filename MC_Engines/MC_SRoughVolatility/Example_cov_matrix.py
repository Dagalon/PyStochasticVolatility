import matplotlib.pyplot as plt
import numpy as np
from MC_Engines.MC_SRoughVolatility import ToolsVariance

no_points = 10
beta = 1.4
T = np.exp(-beta - 1)
t = np.linspace(0.0, T, no_points)

variance = ToolsVariance.get_variance(t, beta)

cov = np.zeros(shape=(no_points, no_points))


# plot kernel
# s = 0.01
# t = 0.03
# u_i = np.linspace(0, 0.9999, 100)
# k_i = []
# for u in u_i:
#     k_i.append(ToolsVariance.get_kernel(u, s, t, beta))
#
# plt.plot(u_i, k_i)
# plt.show()

# var_from_cov = []
# for i in range(0, no_points):
#     var_from_cov.append(ToolsVariance.get_volterra_covariance(t[i], t[i], beta))
#     error = variance[i] - var_from_cov[-1]
#
# plt.plot(t, variance)
# plt.plot(t, var_from_cov)
# plt.show()


for i in range(0, no_points):
    for j in range(0, i + 1):
        cov[i, j] = ToolsVariance.get_volterra_covariance(t[i], t[j], 1.4)
        cov[j, i] = cov[i, j]
        if i == j:
            error = variance[i] - cov[i, i]


print(cov)
