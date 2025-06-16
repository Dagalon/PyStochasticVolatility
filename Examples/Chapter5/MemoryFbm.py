import numpy as np
import matplotlib.pylab as plt


def rho_h(n, h):
    return 0.5 * (np.power(n + 1, 2 * h) + np.power(n - 1, 2 * h) - 2 * np.power(n, 2 * h))


ns = np.arange(1, 11, 0.5)

h_lower = 0.2
h_mid = 0.5
h_upper = 0.8

rho_lower = []
rho_mid = []
rho_upper = []

for ni in ns:
    rho_lower.append(rho_h(ni, h_lower))
    rho_mid.append(rho_h(ni, h_mid))
    rho_upper.append(rho_h(ni, h_upper))


plt.plot(ns, rho_lower, label='H=0.1', color='olive', linestyle='dashdot')
plt.plot(ns, rho_mid, label='H=0.2', color='blue', linestyle='dashed')
plt.plot(ns, rho_upper, label='H=0.8', color='orange', linestyle='dashdot')

plt.xlabel('n')
plt.ylabel('rho(n)')

plt.legend()
plt.show()
