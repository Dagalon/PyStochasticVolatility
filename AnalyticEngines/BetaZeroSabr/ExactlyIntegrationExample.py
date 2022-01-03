import EuropeanOptionTools
import numpy as np
import matplotlib.pylab as plt


# parameters
f0 = 0.0435
k = 0.0435
alpha = 0.0068
nu = 0.3691
rho = - 0.0286
t = 10.0

b_min = EuropeanOptionTools.get_b_min(alpha, rho, nu, f0, k)
b_max = 5.0
no_points = 100
b_i_s = np.linspace(b_min, b_max, no_points)
h_i_s = np.zeros(no_points)
g_i_s = np.zeros(no_points)
prod_h_g_i_s = np.zeros(no_points)

for i in range(0, no_points):
    h_i_s[i] = EuropeanOptionTools.h(b_i_s[i], alpha, rho, nu, f0, k)
    g_i_s[i] = EuropeanOptionTools.g(b_i_s[i], nu, t)
    prod_h_g_i_s[i] = h_i_s[i] * g_i_s[i]

plt.plot(b_i_s, h_i_s, label="h*g")
plt.legend()
plt.show()

option_value = EuropeanOptionTools.call_option_price(f0, k, t, alpha, rho, nu)
