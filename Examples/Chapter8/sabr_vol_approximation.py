import matplotlib.pylab as plt
import numpy as np
from VolatilitySurface.Tools import SABRTools
from AnalyticEngines.MalliavinMethod import ExpansionTools

# parameters
alpha = 0.35
nu = 0.8
rho = -0.5
parameters = np.array([alpha, nu, rho])

# options
strikes = np.arange(60.0, 140.0, 5.0)
t_s = np.arange(0.25, 20, 0.5)
f0 = 100
T = 0.5

iv_approximation = []
iv_sabr = []
for i in range(0, len(strikes)):
    z = np.log(f0 / strikes[i])
    iv_sabr.append(SABRTools.sabr_vol_jit(alpha, rho, nu, z, T))
    iv_approximation.append(ExpansionTools.get_iv_sabr_approximation(parameters, T, strikes[i], f0))

plt.plot(strikes, iv_sabr, label='IV Hagan', linestyle='--', marker='.', color='black')
plt.plot(strikes, iv_approximation, label='IV Approximation', linestyle='--', color='black')

plt.legend()
plt.show()
