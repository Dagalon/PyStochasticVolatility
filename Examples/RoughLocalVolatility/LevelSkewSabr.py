import numpy as np
import matplotlib.pylab as plt

from MC_Engines.MC_LocalVol import LocalVolFunctionals
from VolatilitySurface.Tools import SABRTools

# info
f0 = 100.0
t = 0.5
h = 0.5
epsilon = 1.0
spreads = np.linspace(-40 * epsilon, 40 * epsilon, 81)
strikes = [f0 + s for s in spreads]

# sabr parameters
alpha = 0.3
nu = 0.6
rho = -0.6
parameters = [alpha, nu, rho]


# skew for sabr formula
zi = np.log(np.divide(f0, strikes))

iv_hagan = [SABRTools.sabr_vol_jit(alpha, rho, nu, z, t) for z in zi]

# skew for local vol equivalent
ks = np.array(strikes)

lv_output = LocalVolFunctionals.derivatives_local_vol_log_normal_sabr(t, ks, f0, alpha, rho, nu)
lv_equivalent = [lv_output[i, 0] for i in range(0, len(ks))]

plt.plot(strikes, iv_hagan, label="iv", color="black", marker="o")
plt.plot(strikes, lv_equivalent, label="lv_equivalent", color="black", linestyle="dotted", marker="x")


plt.xlabel("K")
plt.legend()
plt.show()
