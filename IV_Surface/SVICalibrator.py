import numpy as np
from scipy import optimize
from VolatilitySurface.Tools import SVITools
from functools import partial

atm_sigma = 0.07
t = 0.5

a = atm_sigma * atm_sigma * t
b = 0.01
rho = 0.5
sigma = 0.1
m = 0.01

f0t = 1.15
strikes = np.array([0.8, 0.9, 1.0, 1.10, 1.20])

no_strikes = len(strikes)
sigmas_t = np.zeros(no_strikes)
w_t = np.zeros(no_strikes)
z_i = np.zeros(no_strikes)

for i in range(0, no_strikes):
    z_i[i] = np.log(strikes[i] / f0t)
    sigmas_t[i] = np.sqrt(SVITools.var_raw_svi(z_i[i], a, b, rho, sigma, m))
    w_t[i] = sigmas_t[i] * sigmas_t[i] * t


max_w_t = np.max(w_t)


p0 = np.array([0.1, 0.0])

f_error = partial(SVITools.get_global_error, forward=f0t,
                  strikes=strikes, sigmas=sigmas_t, t=t)

bounds = ((np.min(z_i), np.max(z_i)), (0.0, 2.0))

p_reduced_i = optimize.minimize(f_error, p0, method='L-BFGS-B', bounds=bounds, constraints=None, jac=None, tol=1e-12)
val_reduced_i = f_error(p_reduced_i.x)

for i in range(0, no_strikes):
    var_model = SVITools.var_raw_svi(z_i[i], SVITools.p_sub_calibrators[0], SVITools.p_sub_calibrators[1],
                                     SVITools.p_sub_calibrators[2], p_reduced_i.x[0], p_reduced_i.x[1])

    var_market = sigmas_t[i] * sigmas_t[i]




